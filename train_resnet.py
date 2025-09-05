"""
基于ResNet的深度神经网络分类训练系统
整合功能：
1. 模块化分层架构
2. 超参数全局配置
3. 混合精度训练
4. 分布式数据并行
5. 学习率策略
6. 动态数据增强
"""

import os
import json
import time
import torch
import logging
import argparse
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pynvml as nv
from logging.handlers import TimedRotatingFileHandler

from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from model.tools import get_mapping, DataLoaderX, compute_labels_mahalanobis
from model.ResNet import ResNet, ArcFace, TimeFFTResNet
# from model.nt_xent import SupConLoss
from model.plot import plot_confusion_matrix, plot_embedding
from dataset import load_daily_dataset, load_train_dataset

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


class ResNetPipeline:
    def __init__(self, args):

        # nv.nvmlInit()
        torch.random.manual_seed(args.seed)
        self.args = args
        self.exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_paths()
        self.action_id = json.load(open(os.path.join(args.root, 'action_id.json')))[args.action_id]
        self.ele_mapping, _, _ = get_mapping(args.root)

        # 初始化核心组件
        if args.info_batch:
            reduction = 'none'
        else:
            reduction = 'mean'

        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction=reduction)
        # if args.supcon_loss:
        #     self.supcon_loss = SupConLoss(temperature=0.1, base_temperature=0.1, reduction=reduction)

        if args.ace_loss:
            self.ace_loss = None

        self._model_checkpoint_path = []

        self._time_cost = []
        self._logger = self._setup_logging()

        self.optimizer = None
        self.model = None
        self.search_space = None
        self.swa_model = None
        self.swa_scheduler = None
        self.device = None

    def set_model(self, model, optimizer, scheduler, swa_model, swa_scheduler, ace_loss, device):
        """设置模型、优化器和学习率调度器"""
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.swa_scheduler = swa_scheduler
        self.device = device
        if hasattr(self, 'ace_loss'):
            self.ace_loss = ace_loss

    def _setup_logging(self):
        """设置日志记录"""
        formatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s|%(message)s')
        file_handler = TimedRotatingFileHandler(os.path.join(self.exp_path, f'{self.exp_id}.log'))
        file_handler.setFormatter(formatter)
        memory_handler = logging.handlers.MemoryHandler(1024, flushLevel=logging.DEBUG,
                                                        target=file_handler,
                                                        flushOnClose=True)

        memory_handler.setLevel(logging.INFO)
        memory_handler.setFormatter(formatter)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(memory_handler)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        logger.info(f"Experiment ID: {self.exp_id}")
        logger.info(f"Arguments: {json.dumps(vars(self.args), indent=4)}")
        return logger

    # def _logging_gpu_info(self):
    #     if torch.cuda.is_available():
    #         device = torch.cuda.current_device()
    #         gpu_info = nv.nvmlDeviceGetHandleByIndex(device)
    #         gpu_name = nv.nvmlDeviceGetName(gpu_info)
    #         utilization = nv.nvmlDeviceGetUtilizationRates(gpu_info)
    #         total_memory = nv.nvmlDeviceGetMemoryInfo(gpu_info).total / (1024 ** 2)
    #         power_usage = nv.nvmlDeviceGetPowerUsage(gpu_info) / 1000  # 转换为瓦特
    #         gpu_free = nv.nvmlDeviceGetMemoryInfo(gpu_info).free / (1024 ** 2)
    #         self._logger.info(
    #             f"GPU Info: {gpu_name}, Free Memory: {gpu_free:.2f} MB | Utilization: {utilization.gpu}% |Memory: {total_memory:.2f} MB | Power: {power_usage:.2f} W")

    # def _logging_cpu_info(self):
    #     cpu_percent = str(psutil.cpu_percent())
    #     mem_total = str(psutil.virtual_memory().total / 1024 / 1024 / 1024) + ' GB'
    #     mem_used = str(psutil.virtual_memory().used / 1024 / 1024 / 1024) + ' GB'
    #     mem_available = str(psutil.virtual_memory().available / 1024 / 1024 /
    #                         1024) + ' GB'
    #     mem_percent = str(psutil.virtual_memory().percent) + ' %'
    #     self._logger.info(
    #         f"CPU Info: {cpu_percent}%, Memory Total: {mem_total}, Used: {mem_used}, Available: {mem_available}, Percent: {mem_percent}")

    def _setup_paths(self):
        """创建实验目录结构"""
        self.exp_path = os.path.join(
            self.args.output_root,
            f"ResNet_{self.args.model_depth}_{self.args.kernel_size}",
            self.exp_id
        )
        os.makedirs(self.exp_path, exist_ok=True)

    def _generate_date_pairs(self):
        if self.args.date_pairs is None:
            """生成训练-测试日期组合"""
            dates = self.args.dates
            window_size = self.args.date_window
            step_size = self.args.date_step

            pairs = []
            for i in range(0, len(dates) - window_size, step_size):
                train_dates = dates[:i + window_size]
                # 训练日期组合
                test_dates = []
                for j in range(i + window_size, len(dates)):
                    test_dates.append(dates[j])
                pairs.append((train_dates, test_dates))
        else:
            pairs = self.args.date_pairs
        return pairs

    def execute(self, search_space=None):
        """执行主训练流程"""
        # 生成日期组合
        date_pairs = self._generate_date_pairs()
        if self.args.svd:
            svd = {'U':np.load(self.args.svd_U), 'rank':self.args.svd_rank}
        else:
            svd = None
        # 主训练循环
        for train_dates, test_dates in tqdm(date_pairs, desc="Processing date groups"):
            # 数据加载与预处理
            try:
                date_path = [os.path.join(self.args.data_path, i) for i in train_dates]
                (train_dataset, train_valid_dataset, valid_dataset,
                 train_trail, train_labels_trail) = load_train_dataset(date_path, windows_size=self.args.window_size,
                    step=self.args.window_stride, include_failure=self.args.include_failure, decorrelate=self.args.decorrelate, svd=svd)
                self._logger.info(f'Loaded data for: {train_dates}')
            except DataLoadError as e:
                print(f"Skipping due to error: {str(e)}")
                continue

            if self.args.today:
                # if args.info_batch:
                #     train_valid_dataset = InfoBatch(train_valid_dataset, args.epochs, args.prune_ratio, args.delta)
                train_valid_loader = DataLoader(
                    train_valid_dataset,
                    batch_size=args.batch_size,
                    shuffle=True if not args.info_batch else False,
                    num_workers=args.num_workers if not args.info_batch else 0,
                    sampler=train_dataset.sampler if args.info_batch else None
                )

                valid_loader = DataLoaderX(
                    valid_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=args.num_workers)

                # 验证当天
                if args.model_path is None:
                    self.train(train_valid_loader, valid_loader)
                    self._logger.info(f'Training today model...')
                else:
                    self._logger.info(f'Loaded model from: {args.model_path}')
                    model_params = torch.load(args.model_path, weights_only=True)
                    self.model.load_state_dict(model_params)
            else:
                # 训练与验证
                # self._model_checkpoint_path = []
                # if args.info_batch:
                #     train_dataset = InfoBatch(train_dataset, args.epochs, args.prune_ratio, args.delta)

                if args.random_sampling and not args.info_batch:
                    # 随机采样
                    indices = np.random.choice(len(train_dataset), size=int(len(train_dataset) * args.prune_ratio),
                                            replace=False)
                    train_dataset = torch.utils.data.Subset(train_dataset, indices)
                    self._logger.info(f'Randomly sampled {len(train_dataset)} samples from the training dataset.')

                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True if not args.info_batch else False,
                    num_workers=args.num_workers if not args.info_batch else 0,
                    sampler=train_dataset.sampler if args.info_batch else None)

                if args.model_path is None:
                    self.train(train_dataloader, None, train_dataset)
                    self._logger.info(f'Training {train_dates} model...')
                else:
                    self._logger.info(f'Loaded model from: {args.model_path}')
                    model_params = torch.load(args.model_path, weights_only=True)
                    self.model.load_state_dict(model_params)


            for test_date in test_dates:
                date_path = os.path.join(self.args.data_path, test_date)
                dataset = load_daily_dataset(date_path,windows_size=self.args.window_size,
                    step=self.args.window_stride, include_failure=self.args.include_failure, decorrelate=self.args.decorrelate, svd=svd)
                test_loader = DataLoaderX(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=args.num_workers)

                if args.model_path is None:
                    self._logger.info(f'Loaded data for: {test_date}')
                    # 测试阶段
                    val_accs = []
                    for epoch, model_path in enumerate(self._model_checkpoint_path):
                        model_params = torch.load(model_path, weights_only=True)  # '/home/wangrp/桌面/tiantan-66.pt'
                        epoch = epoch + 1
                        # model_params = {key.replace('module.', ''): value for key, value in model_params.items() if 'module.' in key}
                        if self.swa_model is not None and epoch >= int(self.args.swa_start * self.args.epochs):
                            # 使用EMA模型
                            self.swa_model.load_state_dict(model_params)
                        else:
                            # 使用普通模型
                            self.model.load_state_dict(model_params)
                        self._logger.info(f'Loaded model from: {model_path}')
                        val_loss, val_acc, metrics = self._evaluate_model(test_loader, epoch)
                        val_accs.append(val_acc)
                        self._logger.info(
                            f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc:.4f}, Test F1: {metrics['f1']:.4f}, ")
                        train_name = '_'.join(train_dates)
                        epoch = model_path.split('-')[-1].split('.')[0]
                        self._save_checkpoint(epoch, None, metrics, train_name, test_date)
                    self._logger.info(f'Model evaluation {test_date} complete.')
                    # if search_space is not None:
                    #     train.report({"accuracy": max(val_accs)})
                else:
                    epoch = os.path.basename(args.model_path).split('-')[-1].split('.')[0]
                    val_loss, val_acc, metrics = self._evaluate_model(test_loader, int(epoch))
                    self._logger.info(
                        f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc:.4f}, Test F1: {metrics['f1']:.4f}, ")
                    train_name = '_'.join(train_dates)
                    self._save_checkpoint('test', None, metrics, train_name, test_date)

    def train(self, train_loader, valid_loader, train_dataset=None):
        """训练与验证"""

        self.model.initialize_weights()
        train_dates = str(train_loader.dataset)
        test_date = str(valid_loader.dataset) if valid_loader else None
        for epoch in range(self.args.epochs):
            start = time.perf_counter()
            train_loss, train_accuracy, train_feature_vector = self._train_epoch(epoch, train_loader, train_dataset=train_dataset)
            cost = time.perf_counter() - start
            self._logger.info(f'Training time: {cost:.2f}s')
            self._time_cost.append(cost)
            if epoch % self.args.eval_interval == 0 and epoch > 0:
                # 验证阶段
                metrics = None
                if valid_loader is not None:
                    valid_loss, valid_accuracy, metrics = self._evaluate_model(valid_loader, epoch)
                    self._logger.info(
                        f"Epoch {epoch + 1}: Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}, Valid F1: {metrics['f1']:.4f}")
                filename = '{}-{}'.format(train_dates, epoch)
                metrics['train_feature_vector'] = train_feature_vector
                self._save_checkpoint(epoch, filename, metrics, train_dates, test_date)

            # self._logging_gpu_info()
        self._logger.info(
            f'Training time : {np.mean(self._time_cost, axis=0):.2f}±{np.std(self._time_cost, axis=0):.2f}s')

    def _train_epoch(self, epoch, train_loader, train_dataset=None):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        feature_vector = []
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}"):
            if self.args.svd:
                X_batch = X_batch.to(self.device)
            else:
                X_batch = X_batch.to(self.device)[:, self.ele_mapping]
            y_batch = y_batch.to(self.device)

            if args.dynamic_random_sampling:
                # 动态随机采样
                indices = np.random.choice(len(X_batch), size=int(len(X_batch) * self.args.prune_ratio), replace=False)
                X_batch = X_batch[indices]
                y_batch = y_batch[indices]

            # 前向传播
            X_batch = self.model.add_noise(X_batch)
            outputs, features = self.model(X_batch)
            feature_vector.append(features.detach().view(features.shape[0], -1).cpu().numpy())

            loss = self.criterion(outputs, y_batch)

            # 计算ArcFace损失
            if hasattr(self, 'ace_loss'):
                arcface_loss = self.ace_loss(features.view(features.shape[0], -1), y_batch) * 0.1
                loss += arcface_loss

            # 计算监督对比损失
            if hasattr(self, 'supcon_loss'):
                supcon_loss = self.supcon_loss(features, y_batch)
                loss += supcon_loss

            # infoBatch训练
            if self.args.info_batch and train_dataset is not None:
                loss = train_dataset.update(loss)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            self.optimizer.step()

            # 更新EMA模型
            if self.swa_model is not None and epoch >= int(self.args.swa_start * self.args.epochs):
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()

            # 更新统计数据
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        if self.swa_model is not None and epoch >= int(self.args.swa_start * self.args.epochs):
            update_bn(train_loader, self.swa_model, device=self.device)
            lr = self.swa_scheduler.get_last_lr()[0]
        else:
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
        accuracy = correct / total
        feature_vector = np.concatenate(feature_vector, axis=0)
        self._logger.info(
            f"Epoch {epoch + 1}: Train Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {accuracy:.4f}, LR: {lr:.6f}")
        return total_loss / len(train_loader), accuracy, feature_vector

    def _evaluate_model(self, test_loader, epoch):
        total_val_loss = 0
        correct = 0
        total = 0
        all_pred, all_labels = [], []
        feature_vector = []
        self.model.eval()
        if self.swa_model is not None and epoch >= int(self.args.swa_start * self.args.epochs):
            self.swa_model.eval()
        with torch.no_grad():
            for X_val, y_val in test_loader:
                if self.args.svd:
                    X_val = X_val.to(self.device)
                else:
                    X_val = X_val.to(self.device)[:, self.ele_mapping]
                y_val = y_val.to(self.device)
                output, features = self.swa_model(X_val) if self.swa_model is not None and epoch >= int(
                    self.args.swa_start * self.args.epochs) else self.model(X_val)
                # output, features = self.model(X_val)
                val_loss = self.criterion(output, y_val)
                feature_vector.append(features.view(features.shape[0], -1).cpu().numpy())
                # 计算ArcFace损失
                # if hasattr(self, 'ace_loss'):
                #     arcface_loss = self.ace_loss(features.view(features.shape[0], -1), y_val) * 0.1
                #     val_loss += arcface_loss
                #
                # if hasattr(self, 'supcon_loss'):
                #     supcon_loss = self.supcon_loss(features, y_val)
                #     val_loss += supcon_loss
                if self.args.info_batch:
                    val_loss = val_loss.mean()
                total_val_loss += val_loss.item()
                preds = torch.argmax(output, dim=1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)

                all_pred.append(preds.cpu().numpy())
                all_labels.append(y_val.cpu().numpy())

        val_loss = total_val_loss / len(test_loader)
        val_acc = correct / total
        y_pred = np.concatenate(all_pred)
        y_test = np.concatenate(all_labels)
        feature_vector = np.concatenate(feature_vector, axis=0)

        # compute_labels_mahalanobis(feature_vector, y_test)
        distance = {'mean': 0}

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'cm': confusion_matrix(y_test, y_pred),
            'mahalanobis': distance['mean'],
            'feature_vector': feature_vector,
            'y_pred': y_pred,
            'y_test': y_test,
        }

        return val_loss, val_acc, metrics

    def _save_checkpoint(self, epoch, filename, metrics, train_dates, test_date):
        """模型保存"""
        if test_date is not None:
            save_path = os.path.join(self.exp_path, 'checkpoints', train_dates, test_date)
            cm_path = os.path.join(self.exp_path, 'confusion_matrix', train_dates, test_date)
        else:
            save_path = os.path.join(self.exp_path, 'checkpoints', train_dates)
            cm_path = os.path.join(self.exp_path, 'confusion_matrix', train_dates)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(cm_path, exist_ok=True)

        if filename is not None:
            model_name = os.path.join(save_path, f'{filename}.pt')
            if self.swa_model is not None and epoch >= int(self.args.swa_start * self.args.epochs):
                # 保存EMA模型
                torch.save(self.swa_model.state_dict(), model_name)
            else:
                torch.save(self.model.state_dict(), model_name)
            self._model_checkpoint_path.append(model_name)

        if metrics is not None:
            torch.save(metrics, os.path.join(save_path, f"metrics_{train_dates}→{test_date}_{epoch}.pth"))
            # 保存混淆矩阵
            plot_confusion_matrix(
                metrics['cm'],
                labels=list(self.action_id.values()),
                title=f'ResNet {epoch} {train_dates}→{test_date} accuracy: {metrics["accuracy"]:.4f} f1: {metrics["f1"]:.4f}',
                fig_path=os.path.join(
                    cm_path,
                    f"cm_{train_dates}_{test_date}_{epoch}.png"
                )
            )
            plot_embedding(metrics['feature_vector'],
                           metrics['y_test'],
                           metrics['y_pred'],
                           metrics['mahalanobis'],
                           np.round(metrics['accuracy'], 4),
                           os.path.join(cm_path,
                                        f'embedding_{train_dates}_{test_date}_{epoch}.png'
                                        ),
                           action_id=self.action_id)


class DataLoadError(Exception):
    """自定义数据加载异常"""
    pass


def hyperparameter_training(search_space, args):
    """
    ray.tune可调用的训练函数，用于超参数搜索
    :param config: 超参数搜索空间配置
    :param args: 全局配置参数
    :return: 无，通过train.report报告指标
    """
    # 初始化训练管道
    pipeline = ResNetPipeline(args)
    # 执行训练流程，传入超参数配置

    model, optimizer, scheduler, ema_model, swa_scheduler, ace_loss, device = setup_model(args, search_space)

    pipeline.set_model(model, optimizer, scheduler, ema_model, swa_scheduler, ace_loss, device)

    pipeline.execute(search_space=search_space)


# def search(args):
#     search_space = {
#         "weight_decay": tune.loguniform(1e-4, 1e-2),
#         "betas_1": tune.loguniform(0.8, 0.99),
#         "betas_2": tune.loguniform(0.9, 0.999),
#         "num_layers": tune.randint(1, 8),
#         "first_layer": tune.choice(list(range(15, 32, 1))),
#         "drop_out": tune.choice([0.0, 0.1, 0.2]),
#         "out_channels": tune.choice(list(range(256, 1024, 128))),
#         "kernel_size": tune.choice([3, 5, 7]),
#     }

#     algo = OptunaSearch()
#     algo = ConcurrencyLimiter(algo, max_concurrent=args.ray_num_cpu)
#     os.makedirs(args.ray_temp, exist_ok=True)
#     os.makedirs(args.ray_save_path, exist_ok=True)
#     ray.init(_temp_dir=args.ray_temp)
#     trainable_with_resources = tune.with_resources(lambda cfg: hyperparameter_training(cfg, args),
#                                                    resources={"cpu": args.ray_num_cpu, "gpu": args.ray_num_gpu})
#     tuner = tune.Tuner(
#         trainable_with_resources,
#         tune_config=tune.TuneConfig(
#             metric="accuracy",
#             mode="max",
#             search_alg=algo,
#             num_samples=10,
#         ),
#         run_config=train.RunConfig(
#             stop={"training_iteration": 5},
#             storage_path=args.ray_save_path,
#         ),
#         param_space=search_space,
#     )
#     results = tuner.fit()
#     # save results
#     print("Best config is:", results.get_best_result().config)
#     os.makedirs(args.ray_save_path, exist_ok=True)
#     file_name = f'{datetime.now()}.xlsx'
#     results.get_dataframe().to_excel(os.path.join(args.ray_save_path, file_name))


def setup_model(args, search_space=None):
    """设置模型"""
    # device = torch.device('cpu')
    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
    if args.supcon_loss:
        model = TimeFFTResNet(
            in_channels=args.input_dim,
            out_channels=args.base_filters if search_space is None else search_space['out_channels'],
            kernel_size=args.kernel_size if search_space is None else search_space['kernel_size'],
            n_classes=args.num_classes,
            n_layers=args.model_depth if search_space is None else search_space['num_layers'],
            first_kernel_size=25
            if search_space is None else search_space['first_layer'],
            drop_out=0.0 if search_space is None else search_space['drop_out'],
        ).to(device)
    else:
        model = ResNet(
            in_channels=args.svd_rank if args.svd else args.input_dim,
            out_channels=args.base_filters if search_space is None else search_space['out_channels'],
            kernel_size=args.kernel_size if search_space is None else search_space['kernel_size'],
            n_classes=args.num_classes,
            n_layers=args.model_depth if search_space is None else search_space['num_layers'],
            first_kernel_size=25

            if search_space is None else search_space['first_layer'],
            drop_out=0.0 if search_space is None else search_space['drop_out'],
        ).to(device)

    if args.ace_loss:
        embedding_size = args.base_filters if search_space is None else search_space['out_channels']
        if args.supcon_loss:
            embedding_size *= 2
        ace_loss = ArcFace(num_classes=args.num_classes,
                           embedding_size=embedding_size,
                           reduction='none' if args.info_batch else 'mean').to(device)

    optimizer = torch.optim.AdamW(
        model.parameters() if not args.ace_loss else list(model.parameters()) + list(ace_loss.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay if search_space is None else search_space['weight_decay'],
        betas=(0.9, 0.999) if search_space is None else (
            search_space['betas_1'],
            search_space['betas_2']
        )
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs*args.swa_start,
        eta_min=args.learning_rate * 0.01,
    )

    ema_model = AveragedModel(model, device=device)

    swa_scheduler = SWALR(optimizer, swa_lr=args.learning_rate)
    return model, optimizer, scheduler, ema_model, swa_scheduler, None if not args.ace_loss else ace_loss, device


def parse_args():
    """参数配置"""
    parser = argparse.ArgumentParser(description='ResNet1D训练系统')

    # 数据参数
    parser.add_argument('--root', type=str, default='/media/ubuntu/Storage/ecog_data',
                        help='脑电数据根目录')

    parser.add_argument('--data_path', type=str,
                        default='/media/ubuntu/Storage/ecog_data/daily_bdy_new',
                        # default='/media/ubuntu/Storage/ecog_data/preprocessed_removed_wrong_trials',
                        # default='/media/ubuntu/Storage/ecog_data/preprocessed',
                        help='脑电数据根目录')
    parser.add_argument('--dates', nargs='+',
                        default=['20250319',
                                 '20250320',
                                 '20250321',
                                 '20250323',
                                 '20250324',
                                 '20250325',
                                 '20250326',
                                 '20250327',
                                 '20250329',
                                 '20250331',
                                 '20250401',
                                 '20250402',
                                 '20250403',
                                 '20250407',
                                 '20250408',
                                 '20250409'],
                        help='实验日期列表，按时间顺序排列')
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--include_failure', type=bool, default=False, help='No use. Not implemented')
    parser.add_argument('--decorrelate', type=bool, default=False, help='Use SVD to decorrelate the channels of the data when loading the dataset')
    parser.add_argument('--svd_U', type=str, default='/home/ubuntu/ecog_proj/svd/U_20250325_20250326_20250327_20250329_20250331.npy', help='For loading the SVD U matrix from the specified file')
    parser.add_argument('--svd', type=bool, default=True, help='Use SVD to reduce noise when loading the dataset')
    parser.add_argument('--svd_rank', type=int, default=16, help='The rank of the SVD matrix')
    # 日期组合参数
    parser.add_argument('--date_window', type=int, default=1,
                        help='训练日期窗口大小')
    parser.add_argument('--date_step', type=int, default=1,
                        help='日期窗口滑动步长')
    parser.add_argument('--date_pairs', default=
                        # [(['20250401'],['20250402'])])
                        # [(['20250325'], ['20250401'])])
                        [(['20250325','20250326','20250327','20250329','20250331'], ['20250401'])])

    # 模型参数
    parser.add_argument('--model_path', type=str,
                        default=None,
                        # default='/home/wangrp/桌面/test-79.pt',
                        )

    # 模型结构
    parser.add_argument('--model_depth', type=int, default=1)
    parser.add_argument('--base_filters', type=int, default=512)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--groups', type=int, default=32)
    parser.add_argument('--ace_loss', action='store_true', help='使用ArcFace损失')
    parser.add_argument('--supcon_loss', action='store_true', help='使用监督对比损失')

    # 优化策略
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--scheduler', default='cosine', choices=['step', 'cosine', 'plateau'])
    parser.add_argument('--scheduler_args', type=json.loads, default='{"eta_min": 2e-6}', )

    # 训练配置
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--window_size', type=int, default=256)
    parser.add_argument('--window_stride', type=int, default=32)
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluation interval")
    parser.add_argument("--swa_start", type=float, default=1.0, help="SWA开始训练比例")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="标签平滑系数")

    # 系统参数
    parser.add_argument('--action_id', type=str, default='tiantan', help='Use today\'s date for validation')
    parser.add_argument('--today', type=bool, default=True, help='Use today\'s date for validation')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_root', type=str,
                        # default='contra_resnet_results_compare'
                        default='/media/ubuntu/Storage/ecog_data/daily_resnet_results'
                        # default='tiantan_s01'
                        )

    # infoBatch参数
    parser.add_argument('--info_batch', action='store_true', help='Use InfoBatch for training')
    parser.add_argument('--prune_ratio', type=float, default=0.5, help='Prune ratio for InfoBatch')
    parser.add_argument('--delta', type=float, default=0.875, help='Delta value for InfoBatch')

    parser.add_argument('--random_sampling', action='store_true', help='Use random sampling compared to InfoBatch')
    parser.add_argument('--dynamic_random_sampling', action='store_true',
                        help='Use random sampling compared to InfoBatch')

    # 其他参数
    parser.add_argument('--seed', type=int, default=1024, help='Random seed for reproducibility')

    # ray tune参数
    parser.add_argument('--ray_tune', action='store_true', help='Use Ray Tune for hyperparameter optimization')
    parser.add_argument('--ray_save_path', type=str, default='/home/wangrp/ray',
                        help='Ray Tune results storage path')
    parser.add_argument('--ray_temp', type=str, default='/home/wangrp/temp', )
    parser.add_argument('--ray_num_cpu', type=int, default=32, help='Number of CPUs for Ray Tune')
    parser.add_argument('--ray_num_gpu', type=int, default=1, help='Number of GPUs for Ray Tune')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not args.ray_tune:
        trainer = ResNetPipeline(args)
        model, optimizer, scheduler, ema_model, swa_scheduler, ace_loss, device = setup_model(args)
        trainer.set_model(model, optimizer, scheduler, ema_model, swa_scheduler, ace_loss, device)
        trainer.execute()
    # else:
    #     search(args)
