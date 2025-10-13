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
from logging.handlers import TimedRotatingFileHandler
import psutil
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from model.tools import get_mapping, DataLoaderX, compute_labels_mahalanobis
from model.ResNet import *
import random
# from model.nt_xent import SupConLoss
import pynvml as nv
from model.plot import plot_confusion_matrix, plot_embedding
from dataset import *
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  # 导入带重启的调度器




def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

import wandb


m1_index = np.array([1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.,  16.,  18.,  19.,  20.,  21.,  22.,
        23.,  24.,  25.,  26.,  27.,  28.,  30.,  31.,  32.,  33.,  34.,
        35.,  36.,  37.,  38.,  39.,  40.,  42.,  43.,  44.,  48.,
        49.,  47.,  51.,  52.,  54.,  55.,  56.,  57.,  61.,  62.,  63.,
        64.,  67.,  72.,  73.,  74.,  75.,  79.,  80.,  89.,  91.,  95.,
        97.,  98., 102., 104., 107., 114., 118., 120., 125., 127.],
      dtype=np.int16)

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss 实现（四分类）
        :param alpha: 类别权重向量（长度=4），若为标量则所有类别权重相同
        :param gamma: 聚焦参数（γ），控制难样本权重（γ↑ → 更关注困难样本）
        :param reduction: 损失归约方式（'mean', 'sum', 'none'）
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        # 若 alpha 为标量则扩展为四分类权重向量
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha] * 4)
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        
        if self.alpha is not None:
            self.alpha = self.alpha.float()

    def forward(self, inputs, targets):
        """
        :param inputs: 模型原始输出（未softmax），形状 [batch, 4]
        :param targets: 真实类别标签（非one-hot），形状 [batch]
        """
        # 计算交叉熵损失（不归约）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # [batch]
        
        # 计算真实类别对应的预测概率 p_t
        pred_prob = F.softmax(inputs, dim=1)  # [batch, 4]
        p_t = pred_prob.gather(1, targets.unsqueeze(1)).squeeze()  # [batch]
        
        # 计算调制因子 (1 - p_t)^γ
        modulating_factor = (1 - p_t) ** self.gamma
        
        # 计算最终损失（基础为交叉熵）
        loss = modulating_factor * ce_loss
        
        # 添加类别权重 alpha
        if self.alpha is not None:
            alpha_weight = self.alpha[targets].to(inputs.device)  # 按标签选择对应权重
            loss = alpha_weight * loss
        
        # 归约损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
        

class ResNetPipeline:
    def __init__(self, args):

        nv.nvmlInit()
        torch.random.manual_seed(args.seed)
        self.args = args
        self.exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_paths()
        self.action_id = json.load(open(os.path.join(args.root, 'action_id.json')))[args.action_id]
        self.ele_mapping, _, _ = get_mapping(args.root)
        if self.args.mode == '3d':
            self.ele_mapping = torch.arange(0,128, dtype=torch.long) 
        # self.ele_mapping = torch.tensor(m1_index, dtype=torch.long)
        # 初始化核心组件
        if args.info_batch:
            reduction = 'none'
        else:
            reduction = 'mean'

        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction=reduction)
        # self.criterion = FocalLoss(
        #                 alpha=1,  # 各类别权重（按索引0~3）
        #                 gamma=1,
        #                 reduction='mean'
        #                 )

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
        wandb.login()
        # wandb.init(project="BCI", name=f'{self.exp_id}_{self.args.tag}')
        # wandb.init(mode="disabled")

    def set_model(self, model, optimizer, scheduler, swa_model, swa_scheduler, ace_loss, device):
        """设置模型、优化器和学习率调度器"""
        self.model = model
        self.model.initialize_weights()
        # for i in range(3):
        #     self.reset_model()
        #     print(model.conv1[0].weight.mean().item())
        #     print(model.conv1[0].weight.std().item())
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.swa_scheduler = swa_scheduler
        self.device = device
        if hasattr(self, 'ace_loss'):
            self.ace_loss = ace_loss

    def reset_model(self):
        """重置模型和优化器"""
        seed_everything(42)
        self.model.initialize_weights()
        # 重新创建 optimizer 和 scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,
            T_mult=2,
            eta_min=1e-5,
        )
        # 重置统计变量
        self._time_cost = []
        self._model_checkpoint_path = []

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

    def _logging_gpu_info(self):
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            gpu_info = nv.nvmlDeviceGetHandleByIndex(device)
            gpu_name = nv.nvmlDeviceGetName(gpu_info)
            utilization = nv.nvmlDeviceGetUtilizationRates(gpu_info)
            total_memory = nv.nvmlDeviceGetMemoryInfo(gpu_info).total / (1024 ** 2)
            power_usage = nv.nvmlDeviceGetPowerUsage(gpu_info) / 1000  # 转换为瓦特
            gpu_free = nv.nvmlDeviceGetMemoryInfo(gpu_info).free / (1024 ** 2)
            self._logger.info(
                f"GPU Info: {gpu_name}, Free Memory: {gpu_free:.2f} MB | Utilization: {utilization.gpu}% |Memory: {total_memory:.2f} MB | Power: {power_usage:.2f} W")

    def _logging_cpu_info(self):
        cpu_percent = str(psutil.cpu_percent())
        mem_total = str(psutil.virtual_memory().total / 1024 / 1024 / 1024) + ' GB'
        mem_used = str(psutil.virtual_memory().used / 1024 / 1024 / 1024) + ' GB'
        mem_available = str(psutil.virtual_memory().available / 1024 / 1024 /
                            1024) + ' GB'
        mem_percent = str(psutil.virtual_memory().percent) + ' %'
        self._logger.info(
            f"CPU Info: {cpu_percent}%, Memory Total: {mem_total}, Used: {mem_used}, Available: {mem_available}, Percent: {mem_percent}")

    def _setup_paths(self):
        """创建实验目录结构"""
        self.exp_path = os.path.join(
            self.args.output_root,
            f"ResNet_{self.args.model_depth}_{self.args.kernel_size}",
            self.exp_id
        )
        os.makedirs(self.exp_path, exist_ok=True)

    def _generate_date_pairs(self):
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
        return pairs

    def execute(self, search_space=None):
        """执行主训练流程"""
        # 生成日期组合
        date_pairs = self._generate_date_pairs()[-1:]

        # 主训练循环
        date_pairs = [
            # [[20250401,],[20250402,]],
            # [[20250402,],[20250401,]],
            # [[20250325,],[20250326,20250327,20250329,20250331,20250401]],
            # [[20250325,20250326,20250327,],[20250329,20250331,20250401]],
            # [[20250325,20250326,20250327,20250329,20250331,],[20250401]],
            [[20250323,20250324,20250325,],[20250326,20250327,20250329,20250331,20250401]],
            # [[20250319,],[20250320,20250402,]],
            # [[20250320,],[20250319,20250402,]],
            # [[20250326,],[20250319,20250401,20250320,]],
            # [[20250327,],[20250319,20250401,20250320,]],
            # [[20250331,],[20250319,20250401,20250320,]],
            # [[20250320,],[20250401,20250402,20250325]],
            # [[20250409,],[20250401,20250402,20250325]],
            # [[20250409,20250320],[20250401,20250402,20250325]],
            # [[20250325,
            #     20250326,
            #     20250327,
            #     20250329,
            #     20250331,
            # ],
            #  [20250319,20250401,20250320,20250402]],
            ]
        for i,(train_dates, test_dates) in tqdm(enumerate(date_pairs), desc="Processing date groups"):
            self.reset_model()
            wandb.init(project="BCI", name=f'{self.exp_id}_{self.args.tag}_{train_dates}')
            # 数据加载与预处理
            # train_dates= [20250325,
            #                 # 20250326,
            #                 # 20250327,
            #                 # 20250329,
            #                 # 20250331,
            #                 ]
            # test_dates = [
            #                 20250401,
            #                 20250402,]
            train_dates = [str(i) for i in train_dates]
            test_dates = [str(i) for i in test_dates]
            try:
                date_path = [os.path.join(self.args.data_path, i) for i in train_dates]
                #-----------------v1
                train_dataset, train_valid_dataset, valid_dataset, train_trail, train_labels_trail = load_train_dataset(
                    date_path, windows_size=256, step=32,)
                #-----------------v2
                # train_dataset, train_valid_dataset, valid_dataset, train_trail, train_labels_trail = load_train_dataset_post_resample(date_path, windows_size=self.args.window_size, step=self.args.window_stride,)
                self._logger.info(f'Loaded data for: {train_dates}')
            except DataLoadError as e:
                print(f"Skipping due to error: {str(e)}")
                continue

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
                shuffle=False,
                drop_last=True,
                num_workers=args.num_workers)
            # test_datasets = [load_daily_dataset_post_resample(d, windows_size=self.args.window_size, step=self.args.window_stride) for d in [os.path.join(self.args.data_path, i) for i in test_dates]]
            test_datasets = [load_daily_dataset(d, windows_size=256, step=32) for d in [os.path.join(self.args.data_path, i) for i in test_dates]]

            test_loaders = {
                date:DataLoaderX(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=args.num_workers) for date, dataset in zip(test_dates, test_datasets)
            }
            # 验证当天
            self.train(train_valid_loader, valid_loader, test_loaders)
            wandb.finish()


    def train(self, train_loader, valid_loader, test_loaders=None):
        """训练与验证"""
        train_dates = str(train_loader.dataset)
        test_date = str(valid_loader.dataset) if valid_loader else None
        for epoch in range(self.args.epochs):
            start = time.perf_counter()
            train_loss, train_accuracy = self._train_epoch(epoch, train_loader)
            cost = time.perf_counter() - start
            self._logger.info(f'Training time: {cost:.2f}s')
            self._time_cost.append(cost)
            self.scheduler.step()
            if epoch % self.args.eval_interval == 0:
                # 验证阶段
                metrics = None
                if valid_loader is not None:
                    valid_loss, valid_accuracy, metrics = self._evaluate_model(valid_loader, epoch)
                    self._logger.info(
                        f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                        f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}, Valid F1: {metrics['f1']:.4f}, ")
                    wandb.log({
                        'epoch': epoch,
                        'valid_loss': valid_loss,
                        'valid_accuracy': valid_accuracy,
                        'valid_f1': metrics['f1'],
                    },step=epoch
                    )
                filename = '{}-{}'.format(train_dates, epoch)
                self._save_checkpoint(epoch, filename, metrics, train_dates, test_date)

            if epoch % self.args.eval_interval == 0:
                for test_date,test_loader in test_loaders.items():
                    val_accs = []
                    val_loss, val_acc, metrics = self._evaluate_model(test_loader, epoch)
                    val_accs.append(val_acc)
                    self._logger.info(
                        f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc:.4f}, Test F1: {metrics['f1']:.4f}, ")
                    train_name = train_dates
                    self._save_checkpoint(epoch, None, metrics, train_name, test_date)
                    wandb.log(
                        {
                            'epoch':int(epoch),
                            f'{test_date}_acc':val_acc,
                            f'{test_date}_loss':val_loss,
                            f'{test_date}_f1':metrics['f1'],
                        },step=epoch
                    )
            self._logging_gpu_info()
        self._logger.info(
            f'Training time : {np.mean(self._time_cost, axis=0):.2f}±{np.std(self._time_cost, axis=0):.2f}s')

    def _train_epoch(self, epoch, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.args.epochs}"):
            X_batch = X_batch.to(self.device)[:, self.ele_mapping]
            y_batch = y_batch.to(self.device)
            # 前向传播
            X_batch = self.model.add_noise(X_batch)
            outputs_list, features = self.model(X_batch)
            # loss = self.criterion(outputs, y_batch)
            loss_list = torch.stack([self.criterion(i, y_batch) for i in outputs_list])
            loss = loss_list.mean()
            outputs = outputs_list[-1]
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            optimizer.step()
            # 更新统计数据
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        self._logger.info(
            f"Epoch {epoch}: Train Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {accuracy:.4f}")
        metrics = {
            'epoch': epoch,
            'train_loss': total_loss / len(train_loader),
            'train_accuracy': accuracy,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'loss1': loss_list[0].item(),
            'loss2': loss_list[1].item(),
            'loss3': loss_list[2].item(),
            'loss4': loss_list[3].item(),
        }
        wandb.log(metrics,step=epoch)
        return total_loss / len(train_loader), accuracy

    def _evaluate_model(self, test_loader, epoch):
        total_val_loss = 0
        correct = 0
        total = 0
        all_pred, all_labels = [], []
        feature_vector = []
        self.model.eval()
        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val = X_val.to(self.device)[:, self.ele_mapping]
                y_val = y_val.to(self.device)
                output, features = self.model(X_val)
                output = output[-1] #...........
                val_loss = self.criterion(output, y_val)
                feature_vector.append(features.view(features.shape[0], -1).cpu().numpy())
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
            'epoch': epoch,
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
            torch.save(self.model.state_dict(), model_name)
            self._model_checkpoint_path.append(model_name)

        if metrics is not None:
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




def setup_model(args, search_space=None):
    """设置模型"""
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
        # model = Simple1d(
        #     in_channels=args.input_dim,

        #     c1_kernel_num=64,
        #     c1_kernel_size=25,
        #     c1_stride= 5,
            
        #     c2_kernel_num=128,
        #     c2_kernel_size=3,
        #     c2_stride= 2,

        #     c3_kernel_num=64,
        #     c3_kernel_size=3,
        #     c3_stride= 2,
        #     activation='elu',
        #     n_classes=args.num_classes,
        # ).to(device)
        # model = ResNetv2(
        #     128, 64
        # ).to(device)
        model = MultiScale1DCNN_v2(128,out_channels=64, num_class=4, scales=[5,15,25]).to(device)
        # model = Spatial3dconvMultiscale_v2_Att(scale1_small=[5,5,15], scale1_mid=[8,8,15], scale1_large=[10,10,25]).to(device)
        # model = ResNetv2_group(128,128).to(device)
        # model = Spatial3dconvMultiscale_v2_sep(3,3).to(device)
        # model = Spatial3dconvMultiscale(3,3).to(device)
        # model = ResNet(
        #     in_channels=128,
        #     out_channels=512,
        #     kernel_size=args.kernel_size if search_space is None else search_space['kernel_size'],
        #     n_classes=args.num_classes,
        #     n_layers=args.model_depth if search_space is None else search_space['num_layers'],
        #     first_kernel_size=25

        #     if search_space is None else search_space['first_layer'],
        #     drop_out=0.0 if search_space is None else search_space['drop_out'],
        # ).to(device)
        # model = Spatial3dconv(128,256).to(device)
        # model = Spatial3dconv(
        #             in_channels=args.input_dim,
        #             out_channels=args.base_filters if search_space is None else search_space['out_channels'],
        #             kernel_size=args.kernel_size if search_space is None else search_space['kernel_size'],
        #             n_classes=args.num_classes,
        #             n_layers=args.model_depth if search_space is None else search_space['num_layers'],
        #             first_kernel_size=25
        # ).to(device)
    if args.ace_loss:
        embedding_size = args.base_filters if search_space is None else search_space['out_channels']
        if args.supcon_loss:
            embedding_size *= 2
        ace_loss = ArcFace(num_classes=args.num_classes,
                           embedding_size=embedding_size,
                           reduction='none' if args.info_batch else 'mean').to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=args.epochs,
    #     eta_min=args.epochs * 0.01,
    # )
    # 替换原有的 CosineAnnealingLR
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,                # 第一次重启的周期长度（单位：epoch）
        T_mult=2,               # 重启周期倍增因子（每次重启后周期长度翻倍）
        eta_min=1e-5,           # 最小学习率（固定为极小值，避免归零）
    )
    ema_model = AveragedModel(model, device=device)

    swa_scheduler = SWALR(optimizer, swa_lr=args.learning_rate)
    return model, optimizer, scheduler, ema_model, swa_scheduler, None if not args.ace_loss else ace_loss, device


def parse_args():
    """参数配置"""
    parser = argparse.ArgumentParser(description='ResNet1D训练系统')

    # 数据参数
    parser.add_argument('--root', type=str, default='/mnt/c/gaochao/CODE/BCI/XZD4class/shared_BDY_S01',
                        help='脑电数据根目录')

    parser.add_argument('--data_path', type=str,
                        default='/mnt/c/gaochao/CODE/BCI/daily_bdy_20250908',
                        # default='/mnt/ata-aigo_SSD/Dataset/tiantan/S01/daily',
                        help='脑电数据根目录')
    parser.add_argument('--dates', nargs='+',
                        # default=[
                            # '20250428_0429_0430_right_elbow',
                            # '20250506',
                            # '20250507',
                            # '20250508',
                            # '20250513',
                            # '20250516',
                            # '20250521',
                            # '20250522',
                            # '20250523',
                            # '20250526',
                            # '20250530_cl',
                            # '20250603',
                            # '20250604',
                            # '20250605',
                            # '20250606',
                            # '20250609',
                            # '20250610',
                        # ],
                        # default=[
                        #     '20250530_8_directions',
                        #     '20250603_8_directions',
                        #     '20250604_8_directions'
                        # ],
                        # default=[
                        # '20250520',
                        # '20250521',
                        # '20250522',
                        # '20250523',
                        # '20250526',
                        # '20250527',
                        # '20250528',
                        # '20250529',
                        # '20250604',
                        # '20250606',
                        # '20250610',
                        # ],
                        # default=['20250323',
                        #          '20250324',
                        #          '20250325',
                        #          '20250326',
                        #          '20250327',
                        #          '20250329',
                        #          '20250331',
                        #          '20250401', ],
                        default=[
                            20250325,
                            20250326,
                            20250327,
                            20250329,
                            # 20250331,
                            # 20250401,
                            # 20250402,
                        ],
                        # default=['daily_train', 'daily_test'],
                        # default=['train_0323_0331', 'test_0401'],
                        # default=['single', 'dual'],
                        help='实验日期列表，按时间顺序排列')
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=4)

    # 日期组合参数
    parser.add_argument('--date_window', type=int, default=1,
                        help='训练日期窗口大小')
    parser.add_argument('--date_step', type=int, default=1,
                        help='日期窗口滑动步长')

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
    parser.add_argument('--learning_rate', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--scheduler', default='cosine', choices=['step', 'cosine', 'plateau'])
    parser.add_argument('--scheduler_args', type=json.loads, default='{"eta_min": 1e-4}', )

    # 训练配置
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--window_size', type=int, default=600)
    parser.add_argument('--window_stride', type=int, default=60)
    parser.add_argument("--eval_interval", type=int, default=5, help="Evaluation interval")
    parser.add_argument("--swa_start", type=float, default=0.1, help="SWA开始训练比例")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="标签平滑系数")

    # 系统参数
    parser.add_argument('--action_id', type=str, default='tiantan', help='Use today\'s date for validation')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_root', type=str,
                        # default='contra_resnet_results_compare'
                        default='daily_resnet_results'
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
    parser.add_argument('--tag', type=str, default='multiscale1d_v2_2345 new_xzd_data_m1s1')
    parser.add_argument('--mode', type=str, default='1d',help='set 3d if use 3d ele_mapping')
    return parser.parse_args()


if __name__ == '__main__':
    seed_everything()
    args = parse_args()

    if not args.ray_tune:
        trainer = ResNetPipeline(args)
        model, optimizer, scheduler, ema_model, swa_scheduler, ace_loss, device = setup_model(args)
        print(model)
        trainer.set_model(model, optimizer, scheduler, ema_model, swa_scheduler, ace_loss, device)
        trainer.execute()
    else:
        search(args)
