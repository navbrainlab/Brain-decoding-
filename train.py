import os
import numpy as np
from dataset import *
import argparse
import time
import random
import logging
import json
from pyriemann.utils.mean import mean_riemann
import torch
from torch.utils.data import DataLoader
from basenet import BaseNet
from MultiScale import MultiScale1DCNN_v2

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(exp_path, exp_id, args):
    """设置日志记录"""
    formatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s|%(message)s')
    file_handler = logging.FileHandler(os.path.join(exp_path, f'train_{exp_id}.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicate logs and ensure a clean configuration
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    logger.info(f"Experiment ID: {exp_id}")
    logger.info(f"Arguments: {json.dumps(vars(args), indent=4)}")
    return logger
    

def main(args):

    # Generate experiment timestamp
    experiment_timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Create output directory if it doesn't exist
    output_path = os.path.join(args.output_path, experiment_timestamp)
    os.makedirs(output_path, exist_ok=True)
    logger = setup_logging(output_path, experiment_timestamp, args)
    # Save args used for this experiment
    args_path = os.path.join(output_path, 'args.json')
    with open(args_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)
    seed_everything(args.seed)

    # Off line training in the source domains

    # Load data for each source domain
    for source_date in args.source_dates:

        logger.info(f"Training on source domain: {source_date}")
        if 'daily_bdy' in args.data_path:
            if len(args.source_dates) != 1:
                raise ValueError('Single source domain or multiple source domains?')
            logger.info(f"Single source domain training")
            if '_' in source_date:
                source_data_path = [os.path.join(args.data_path, sd) for sd in source_date.split('_')]
            else:
                source_data_path = [os.path.join(args.data_path, source_date)]
            try:
                with open(args.indices_path, 'r', encoding='utf-8') as f:
                    _data = json.load(f)
                indices = np.array(_data['indices']) if _data.get('indices', None) is not None else None
            except Exception as e:
                logger.warning(f"Failed to load indices from {args.indices_path}: {e}. Setting indices=None")
                indices = None
            train_data, train_labels, valid_data, valid_labels, train_date, valid_date, indices = load_centered_data_for_otta(
                source_data_path, windows_size=256, step=args.ecog_step, train_ratio=0.8, return_date=True, indices=indices)
            # train_data: [n_samples, n_channels, n_timepoints], train_labels: [n_samples, n_classes], train_date: [n_samples]
            n_channels = train_data.shape[1]
            n_timepoints = train_data.shape[2]
                
        elif 'imagery_basic' in args.data_path:
            raise NotImplementedError
            source_data_path = os.path.join(args.data_path, source_date+'_im_t_h.mat')
            source_dataset_train, source_dataset_test = load_Stanford_imagery_basic(
                source_data_path, windows_size=1000, step=100, train_ratio=0.8, shuffle=False)
        elif 'imagery_feedback' in args.data_path:
            raise NotImplementedError
            if '&' in source_date:
                if len(args.source_dates) != 1:
                    raise ValueError('Single source domain or multiple source domains?')
                logger.info(f"Single source domain training")
            source_data_path = [os.path.join(args.data_path, sd+'.mat') for sd in source_date.split('&')]
            source_dataset_train, source_dataset_test = load_Stanford_imagery_feedback(
                source_data_path, windows_size=1000, step=100, train_ratio=0.8)
        elif 'bcic' in args.data_path:
            subject_id = int(args.data_path.split('_')[-1])
            if '20250325' in args.indices_path:
                logger.warning("Wrong indices file for BCIC dataset. Setting indices=None")
                indices = None
            else:
                try:
                    with open(args.indices_path, 'r', encoding='utf-8') as f:
                        _data = json.load(f)
                    # Accept either {"indices": [...]} or a bare list
                    if isinstance(_data, dict) and 'indices' in _data:
                        indices = np.array(_data['indices'])
                    elif isinstance(_data, list):
                        indices = np.array(_data)
                    else:
                        logger.warning(f"Unexpected JSON structure in {args.indices_path}. Setting indices=None")
                        indices = None
                except Exception as e:
                    logger.warning(f"Failed to load indices from {args.indices_path}: {e}. Setting indices=None")
                    indices = None
            train_data, train_labels, valid_data, valid_labels, train_date, valid_date, indices = load_centered_bcic(subject_id, session=source_date, return_date=True, indices=indices)
            n_channels = train_data.shape[1]
            n_timepoints = train_data.shape[2]
        elif '.h5' in args.data_path:
            if len(args.source_dates) != 1:
                raise ValueError('Single source domain or multiple source domains?')
            logger.info(f"Single source domain training")
            if '_' in source_date:
                source_dates = [sd for sd in source_date.split('_')]
            else:
                source_dates = [source_date]
            # try:
            #     with open(args.indices_path, 'r', encoding='utf-8') as f:
            #         _data = json.load(f)
            #     indices = np.array(_data['indices']) if _data.get('indices', None) is not None else None
            # except Exception as e:
            #     logger.warning(f"Failed to load indices from {args.indices_path}: {e}. Setting indices=None")
            #     indices = None
            train_data, train_labels, valid_data, valid_labels, train_date, valid_date, indices = load_centered_new_data_for_otta(
                args.data_path, source_dates, train_ratio=0.8, return_date=True, indices=None, class_num=args.new_data_n_classes)
            # train_data: [n_samples, n_channels, n_timepoints], train_labels: [n_samples, n_classes], train_date: [n_samples]          
            n_channels = train_data.shape[1]
            n_timepoints = train_data.shape[2]
        else:
            raise ValueError("Unrecognized dataset")


    # Data alignment in the source domain

    if args.alignment == 'Riemannian':
        # Compute covariance matrices
        cov_matrices = np.array([np.cov(sample) for sample in train_data])  # shape: [n_samples, n_channels, n_channels]

        # Calculate the Riemannian means of train data from different dates for alignment
        # Here train_date is a numpy array of dates corresponding to each sample
        unique_dates = np.unique(train_date)
        for date in unique_dates:
            mask = (train_date == date)
            date_covs = cov_matrices[mask] # shape: [n_date_samples, n_channels, n_channels]
            date_mean = mean_riemann(date_covs) # shape: [n_channels, n_channels]
            # Alignment: X_aligned = M_date^{-1/2} * X
            eigvals_dm, eigvecs_dm = np.linalg.eigh(date_mean)
            eigvals_dm = np.clip(eigvals_dm, a_min=1e-8, a_max=None)
            M_date_inv_sqrt = eigvecs_dm @ np.diag(1.0 / np.sqrt(eigvals_dm)) @ eigvecs_dm.T # shape: [n_channels, n_channels]
            train_data[mask] = M_date_inv_sqrt[None, :, :] @ train_data[mask] # [n_date_samples, n_channels, n_timepoints]
        logger.info(f"Training set Riemannian alignment complete")

        # Align validation data
        valid_cov_matrices = np.array([np.cov(sample) for sample in valid_data])  # shape: [n_valid_samples, n_channels, n_channels]
        for date in unique_dates:
            mask = (valid_date == date)
            date_covs = valid_cov_matrices[mask]
            date_mean = mean_riemann(date_covs)
            # Alignment: X_aligned = M_date^{-1/2} * X
            eigvals_dm, eigvecs_dm = np.linalg.eigh(date_mean)
            eigvals_dm = np.clip(eigvals_dm, a_min=1e-8, a_max=None)
            M_date_inv_sqrt = eigvecs_dm @ np.diag(1.0 / np.sqrt(eigvals_dm)) @ eigvecs_dm.T
            valid_data[mask] = M_date_inv_sqrt[None, :, :] @ valid_data[mask] # [n_date_samples, n_channels, n_timepoints]
        logger.info(f"Validation set Riemannian alignment complete")

    elif args.alignment == 'Euclidean':
        cov_matrices = np.array([np.cov(sample) for sample in train_data])  # shape: [n_samples, n_channels, n_channels]
        unique_dates = np.unique(train_date)
        for date in unique_dates:
            mask = (train_date == date)
            date_covs = cov_matrices[mask]
            date_mean = np.mean(date_covs, axis=0)
            # Alignment: X_aligned = M_date^{-1/2} * X
            eigvals_dm, eigvecs_dm = np.linalg.eigh(date_mean)
            eigvals_dm = np.clip(eigvals_dm, a_min=1e-8, a_max=None)
            M_date_inv_sqrt = eigvecs_dm @ np.diag(1.0 / np.sqrt(eigvals_dm)) @ eigvecs_dm.T # shape: [n_channels, n_channels]
            train_data[mask] = M_date_inv_sqrt[None, :, :] @ train_data[mask] # [n_date_samples, n_channels, n_timepoints]
        logger.info(f"Training set Euclidean alignment complete")

        # Align validation data
        valid_cov_matrices = np.array([np.cov(sample) for sample in valid_data])  # shape: [n_valid_samples, n_channels, n_channels]
        for date in unique_dates:
            mask = (valid_date == date)
            date_covs = valid_cov_matrices[mask]
            date_mean = np.mean(date_covs, axis=0)
            # Alignment: X_aligned = M_date^{-1/2} * X
            eigvals_dm, eigvecs_dm = np.linalg.eigh(date_mean)
            eigvals_dm = np.clip(eigvals_dm, a_min=1e-8, a_max=None)
            M_date_inv_sqrt = eigvecs_dm @ np.diag(1.0 / np.sqrt(eigvals_dm)) @ eigvecs_dm.T
            valid_data[mask] = M_date_inv_sqrt[None, :, :] @ valid_data[mask] # [n_date_samples, n_channels, n_timepoints]
        logger.info(f"Validation set Euclidean alignment complete")
    else:
        raise ValueError("Unrecognized alignment method")
    

    # Train source domain model

    # Define the model, loss function, optimizer here
    logger.info(f"Starting source domain training for {args.epochs} epochs")
    if args.model_name == 'BaseNet':
        model = BaseNet(input_window_samples=n_timepoints,
                        n_channels=n_channels,
                        n_temporal_filters=args.n_temporal_filters,
                        temp_filter_length_inp=args.temp_filter_length_inp,
                        spatial_expansion=args.spatial_expansion,
                        pool_length_inp=args.pool_length_inp,
                        pool_stride_inp=args.pool_stride_inp,
                        dropout_inp=args.dropout_inp,
                        ch_dim=args.ch_dim,
                        temp_filter_length=args.temp_filter_length,
                        pool_length=args.pool_length,
                        pool_stride=args.pool_stride,
                        dropout=args.dropout,
                        n_classes=train_labels.shape[1],
                        use_feedforward=args.use_feedforward)
    elif args.model_name == 'MultiScale1DCNN_v2':
        model = MultiScale1DCNN_v2(in_channels=n_channels, num_class=train_labels.shape[1])
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    
    model.to(args.device)
    if args.criterion == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.criterion == 'MSE':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unsupported criterion: {args.criterion}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)
    
    # Construct dataset and dataloader:
    train_dataset = XZD_Dataset(train_data, train_labels, name='source_train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False) # Already shuffled in data loading
    valid_dataset = XZD_Dataset(valid_data, valid_labels, name='source_valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Record and save training/validation accuracy and loss history.
    # Save best model based on validation accuracy, along with its validation predictions.
    train_acc_history = []
    valid_acc_history = []
    train_loss_history = []
    valid_loss_history = []
    best_valid_acc = 0.0
    best_model_path = os.path.join(output_path, f"best_model_{source_date}.pt")
    best_val_preds_path = os.path.join(output_path, f"best_val_predictions_{source_date}.npz")

    for epoch in range(args.epochs):
        # Training loop
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_data, batch_labels in train_dataloader:
            batch_data = batch_data.to(args.device)
            batch_labels = batch_labels.to(args.device)
            targets = torch.argmax(batch_labels, dim=1)
            optimizer.zero_grad()
            outputs = model(batch_data)
            if isinstance(outputs, tuple):
                 outputs, _ = outputs
                 outputs = outputs[-1]
            loss = criterion(outputs, batch_labels.float())
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                running_loss += loss.item() * targets.size(0)

        train_loss_epoch = running_loss / max(total, 1)
        train_acc_epoch = correct / max(total, 1)
        train_loss_history.append(train_loss_epoch)
        train_acc_history.append(train_acc_epoch)

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_probs = []
        all_val_targets = []
        with torch.no_grad():
            for vdata, vlabels in valid_dataloader:
                vdata = vdata.to(args.device)
                vlabels = vlabels.to(args.device)
                vtargets = torch.argmax(vlabels, dim=1)
                voutputs = model(vdata)
                if isinstance(voutputs, tuple):
                    voutputs, _ = voutputs
                    voutputs = voutputs[-1]
                vloss = criterion(voutputs, vlabels.float())

                vpreds = torch.argmax(voutputs, dim=1)
                vprobs = torch.softmax(voutputs, dim=1)

                val_correct += (vpreds == vtargets).sum().item()
                val_total += vtargets.size(0)
                val_running_loss += vloss.item() * vtargets.size(0)

                all_val_preds.append(vpreds.detach().cpu().numpy())
                all_val_probs.append(vprobs.detach().cpu().numpy())
                all_val_targets.append(vtargets.detach().cpu().numpy())

        valid_loss_epoch = val_running_loss / max(val_total, 1)
        valid_acc_epoch = val_correct / max(val_total, 1)
        valid_loss_history.append(valid_loss_epoch)
        valid_acc_history.append(valid_acc_epoch)

        logger.info(f"Epoch {epoch+1:03d}/{args.epochs} | Train Loss: {train_loss_epoch:.4f} Acc: {train_acc_epoch:.4f} | Valid Loss: {valid_loss_epoch:.4f} Acc: {valid_acc_epoch:.4f}")

        # Save best model and its validation predictions
        if valid_acc_epoch > best_valid_acc:
            best_valid_acc = valid_acc_epoch
            torch.save(model.state_dict(), best_model_path)
            val_preds_np = np.concatenate(all_val_preds, axis=0)
            val_probs_np = np.concatenate(all_val_probs, axis=0)
            val_targets_np = np.concatenate(all_val_targets, axis=0)
            np.savez(best_val_preds_path,
                        pred_classes=val_preds_np,
                        pred_probs=val_probs_np,
                        targets=val_targets_np,
                        best_valid_acc=best_valid_acc)
            logger.info(f"New best model saved (Acc: {best_valid_acc:.4f}) to {best_model_path}")

        scheduler.step()

    # Save training history per source_date
    history_path = os.path.join(output_path, f"training_history_{source_date}.json")
    history = {
        'train_acc': [float(x) for x in train_acc_history],
        'valid_acc': [float(x) for x in valid_acc_history],
        'train_loss': [float(x) for x in train_loss_history],
        'valid_loss': [float(x) for x in valid_loss_history],
        'best_valid_acc': float(best_valid_acc),
        'best_model_path': best_model_path,
        'best_val_preds_path': best_val_preds_path,
        # Guard for None indices
        'indices': indices.tolist() if indices is not None else None,
        'train_ratio': 0.8
    }
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    # Generate an adaptation config file for adapt.py so you don't need to set its args manually
    adapt_config = {
        # Path where adapt.py can find the trained model and source args (args.json)
        'checkpoint_root': output_path,
    }
    adapt_cfg_path = os.path.join(output_path, 'adapt_config.json')
    with open(adapt_cfg_path, 'w', encoding='utf-8') as f:
        json.dump(adapt_config, f, indent=2)
    logger.info(f"Wrote adaptation config to {adapt_cfg_path}. You can pass this to adapt.py with --config.")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--source_dates', nargs='+', default=['20250707_20250708_20250709'])
    parser.add_argument('--source_dates', nargs='+', default=['20250325_20250326_20250327'])
    # parser.add_argument('--source_dates', nargs='+', default=['T'])
    parser.add_argument('--data_path', default='/media/ubuntu/Storage1/ecog_data/daily_bdy_20250908',
    # parser.add_argument('--data_path', default='bcic_1',
    # parser.add_argument('--data_path', default='/media/ubuntu/Storage1/ecog_data/new_day_data/C07_BDY-S01_CF-8Fr/daily/beida_s01_angle_filtered.h5',
                        type=str, help='Path to the dataset.' \
                        'For BCI competition data, use "bcic_1", where 1 is the subject number.')
    parser.add_argument('--output_path', default='/media/ubuntu/Storage1/ecog_data/OTTA_results',
                        type=str, help='Path to save outputs and models')
    # parser.add_argument('--indices_path', type=str, default='/media/ubuntu/Storage1/ecog_data/OTTA_results/20251231_142652/training_history_T.json',
    parser.add_argument('--indices_path', type=str, default='',
                        help='Path to the npz file containing selected indices for training/validation split')
    parser.add_argument('--ecog_step', type=int, default=256, help='Step size for windowing ECoG data')
    parser.add_argument('--new_data_n_classes', type=int, default=4, help='Number of classes in the dataset')

    parser.add_argument('--alignment', type=str, default='Riemannian', choices=['Euclidean', 'Riemannian'], 
                        help='Similarity metric for hypergraph construction')
    
    parser.add_argument('--n_temporal_filters', type=int, default=320)
    parser.add_argument('--temp_filter_length_inp', type=int, default=8)
    parser.add_argument('--spatial_expansion', type=int, default=1)
    parser.add_argument('--pool_length_inp', type=int, default=16)
    parser.add_argument('--pool_stride_inp', type=int, default=4)
    parser.add_argument('--dropout_inp', type=float, default=0.25)
    parser.add_argument('--ch_dim', type=int, default=64)
    parser.add_argument('--temp_filter_length', type=int, default=4)
    parser.add_argument('--pool_length', type=int, default=2)
    parser.add_argument('--pool_stride', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--use_feedforward', type=bool, default=False,
                        help='Whether to use an additional feedforward layer before classifier')

    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs in source domain')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for source domain training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for source domain training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for source domain training')
    parser.add_argument('--criterion', type=str, default='CrossEntropy', choices=['CrossEntropy', 'MSE'],
                            help='Loss function for source domain training')    

    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for computation')
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file with argument defaults')
    parser.add_argument('--model_name', type=str, default='MultiScale1DCNN_v2', choices=['BaseNet', 'MultiScale1DCNN_v2'], help='Model to use')

    # If a config file is provided, load it and set parser defaults before final parsing.
    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        try:
            with open(pre_args.config, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            # set JSON values as defaults; CLI will still override them if provided
            parser.set_defaults(**cfg)
            print(f"Loaded config defaults from {pre_args.config}")
        except Exception as e:
            print(f"Failed to load config '{pre_args.config}': {e}")

    args = parser.parse_args()

    main(args)