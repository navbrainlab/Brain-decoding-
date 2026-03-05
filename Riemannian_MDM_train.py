import os
import numpy as np
from dataset import *
import argparse
import time
import random
import logging
import json
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
import torch


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
            # Subject id is the suffix in data_path, e.g., 'bcic_1' -> subject_id=1
            subject_id = int(args.data_path.split('_')[-1])
            # Try loading indices from JSON (training_history) if provided
            try:
                with open(args.indices_path, 'r', encoding='utf-8') as f:
                    _data = json.load(f)
                indices = _data.get('indices', None)
            except Exception as e:
                logger.warning(f"Failed to load indices from {args.indices_path}: {e}. Setting indices=None")
                indices = None
            # Load BCIC with source session and return dates for alignment
            train_data, train_labels, valid_data, valid_labels, train_date, valid_date, indices = load_centered_bcic(
                subject_id,
                session=source_date,
                return_date=True,
                indices=indices
            )
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
            try:
                with open(args.indices_path, 'r', encoding='utf-8') as f:
                    _data = json.load(f)
                indices = np.array(_data['indices']) if _data.get('indices', None) is not None else None
            except Exception as e:
                logger.warning(f"Failed to load indices from {args.indices_path}: {e}. Setting indices=None")
                indices = None
            train_data, train_labels, valid_data, valid_labels, train_date, valid_date, indices = load_centered_new_data_for_otta(
                args.data_path, source_dates, train_ratio=0.8, return_date=True, indices=indices, class_num=args.new_data_n_classes)
            # train_data: [n_samples, n_channels, n_timepoints], train_labels: [n_samples, n_classes], train_date: [n_samples]                   
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
    

    # Compute Riemannian mean of different classes in the source domain
    n_classes = train_labels.shape[1]
    class_means = []
    cov_matrices_aligned = np.array([np.cov(sample) for sample in train_data])  # shape: [n_samples, n_channels, n_channels]
    for c in range(n_classes):
        class_covs = cov_matrices_aligned[train_labels[:, c] == 1]
        class_mean = mean_riemann(class_covs)
        class_means.append(class_mean)
    class_means = np.array(class_means)  # shape: [n_classes, n_channels, n_channels]
    np.savez(os.path.join(output_path, f'class_means_source_{args.source_dates[0]}.npz'), class_means=class_means)
    logger.info(f"Saved class means of source domain to {os.path.join(output_path, f'class_means_source_{args.source_dates[0]}.npz')}")

    # Assign each validation sample to the class according to minimum Riemannian distance to class means
    valid_preds = []
    for sample in valid_data:
        sample_cov = np.cov(sample)
        dists_to_means = [distance_riemann(sample_cov, class_mean) for class_mean in class_means]
        pred_class = np.argmin(dists_to_means)
        valid_preds.append(pred_class)
    valid_preds = np.array(valid_preds)
    valid_true = np.argmax(valid_labels, axis=1)
    accuracy = np.mean(valid_preds == valid_true)
    logger.info(f"Validation accuracy on source domain using Riemannian minimum distance classifier: {accuracy:.4f}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--source_dates', nargs='+', default=['T'])
    # parser.add_argument('--source_dates', nargs='+', default=['20250325_20250326_20250327'])
    parser.add_argument('--source_dates', nargs='+', default=['20250707_20250708_20250709'])
    parser.add_argument('--data_path', default='/media/ubuntu/Storage1/ecog_data/new_day_data/C07_BDY-S01_CF-8Fr/daily/beida_s01_angle_filtered.h5',
    # parser.add_argument('--data_path', default='bcic_1',
    # parser.add_argument('--data_path', default='/media/ubuntu/Storage1/ecog_data/daily_bdy_20250908',
                        type=str, help='Path to the dataset')
    parser.add_argument('--output_path', default='/media/ubuntu/Storage1/ecog_data/MDM_results',
                        type=str, help='Path to save outputs and models')
    parser.add_argument('--indices_path', type=str, default='/media/ubuntu/Storage1/ecog_data/OTTA_results/20260203_145056/training_history_20250707_20250708_20250709.json',
    # parser.add_argument('--indices_path', type=str, default='/media/ubuntu/Storage1/ecog_data/HOTL_results/20251210_135405/source_model_20250325_20250326_20250327.npz',
                        help='Path to the npz file containing selected indices for training/validation split')
    parser.add_argument('--ecog_step', type=int, default=32, help='Step size for windowing ECoG data')
    parser.add_argument('--new_data_n_classes', type=int, default=4, help='Number of classes in the new data')

    parser.add_argument('--alignment', type=str, default='Riemannian', choices=['Euclidean', 'Riemannian'], 
                        help='Similarity metric for hypergraph construction')
    
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file with argument defaults')

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