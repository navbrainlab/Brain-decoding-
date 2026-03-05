import os
import numpy as np
from dataset import *
import argparse
import time
import matplotlib.pyplot as plt
import random
import logging
import json
from collections import deque
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
import torch
from tqdm import tqdm
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
    file_handler = logging.FileHandler(os.path.join(exp_path, f'adapt_{exp_id}.log'))
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
    
def align(buffer_array, data, alignment, buffer_weighting='uniform', buffer_decay=1.0):
    # Calculate the covariance matrix of each data in the buffer
    buffer_array = np.array([np.cov(item) for item in buffer_array])
    if alignment == 'Riemannian':
        if buffer_weighting == 'uniform':
            weights = np.ones(len(buffer_array))
        elif buffer_weighting == 'linear':
            # Oldest gets smallest weight
            weights = np.linspace(0.1, 1.0, len(buffer_array))
        elif buffer_weighting == 'exponential':
            weights = np.exp(np.linspace(buffer_decay, 0, len(buffer_array)))
        else:
            raise ValueError(f"Unknown buffer_weighting: {buffer_weighting}")
        weights = weights / (weights.sum() + 1e-12)
        riemann_mean = mean_riemann(buffer_array, sample_weight=weights)

        # Align current covariance with weighted mean
        eigvals_dm, eigvecs_dm = np.linalg.eigh(riemann_mean)
        eigvals_dm = np.clip(eigvals_dm, a_min=1e-8, a_max=None)
        M_inv_sqrt = eigvecs_dm @ np.diag(1.0 / np.sqrt(eigvals_dm)) @ eigvecs_dm.T
        aligned_data = M_inv_sqrt @ data   

    elif alignment == 'Euclidean':
        if buffer_weighting == 'uniform':
            weights = np.ones(len(buffer_array))
        elif buffer_weighting == 'linear':
            # Oldest gets smallest weight
            weights = np.linspace(0.1, 1.0, len(buffer_array))
        elif buffer_weighting == 'exponential':
            weights = np.exp(np.linspace(buffer_decay, 0, len(buffer_array)))
        else:
            raise ValueError(f"Unknown buffer_weighting: {buffer_weighting}")
        weights = weights / (weights.sum() + 1e-12)
        euclidean_mean = np.average(buffer_array, axis=0, weights=weights)
        eigvals_dm, eigvecs_dm = np.linalg.eigh(euclidean_mean)
        eigvals_dm = np.clip(eigvals_dm, a_min=1e-8, a_max=None)
        M_inv_sqrt = eigvecs_dm @ np.diag(1.0 / np.sqrt(eigvals_dm)) @ eigvecs_dm.T
        aligned_data = M_inv_sqrt @ data   

    else:
        raise ValueError(f"Unknown alignment method: {alignment}")
    return aligned_data

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def main(args):
    # Set random seed for reproducibility
    seed_everything(args.seed)

    # Set up logging
    experiment_timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(args.source_hypergraph_root, experiment_timestamp, args)
    # Load checkpoint arguments
    checkpoint_args_path = os.path.join(args.checkpoint_root, 'args.json')
    with open(checkpoint_args_path, 'r', encoding='utf-8') as f:
        checkpoint_args = json.load(f)

    for target_date in args.target_dates:
        
        # Load source domain hypergraph
        source_hypergraph_path = os.path.join(args.source_hypergraph_root, f'source_model_{args.source_dates[0]}.npz')
        source_hypergraph = np.load(source_hypergraph_path, allow_pickle=True)
        logger.info(f"Loaded source hypergraph from {source_hypergraph_path}")
        M_spd = torch.from_numpy(source_hypergraph['M_spd']).float().to(args.device)
        M_deep = torch.from_numpy(source_hypergraph['M_deep']).float().to(args.device)
        ecog_step = int(source_hypergraph['ecog_step'])
        similarity = str(source_hypergraph['similarity'])
        alignment = str(source_hypergraph['alignment'])
        riemann_mean_source = source_hypergraph['riemann_mean_spd']
        spd_weight = torch.from_numpy(source_hypergraph['spd_weight']).float().to(args.device)
        deep_weight = torch.from_numpy(source_hypergraph['deep_weight']).float().to(args.device)

        # Load target domain data
        # Load target domain data for different datasets
        if 'daily_bdy' in args.data_path:
            try:
                indices = np.load(os.path.join(args.indices_root, f'adaptation_results_{target_date}.npz'), allow_pickle=True)['indices']
            except Exception as e:
                logger.warning(f"Failed to load indices from {args.indices_root}: {e}. Setting indices=None")
                indices = None
            target_data_path = [os.path.join(args.data_path, target_date)]
            target_data, target_labels, _, _, indices = load_centered_data_for_otta(
                target_data_path,
                windows_size=256,
                step=ecog_step,
                train_ratio=1.0,
                return_date=False,
                indices=indices
            )
        elif 'bcic' in args.data_path:
            # Subject id is the suffix in data_path, e.g., 'bcic_1' -> subject_id=1
            subject_id = int(args.data_path.split('_')[-1])
            # Indices for train/valid split may be stored in a JSON training_history file
            try:
                indices = np.load(os.path.join(args.indices_root, f'adaptation_results_{target_date}.npz'), allow_pickle=True)['indices']
            except Exception as e:
                logger.warning(f"Failed to load indices from {args.indices_root}: {e}. Setting indices=None")
                indices = None
            # Load BCIC target session with full ratio as target-only (no train split)
            target_data, target_labels, _, _, indices = load_centered_bcic(
                subject_id,
                session=target_date,
                return_date=False,
                indices=indices,
                train_ratio=1.0
            )
        elif '.h5' in args.data_path:
            try:
                indices = np.load(os.path.join(args.indices_root, f'adaptation_results_{target_date}.npz'), allow_pickle=True)['indices']
            except Exception as e:
                logger.warning(f"Failed to load indices from {args.indices_root}: {e}. Setting indices=None")
                indices = None
            target_data, target_labels, _, _, indices = load_centered_new_data_for_otta(
                args.data_path, [target_date], train_ratio=1.0, return_date=False, indices=indices, class_num=checkpoint_args['new_data_n_classes'])
            # train_data: [n_samples, n_channels, n_timepoints], train_labels: [n_samples, n_classes], train_date: [n_samples]              
        else:
            raise ValueError("Unrecognized dataset in data_path; expected a path containing 'daily_bdy' or 'bcic'")
        logger.info(f"Loaded {target_date} data with shape {target_data.shape}")
        n_channels = target_data.shape[1]
        n_timepoints = target_data.shape[2]

        # Deep model and optimizer initialization
        # Deep model selection: CLI overrides checkpoint args.json; default remains BaseNet.
        model_name = args.model_name if args.model_name is not None else checkpoint_args.get('model_name', 'BaseNet')
        if model_name == 'BaseNet':
            model = BaseNet(
                input_window_samples=n_timepoints,
                n_channels=n_channels,
                n_temporal_filters=checkpoint_args['n_temporal_filters'],
                temp_filter_length_inp=checkpoint_args['temp_filter_length_inp'],
                spatial_expansion=checkpoint_args['spatial_expansion'],
                pool_length_inp=checkpoint_args['pool_length_inp'],
                pool_stride_inp=checkpoint_args['pool_stride_inp'],
                dropout_inp=checkpoint_args['dropout_inp'],
                ch_dim=checkpoint_args['ch_dim'],
                temp_filter_length=checkpoint_args['temp_filter_length'],
                pool_length=checkpoint_args['pool_length'],
                pool_stride=checkpoint_args['pool_stride'],
                dropout=checkpoint_args['dropout'],
                n_classes=target_labels.shape[1],
                use_feedforward=checkpoint_args['use_feedforward'] if 'use_feedforward' in checkpoint_args else False,
                return_features=True,
            )
        elif model_name == 'MultiScale1DCNN_v2':
            model = MultiScale1DCNN_v2(
                in_channels=n_channels,
                num_class=target_labels.shape[1],
            )
        else:
            raise ValueError(f"Unrecognized model_name: {model_name}")

        model = model.to(args.device)
        # Load source domain model
        checkpoint_path = os.path.join(args.checkpoint_root, f'best_model_{args.source_dates[0]}.pt')
        model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.adapt_lr, weight_decay=args.adapt_weight_decay)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Using deep model: {model_name}")

        # Add data to buffer one by one. Align the data in the buffer when every data is added, and update the model with entropy minimization only when all data in the buffer are updated.
        buffer = deque(maxlen=args.buffer_size)
        new_item_num = 0
        total_samples = 0
        deep_num_correct = 0
        spd_num_correct = 0
        total_num_correct = 0
        accuracy = []
        deep_accuracy = []
        spd_accuracy = []
        preds_list = []
        deep_preds_list = []
        spd_preds_list = []
        label_list = []
        for data, label in tqdm(zip(target_data, target_labels), desc=f"Adapting on target domain {target_date}"):
            buffer.append(data)
            buffer_array = np.array(buffer)
            data = align(buffer_array, data, alignment=alignment, buffer_weighting=args.buffer_weighting, buffer_decay=args.buffer_decay) # shape: [n_channels, n_timepoints]
            new_item_num += 1
            new_item_num %= args.buffer_size            
            # Update the model only when all data in the buffer are updated
            if new_item_num == 0 and args.update:
                raise NotImplementedError("Entropy minimization needs to be double checked.")
                x = torch.from_numpy(buffer_array).float().to(args.device)
                model.train()
                outputs = model(x)
                loss = softmax_entropy(outputs).mean(0)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Deep feature extraction and prediction
            input_tensor = torch.tensor(data[np.newaxis, :, :], dtype=torch.float32, device=args.device)
            model.eval()
            with torch.no_grad():
                _, deep_features = model(input_tensor)
            deep_features = deep_features.reshape(deep_features.shape[0], -1).T  # [n_features, 1]
            deep_preds = deep_features.T @ M_deep

            # SPD feature extraction and prediction
            sample_cov = np.cov(data)
            if 'tangent' in similarity:
                spd_features = tangent_space(sample_cov[None, :, :], riemann_mean_source).T  # [n_tangent_features, 1]
            else:
                spd_features = sample_cov.reshape(-1,1)  # [n_features, 1]
            spd_features = torch.from_numpy(spd_features).float().to(args.device)
            spd_preds = spd_features.T @ M_spd

            # Total prediction
            preds = spd_weight * spd_features.T @ M_spd + deep_weight * deep_features.T @ M_deep  # [1, n_classes]
            
            # Record prediction
            deep_preds_list.append(deep_preds.flatten().cpu().numpy())
            spd_preds_list.append(spd_preds.flatten().cpu().numpy())
            preds_list.append(preds.flatten().cpu().numpy())
            label_list.append(label)
            if torch.argmax(deep_preds, dim=1).cpu().numpy() == np.argmax(label):
                deep_num_correct += 1
            if torch.argmax(spd_preds, dim=1).cpu().numpy() == np.argmax(label):
                spd_num_correct += 1
            if torch.argmax(preds, dim=1).cpu().numpy() == np.argmax(label):
                total_num_correct += 1
            total_samples += 1
            deep_accuracy.append(deep_num_correct / total_samples)
            spd_accuracy.append(spd_num_correct / total_samples)
            accuracy.append(total_num_correct / total_samples)
            if total_samples % 100 == 0:
                logger.info(f"Adaptation on {target_date}: Processed {total_samples} samples, Deep Accuracy: {deep_accuracy[-1]:.4f}, SPD Accuracy: {spd_accuracy[-1]:.4f}, Total Accuracy: {accuracy[-1]:.4f}")
        logger.info(f"Finished adaptation on {target_date}: Total samples: {total_samples}, Deep Accuracy: {deep_accuracy[-1]:.4f}, SPD Accuracy: {spd_accuracy[-1]:.4f}, Total Accuracy: {accuracy[-1]:.4f}")

        # Save the results on the target domain
        results_path = os.path.join(args.source_hypergraph_root, f'adaptation_results_{target_date}.npz')
        record_dict = {
            'accuracy': np.array(accuracy),
            'deep_accuracy': np.array(deep_accuracy),
            'spd_accuracy': np.array(spd_accuracy),
            'predictions': np.array(preds_list),
            'deep_predictions': np.array(deep_preds_list),
            'spd_predictions': np.array(spd_preds_list),
            'labels': np.array(label_list),
            'indices': indices
        }
        np.savez(results_path, **record_dict)
        # Save the model after adaptation
        if args.update:
            adapted_model_path = os.path.join(args.source_hypergraph_root, f'adapted_model_{target_date}.npz')
            torch.save(model.state_dict(), adapted_model_path)

        # Plot the accuracy curve
        plt.figure()
        plt.plot(range(1, len(accuracy)+1), accuracy, label='Total Accuracy')
        plt.plot(range(1, len(deep_accuracy)+1), deep_accuracy, label='Deep Features Accuracy')
        plt.plot(range(1, len(spd_accuracy)+1), spd_accuracy, label='SPD Features Accuracy')
        plt.xlabel('Number of Samples')
        plt.ylabel('Accuracy')
        plt.title(f'Online Test Time Adaptation Accuracy on Target Domain {target_date}')
        plt.legend()
        plt_path = os.path.join(args.source_hypergraph_root, f'adaptation_accuracy_{target_date}.png')
        plt.savefig(plt_path)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_root', type=str, default='/media/ubuntu/Storage1/ecog_data/OTTA_results/20260303_171635')
    # parser.add_argument('--source_dates', nargs='+', default=['T'])
    # parser.add_argument('--source_dates', nargs='+', default=['20250707_20250708_20250709'])
    parser.add_argument('--source_dates', nargs='+', default=['20250325_20250326_20250327'])
    # parser.add_argument('--source_dates', nargs='+', default=['20250325'])
    # parser.add_argument('--target_dates', nargs='+', default=['E'])
    # parser.add_argument('--target_dates', nargs='+', default=['20250710', '20250711','20250804', '20250814', '20250825'])   
    parser.add_argument('--target_dates', nargs='+', default=['20250329', '20250331', '20250401'])
    # parser.add_argument('--target_dates', nargs='+', default=['20250326', '20250327','20250329', '20250331', '20250401'])
    # parser.add_argument('--data_path', default='/media/ubuntu/Storage1/ecog_data/new_day_data/C07_BDY-S01_CF-8Fr/daily/beida_s01_angle_filtered.h5',
    # parser.add_argument('--data_path', default='bcic_1',
    parser.add_argument('--data_path', default='/media/ubuntu/Storage1/ecog_data/daily_bdy_20250908',
                        type=str, help='Path to the dataset')
    parser.add_argument('--indices_root', type=str, default='/media/ubuntu/Storage1/ecog_data/OTTA_results/20260303_171635',
    # parser.add_argument('--indices_root', type=str, default='/media/ubuntu/Storage1/ecog_data/HOTL_results/20251210_135405',
    # parser.add_argument('--indices_root', type=str, default='/media/ubuntu/Storage1/ecog_data/HOTL_results/20251211_094853',
                        help='Path to the folder containing npz files with selected indices for each target date')
    parser.add_argument('--source_hypergraph_root', type=str, default='/media/ubuntu/Storage1/ecog_data/Multi-feature_results/20260304_105348',
    # parser.add_argument('--source_hypergraph_root', type=str, default='/media/ubuntu/Storage1/ecog_data/Multi-feature_results/20251231_152954',
    # parser.add_argument('--source_hypergraph_root', type=str, default='/media/ubuntu/Storage1/ecog_data/Multi-feature_results/20251226_171808',
                        help='Path to the folder containing pre-trained source domain models')

    parser.add_argument('--update', type=bool, default=False, help='Whether to perform model weight online update with entropy minimization')
    
    parser.add_argument('--buffer_size', type=int, default=32, help='Buffer size for online adaptation')
    parser.add_argument('--buffer_weighting', type=str, default='uniform', choices=['uniform','linear','exponential'],
                        help='Weighting scheme for FIFO buffer elements (arrival order).')
    parser.add_argument('--buffer_decay', type=float, default=2.0,
                        help='Decay parameter for exponential weighting (larger -> stronger focus on newest).')    

    parser.add_argument('--adapt_lr', type=float, default=1e-3, help='Learning rate for source domain training')
    parser.add_argument('--adapt_weight_decay', type=float, default=1e-4, help='Weight decay for source domain training')

    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computation')
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file with argument defaults')
    parser.add_argument('--model_name', type=str, default=None, choices=['BaseNet', 'MultiScale1DCNN_v2'],
                        help='Deep model architecture for feature extraction. If omitted, uses checkpoint args.json (model_name) or defaults to BaseNet.')

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