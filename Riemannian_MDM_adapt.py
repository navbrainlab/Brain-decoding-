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
from pyriemann.utils.distance import distance_riemann
import torch
from tqdm import tqdm



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
    logger = setup_logging(args.class_means_root, experiment_timestamp, args)
    # Load source domain arguments
    source_args_path = os.path.join(args.class_means_root, 'args.json')
    with open(source_args_path, 'r', encoding='utf-8') as f:
        source_args = json.load(f)

    for target_date in args.target_dates:

        # Load target domain data (daily_bdy or bcic)
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
                step=args.ecog_step,
                train_ratio=1.0,
                return_date=False,
                indices=indices
            )
        elif 'bcic' in args.data_path:
            # Subject id is the suffix in data_path, e.g., 'bcic_1' -> subject_id=1
            subject_id = int(args.data_path.split('_')[-1])
            try:
                indices = np.load(os.path.join(args.indices_root, f'adaptation_results_{target_date}.npz'), allow_pickle=True)['indices']
            except Exception as e:
                logger.warning(f"Failed to load indices from {args.indices_root}: {e}. Setting indices=None")
                indices = None
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
                args.data_path, [target_date], train_ratio=1.0, return_date=False, indices=indices, class_num=source_args['new_data_n_classes'])             
        else:
            raise ValueError("Unrecognized dataset in data_path; expected a path containing 'daily_bdy' or 'bcic'")
        logger.info(f"Loaded {target_date} data with shape {target_data.shape}")

        # Load source domain model
        source_model_path = os.path.join(args.class_means_root, f'class_means_source_{source_args["source_dates"][0]}.npz')
        class_means = np.load(source_model_path)['class_means']
        logger.info(f"Loaded source model from {source_model_path}")

        # Add data to buffer one by one. Align the data in the buffer when every data is added, and update the model with entropy minimization only when all data in the buffer are updated.
        buffer = deque(maxlen=args.buffer_size)
        new_item_num = 0
        num_correct = 0
        total_samples = 0
        accuracy = []
        prediction_list = []
        label_list = []
        for data, label in tqdm(zip(target_data, target_labels), desc=f"Adapting on target domain {target_date}"):
            buffer.append(data)
            buffer_array = np.array(buffer)
            data = align(buffer_array, data, alignment=args.alignment, buffer_weighting=args.buffer_weighting, buffer_decay=args.buffer_decay)
            new_item_num += 1
            new_item_num %= args.buffer_size            
            # Update the model only when all data in the buffer are updated
            if new_item_num == 0 and args.update:
                raise NotImplementedError("Entropy minimization needs to be double checked.")
            # Predict
            sample_cov = np.cov(data)
            dists_to_means = [distance_riemann(sample_cov, class_mean) for class_mean in class_means]
            pred_class = np.argmin(dists_to_means)
            pred = np.zeros_like(label)
            pred[pred_class] = 1.0

            # Record prediction
            prediction_list.append(pred)
            label_list.append(label)
            if np.argmax(pred) == np.argmax(label):
                num_correct += 1
            total_samples += 1
            accuracy.append(num_correct / total_samples)
            if total_samples % 100 == 0:
                logger.info(f"Adaptation on {target_date}: Processed {total_samples} samples, Current Accuracy: {accuracy[-1]:.4f}")
        logger.info(f"Finished adaptation on {target_date}: Total samples: {total_samples}, Final Accuracy: {accuracy[-1]:.4f}")

        # Save the results on the target domain
        results_path = os.path.join(args.class_means_root, f'adaptation_results_{target_date}.npz')
        record_dict = {
            'accuracy': np.array(accuracy),
            'predictions': np.array(prediction_list),
            'labels': np.array(label_list),
            'indices': indices,
        }
        np.savez(results_path, **record_dict)

        # Plot the accuracy curve
        plt.figure()
        plt.plot(range(1, len(accuracy)+1), accuracy, label='Riemannian MDM Adaptation')
        plt.xlabel('Number of Samples')
        plt.ylabel('Accuracy')
        plt.title(f'Online Adaptation Accuracy on Target Domain {target_date}')
        plt.legend()
        plt_path = os.path.join(args.class_means_root, f'adaptation_accuracy_{target_date}.png')
        plt.savefig(plt_path)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--class_means_root', type=str, default='/media/ubuntu/Storage1/ecog_data/MDM_results/20251226_195147')
    parser.add_argument('--class_means_root', type=str, default='/media/ubuntu/Storage1/ecog_data/MDM_results/20260203_222231')
    # parser.add_argument('--source_dates', nargs='+', default=['20250325_20250326_20250327'])
    parser.add_argument('--source_dates', nargs='+', default=['20250707_20250708_20250709'])
    # parser.add_argument('--source_dates', nargs='+', default=['T'])
    # parser.add_argument('--target_dates', nargs='+', default=['20250329', '20250331', '20250401'])
    parser.add_argument('--target_dates', nargs='+', default=['20250710', '20250711', '20250804'])
    # parser.add_argument('--target_dates', nargs='+', default=['E'])
    # parser.add_argument('--data_path', default='/media/ubuntu/Storage1/ecog_data/daily_bdy_20250908',
    parser.add_argument('--data_path', default='/media/ubuntu/Storage1/ecog_data/new_day_data/C07_BDY-S01_CF-8Fr/daily/beida_s01_angle_filtered.h5',
    # parser.add_argument('--data_path', default='bcic_1',
                        type=str, help='Path to the dataset')
    parser.add_argument('--indices_root', type=str, default='/media/ubuntu/Storage1/ecog_data/OTTA_results/20260203_145056',
    # parser.add_argument('--indices_root', type=str, default='/media/ubuntu/Storage1/ecog_data/OTTA_results/20251231_142652/adaptation_results_E.npz',
    # parser.add_argument('--indices_root', type=str, default='/media/ubuntu/Storage1/ecog_data/HOTL_results/20251210_135405',
                        help='Path to the folder containing npz files with selected indices for each target date')
    parser.add_argument('--ecog_step', type=int, default=32, help='Step size for windowing ECoG data')

    parser.add_argument('--alignment', type=str, default='Riemannian', choices=['Euclidean', 'Riemannian'], 
                        help='Similarity metric for hypergraph construction')
    parser.add_argument('--update', type=bool, default=False, help='Whether to perform model weight online update with entropy minimization')
    
    parser.add_argument('--buffer_size', type=int, default=32, help='Buffer size for online adaptation')
    parser.add_argument('--buffer_weighting', type=str, default='uniform', choices=['uniform','linear','exponential'],
                        help='Weighting scheme for FIFO buffer elements (arrival order).')
    parser.add_argument('--buffer_decay', type=float, default=2.0,
                        help='Decay parameter for exponential weighting (larger -> stronger focus on newest).')    

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