import os
import numpy as np
from dataset import *
import argparse
import time
import random
import logging
import json
from sklearn.metrics.pairwise import cosine_similarity
from pyriemann.utils.distance import pairwise_distance
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
import torch
from basenet import BaseNet
from MultiScale import MultiScale1DCNN_v2
from torch.utils.data import DataLoader


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
    file_handler = logging.FileHandler(os.path.join(exp_path, f'{exp_id}.log'))
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

def similarity_computation_and_data_transform(train_data, valid_data, train_labels, similarity, alignment):

    mean = None

    if similarity == 'tangent_cosine':
        cov_matrices = np.array([np.cov(sample) for sample in train_data])  # shape: [n_samples, n_channels, n_channels]
        mean = mean_riemann(cov_matrices)
        # Map all covariance matrices to the tangent space at the mean.
        # Resulting tangent vectors have dimensionality n_channels*(n_channels+1)/2.
        tangent_vectors = tangent_space(cov_matrices, mean)  # shape: [n_samples, n_tangent_features]
        train_data_transformed = tangent_vectors.T  # [n_tangent_features, n_samples]

        similarity_matrix = cosine_similarity(train_data_transformed.T) # [n_samples, n_samples]

        # Align validation data and map it to tangent space
        valid_cov_matrices = np.array([np.cov(sample) for sample in valid_data])  # shape: [n_samples, n_channels, n_channels]
        valid_data_transformed = tangent_space(valid_cov_matrices, mean).T # [n_tangent_features, n_samples]

    elif similarity == 'SPD':
        cov_matrices = np.array([np.cov(sample) for sample in train_data])  # shape: [n_samples, n_channels, n_channels]
        # Compute pairwise Riemannian distances
        dists = pairwise_distance(cov_matrices, metric='riemann') # [n_samples, n_samples]
        # convert distances to similarity
        sigma = np.mean(dists)
        similarity_matrix = np.exp(- (dists ** 2) / (2 * sigma ** 2))
        train_data_transformed = cov_matrices.reshape(cov_matrices.shape[0], -1).T  # [n_channels*n_channels, n_samples]
        valid_cov_matrices = np.array([np.cov(sample) for sample in valid_data])  # shape: [n_samples, n_channels, n_channels]
        valid_data_transformed = valid_cov_matrices.reshape(valid_cov_matrices.shape[0], -1).T # [n_channels*n_channels, n_samples]

    elif similarity == 'tangent_DM':

        cov_matrices = np.array([np.cov(sample) for sample in train_data])  # shape: [n_samples, n_channels, n_channels]
        mean = mean_riemann(cov_matrices)

        # Compute the Riemannian mean of each class
        class_means = []
        n_classes = train_labels.shape[1]
        for cls in range(n_classes):
            cls_mask = (np.argmax(train_labels, axis=1) == cls)
            cls_covs = cov_matrices[cls_mask]
            class_mean = mean_riemann(cls_covs)
            class_means.append(class_mean)
        class_means = np.array(class_means)  # shape: [n_classes, n_channels, n_channels]

        # Map all covariance matrices to the tangent space at the mean.
        # Resulting tangent vectors have dimensionality n_channels*(n_channels+1)/2.
        tangent_vectors = tangent_space(cov_matrices, mean)  # shape: [n_samples, n_tangent_features]
        tangent_class_means = tangent_space(class_means, mean)  # shape: [n_classes, n_tangent_features]
        train_data_transformed = tangent_vectors.T  # [n_tangent_features, n_samples]                  

        # Compute pairwise Euclidean distances to each class mean in tangent space with matrix operations instead of loops
        diff = tangent_vectors[:, None, :] - tangent_class_means[None, :, :]  # shape: [n_samples, n_classes, n_tangent_features]
        dists = np.linalg.norm(diff, axis=2)  # shape: [n_samples, n_classes]
        # Compute cosine similarity from distances
        similarity_matrix = cosine_similarity(dists)  # [n_samples, n_samples]

        valid_cov_matrices = np.array([np.cov(sample) for sample in valid_data])  # shape: [n_samples, n_channels, n_channels]
        valid_data_transformed = tangent_space(valid_cov_matrices, mean).T # [n_tangent_features, n_samples]

    elif similarity == 'SPD_DM':

        cov_matrices = np.array([np.cov(sample) for sample in train_data])  # shape: [n_samples, n_channels, n_channels]
        # Compute the Riemannian mean of each class
        class_means = []
        n_classes = train_labels.shape[1]
        for cls in range(n_classes):
            cls_mask = (np.argmax(train_labels, axis=1) == cls)
            cls_covs = cov_matrices[cls_mask]
            class_mean = mean_riemann(cls_covs)
            class_means.append(class_mean)
        class_means = np.array(class_means)  # shape: [n_classes, n_channels, n_channels]                 

        # Compute pairwise Riemannian distances to each class mean
        dists = np.zeros((cov_matrices.shape[0], n_classes))
        for cls in range(n_classes):
            class_mean = class_means[cls]
            dists[:, cls] = pairwise_distance(cov_matrices, class_mean[None, :, :], metric='riemann').flatten()
        # Compute cosine similarity from distances
        similarity_matrix = cosine_similarity(dists)  # [n_samples, n_samples]

        train_data_transformed = cov_matrices.reshape(cov_matrices.shape[0], -1).T  # [n_channels*n_channels, n_samples]
        valid_cov_matrices = np.array([np.cov(sample) for sample in valid_data])  # shape: [n_samples, n_channels, n_channels]
        valid_data_transformed = valid_cov_matrices.reshape(valid_cov_matrices.shape[0], -1).T # [n_channels*n_channels, n_samples]

    elif similarity == 'cosine': # The Euclidean version of tangent_cosine
        cov_matrices = np.array([np.cov(sample) for sample in train_data])  # shape: [n_samples, n_channels, n_channels]
        mean = np.mean(cov_matrices, axis=0)
        # Map all covariance matrices to the Euclidean tangent space at the mean.
        tangent_vectors = cov_matrices - mean  # shape: [n_samples, n_channels, n_channels]
        tangent_vectors = tangent_vectors.reshape(tangent_vectors.shape[0], -1)  # shape: [n_samples, n_tangent_features]
        train_data_transformed = tangent_vectors.T  # [n_tangent_features, n_samples]

        similarity_matrix = cosine_similarity(train_data_transformed.T) # [n_samples, n_samples]

        # Align validation data and map it to tangent space
        valid_cov_matrices = np.array([np.cov(sample) for sample in valid_data])  # shape: [n_samples, n_channels, n_channels]
        valid_tangent_vectors = valid_cov_matrices - mean  # shape: [n_samples, n_channels, n_channels]
        valid_tangent_vectors = valid_tangent_vectors.reshape(valid_tangent_vectors.shape[0], -1)  # shape: [n_samples, n_tangent_features]
        valid_data_transformed = valid_tangent_vectors.T # [n_tangent_features, n_samples]
    
    elif similarity == 'DM': # The Euclidean version of tangent_DM
        cov_matrices = np.array([np.cov(sample) for sample in train_data])  # shape: [n_samples, n_channels, n_channels]
        # Compute the Euclidean mean of each class
        class_means = []
        n_classes = train_labels.shape[1]
        for cls in range(n_classes):
            cls_mask = (np.argmax(train_labels, axis=1) == cls)
            cls_covs = cov_matrices[cls_mask]
            class_mean = np.mean(cls_covs, axis=0)
            class_means.append(class_mean)
        class_means = np.array(class_means)  # shape: [n_classes, n_channels, n_channels]                 
        # Compute pairwise Euclidean distances to each class mean
        dists = np.zeros((cov_matrices.shape[0], n_classes))
        for cls in range(n_classes):
            class_mean = class_means[cls]
            dists[:, cls] = pairwise_distance(cov_matrices, class_mean[None, :, :], metric='euclid').flatten()
        # Compute cosine similarity from distances
        similarity_matrix = cosine_similarity(dists)  # [n_samples, n_samples]

        mean = np.mean(cov_matrices, axis=0)
        tangent_vectors = cov_matrices - mean  # shape: [n_samples, n_channels, n_channels]
        tangent_vectors = tangent_vectors.reshape(tangent_vectors.shape[0], -1)  # shape: [n_samples, n_tangent_features]
        train_data_transformed = tangent_vectors.T  # [n_tangent_features, n_samples]            
        
        valid_cov_matrices = np.array([np.cov(sample) for sample in valid_data])  # shape: [n_samples, n_channels, n_channels]
        valid_tangent_vectors = valid_cov_matrices - mean  # shape: [n_samples, n_channels, n_channels]
        valid_tangent_vectors = valid_tangent_vectors.reshape(valid_tangent_vectors.shape[0], -1)  # shape: [n_samples, n_tangent_features]
        valid_data_transformed = valid_tangent_vectors.T # [n_tangent_features, n_samples]
    
    else:
        raise ValueError("Unrecognized similarity metric")
    
    return similarity_matrix, train_data_transformed, valid_data_transformed, mean


def hypergraph_construction(similarity_matrix, k):
    hyperedges = []
    n_samples = similarity_matrix.shape[0]
    for i in range(n_samples):
        # Get indices of the K nearest neighbors (excluding the vertex itself)
        nearest_neighbors = np.argsort(-similarity_matrix[i])[:k+1]  # Sort in descending order
        nearest_neighbors = nearest_neighbors[nearest_neighbors != i][:k]  # Exclude the vertex itself
        # Create a hyperedge connecting the vertex and its K nearest neighbors
        hyperedges.append([i] + nearest_neighbors.tolist())
    # Ensure hyperedges are unique
    unique_hyperedges = list(set(tuple(sorted(edge)) for edge in hyperedges))
    # incidence matrix H
    n_hyperedges = len(unique_hyperedges)
    H = np.zeros((n_samples, n_hyperedges))
    for j, edge in enumerate(unique_hyperedges):
        for vertex in edge:
            H[vertex, j] = 1
    W = np.eye(n_hyperedges) / n_hyperedges  # Weight matrix for hyperedges, here we set equal weights
    degree_hyperedge = np.sum(H, axis=0)  # Degree of each hyperedge
    degree_vertex = np.sum(H@W, axis=1)  # Degree of each vertex
    Dv = np.diag(degree_vertex)
    De = np.diag(degree_hyperedge)
    Dv_inv_sqrt = np.diag(1.0 / np.sqrt(degree_vertex + 1e-8))
    De_inv = np.diag(1.0 / degree_hyperedge)
    Delta = np.eye(n_samples) - Dv_inv_sqrt @ H @ W @ De_inv @ H.T @ Dv_inv_sqrt
    return H, W, Delta

def hypergraph_learning(train_data, train_labels, Delta, lambda_hg, miu, logger):
    U = np.eye(train_data.shape[0])  # Initialize U as identity matrix
    lambda_hg = args.lambda_hg
    miu = args.miu
    XDXT = train_data @ Delta @ train_data.T
    XXT = train_data @ train_data.T
    XY = train_data @ train_labels
    for iter in range(20):
        M0 = M.copy() if iter > 0 else np.zeros((train_data.shape[0], train_labels.shape[1]))
        U0 = U.copy()
        M = lambda_hg * np.linalg.inv(XDXT + lambda_hg*XXT + miu*U) @ XY
        U = np.diag(0.5 / (np.sum(M**2, axis=1) + 1e-8))
        diff_M = np.linalg.norm(M - M0, ord='fro') / (np.linalg.norm(M0, ord='fro') + 1e-8)
        diff_U = np.linalg.norm(U - U0, ord='fro') / (np.linalg.norm(U0, ord='fro') + 1e-8)
        logger.info(f"Iter {iter+1}: M={M.mean():.3e}, U={U.mean():.3e}, diff_M={diff_M:.4e}, diff_U={diff_U:.4e}")
        if diff_M < 5e-2 and diff_U < 5e-2:
            break    
    return M, U

def main(args):

    # Generate experiment timestamp
    experiment_timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Create output directory if it doesn't exist
    output_path = os.path.join(args.output_path, experiment_timestamp)
    os.makedirs(output_path, exist_ok=True)
    logger = setup_logging(output_path, experiment_timestamp, args)
    logger.info(f"Experiment Timestamp: {experiment_timestamp}")
    seed_everything(args.seed)

    source_args_path = os.path.join(args.checkpoint_root, 'args.json')
    with open(source_args_path, 'r', encoding='utf-8') as f:
        source_args = json.load(f)

    # Off line training in the source domains

    for source_date in args.source_dates:

        # Load data for each source domain
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
        elif 'bcic' in args.data_path:
            subject_id = int(args.data_path.split('_')[-1])
            try:
                with open(args.indices_path, 'r', encoding='utf-8') as f:
                    _data = json.load(f)
                indices = _data['indices']
            except Exception as e:
                logger.warning(f"Failed to load indices from {args.indices_path}: {e}. Setting indices=None")
                indices = None
            train_data, train_labels, valid_data, valid_labels, train_date, valid_date, indices = load_centered_bcic(
                subject_id, session=source_date, windows_size=1000, step=1000, train_ratio=0.8, return_date=True, indices=indices)
            n_channels = train_data.shape[1]
            n_timepoints = train_data.shape[2]
            logger.info(f"Loaded BCIC data with {n_channels} channels and {n_timepoints} timepoints")
        
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
                args.data_path, source_dates, train_ratio=0.8, return_date=True, indices=indices, class_num=source_args['new_data_n_classes'])
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



    similarity_matrix, train_data_transformed, valid_data_transformed, mean_spd = similarity_computation_and_data_transform(train_data, valid_data, train_labels, args.similarity, args.alignment)
    logger.info(f"{args.similarity} similarity matrix stats: min={similarity_matrix.min():.4e}, max={similarity_matrix.max():.4e}, mean={similarity_matrix.mean():.4e}")
    # train_data_transformed: [n_features, n_samples], valid_data_transformed: [n_features, n_samples]

    # Build hypergraph based on similarity matrix using kNN
    H, W, Delta = hypergraph_construction(similarity_matrix, args.k)

    M, U = hypergraph_learning(train_data_transformed, train_labels, Delta, args.lambda_hg, args.miu, logger)

    Omega_spd = np.trace(M.T @ train_data_transformed @ Delta @ train_data_transformed.T @ M)
    logger.info(f"SPD feature Omega: {Omega_spd:.4e}")
    error_spd = np.linalg.norm(train_data_transformed.T @ M - train_labels, ord='fro')**2
    logger.info(f"SPD feature training error: {error_spd:.4e}")
    regularization_spd = np.sum(np.sqrt(np.sum(M**2, axis=1)))
    logger.info(f"SPD feature regularization: {regularization_spd:.4e}")
    gamma_spd = Omega_spd + args.lambda_hg * error_spd + args.miu * regularization_spd
    logger.info(f"gamma (SPD features): {gamma_spd:.4e}")


    # Deep feature

    train_dataset = XZD_Dataset(train_data, train_labels, name='source_train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False) # Already shuffled in data loading
    valid_dataset = XZD_Dataset(valid_data, valid_labels, name='source_valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    n_channels = train_data.shape[1]
    n_timepoints = train_data.shape[2]

    # Deep model selection: CLI overrides checkpoint args.json; default remains BaseNet.
    model_name = args.model_name if args.model_name is not None else source_args.get('model_name', 'BaseNet')
    if model_name == 'BaseNet':
        model = BaseNet(
            input_window_samples=n_timepoints,
            n_channels=n_channels,
            n_temporal_filters=source_args['n_temporal_filters'],
            temp_filter_length_inp=source_args['temp_filter_length_inp'],
            spatial_expansion=source_args['spatial_expansion'],
            pool_length_inp=source_args['pool_length_inp'],
            pool_stride_inp=source_args['pool_stride_inp'],
            dropout_inp=source_args['dropout_inp'],
            ch_dim=source_args['ch_dim'],
            temp_filter_length=source_args['temp_filter_length'],
            pool_length=source_args['pool_length'],
            pool_stride=source_args['pool_stride'],
            dropout=source_args['dropout'],
            n_classes=train_labels.shape[1],
            use_feedforward=source_args['use_feedforward'] if 'use_feedforward' in source_args else False,
            return_features=True,
        )
    elif model_name == 'MultiScale1DCNN_v2':
        model = MultiScale1DCNN_v2(
            in_channels=n_channels,
            num_class=train_labels.shape[1],
        )
    else:
        raise ValueError(f"Unrecognized model_name: {model_name}")

    model = model.to(args.device)
    logger.info(f"Using deep model: {model_name}")

    source_model_path = os.path.join(args.checkpoint_root, f'best_model_{args.source_dates[0]}.pt')
    model.load_state_dict(torch.load(source_model_path, map_location=args.device))    
    logger.info(f"Loaded model from {source_model_path}")

    # Extract deep features
    model.eval()
    for train_data_batch, _ in train_dataloader:
        train_data_batch = train_data_batch.to(args.device)
        with torch.no_grad():
            _, train_features_batch = model(train_data_batch)
        train_features_batch = train_features_batch.cpu().numpy()
        if args.normalize:
            train_features_batch = (train_features_batch - np.mean(train_features_batch, axis=1, keepdims=True)) / (np.std(train_features_batch, axis=1, keepdims=True) + 1e-8)
        if 'train_features' not in locals():
            train_features = train_features_batch
        else:
            train_features = np.concatenate((train_features, train_features_batch), axis=0)
    logger.info(f"Extracted train features with shape {train_features.shape}")

    for valid_data_batch, _ in valid_dataloader:
        valid_data_batch = valid_data_batch.to(args.device)
        with torch.no_grad():
            _, valid_features_batch = model(valid_data_batch)
        valid_features_batch = valid_features_batch.cpu().numpy()
        if args.normalize:
            valid_features_batch = (valid_features_batch - np.mean(valid_features_batch, axis=1, keepdims=True)) / (np.std(valid_features_batch, axis=1, keepdims=True) + 1e-8)
        if 'valid_features' not in locals():
            valid_features = valid_features_batch
        else:
            valid_features = np.concatenate((valid_features, valid_features_batch), axis=0)
    logger.info(f"Extracted valid features with shape {valid_features.shape}")

    train_features = train_features.reshape(train_features.shape[0], -1).T  # [n_features, n_samples]
    valid_features = valid_features.reshape(valid_features.shape[0], -1).T  # [n_features, n_samples]
    similarity_matrix_deep = cosine_similarity(train_features.T) # [n_samples, n_samples]
    logger.info(f"Deep feature cosine similarity matrix stats: min={similarity_matrix_deep.min():.4e}, max={similarity_matrix_deep.max():.4e}, mean={similarity_matrix_deep.mean():.4e}")

    H_deep, W_deep, Delta_deep = hypergraph_construction(similarity_matrix_deep, args.k)

    M_deep, U_deep = hypergraph_learning(train_features, train_labels, Delta_deep, args.lambda_hg, args.miu, logger)

    Omega_deep = np.trace(M_deep.T @ train_features @ Delta_deep @ train_features.T @ M_deep)
    logger.info(f"Deep feature Omega: {Omega_deep:.4e}")
    error_deep = np.linalg.norm(train_features.T @ M_deep - train_labels, ord='fro')**2
    logger.info(f"Deep feature training error: {error_deep:.4e}")
    regularization_deep = np.sum(np.sqrt(np.sum(M_deep**2, axis=1)))
    gamma_deep =  Omega_deep + args.lambda_hg * error_deep + args.miu * regularization_deep
    logger.info(f"Deep feature regularization: {regularization_deep:.4e}")
    logger.info(f"gamma (SPD features): {gamma_spd:.4e}, gamma (deep features): {gamma_deep:.4e}")

    spd_weight = 0.5 + (gamma_spd+gamma_deep)/(4*args.eta) - gamma_spd/(2*args.eta)
    deep_weight = 0.5 + (gamma_spd+gamma_deep)/(4*args.eta) - gamma_deep/(2*args.eta)
    logger.info(f"Feature weights - SPD: {spd_weight:.4f}, Deep: {deep_weight:.4f}")

    # Validation on source domain (vectorized)
    # valid_data_transformed: [n_features, n_samples], M: [n_features, n_classes]
    # valid_labels: either [n_samples, n_classes]
    preds = spd_weight * valid_data_transformed.T @ M + deep_weight * valid_features.T @ M_deep  # [n_samples, n_classes]
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(valid_labels, axis=1)
    num_correct = int(np.sum(pred_labels == true_labels))
    total_samples = valid_data_transformed.shape[1]
    source_accuracy = num_correct / total_samples if total_samples > 0 else 0.0
    logger.info(f"Source domain {source_date} validation accuracy: {source_accuracy:.4f}")
    spd_preds = valid_data_transformed.T @ M
    spd_pred_labels = np.argmax(spd_preds, axis=1)
    spd_num_correct = int(np.sum(spd_pred_labels == true_labels))
    spd_accuracy = spd_num_correct / total_samples if total_samples > 0 else 0.0
    logger.info(f"Source domain {source_date} validation accuracy (SPD features only): {spd_accuracy:.4f}")
    deep_preds = valid_features.T @ M_deep
    deep_pred_labels = np.argmax(deep_preds, axis=1)
    deep_num_correct = int(np.sum(deep_pred_labels == true_labels))
    deep_accuracy = deep_num_correct / total_samples if total_samples > 0 else 0.0
    logger.info(f"Source domain {source_date} validation accuracy (Deep features only): {deep_accuracy:.4f}")


    # Save source model
    source_model_path = os.path.join(output_path, f'source_model_{source_date}.npz')
    source_model_dict = {
        'M_spd': M,
        'U_spd': U,
        'H_spd': H,
        'W_spd': W,
        'Delta_spd': Delta,
        'M_deep': M_deep,
        'U_deep': U_deep,
        'H_deep': H_deep,
        'W_deep': W_deep,
        'Delta_deep': Delta_deep,
        'riemann_mean_spd': mean_spd,
        'ecog_step': args.ecog_step,
        'similarity': args.similarity,
        'alignment': args.alignment,
        'k': args.k,
        'lambda_hg': args.lambda_hg,
        'miu': args.miu,
        'eta': args.eta,
        'spd_weight': spd_weight,
        'deep_weight': deep_weight,
        'indices': indices
    }
    np.savez(source_model_path, **source_model_dict)
    logger.info(f"Saved source model to {source_model_path}")
    # Save validation results
    results_path = os.path.join(output_path, f'source_validation_results_{source_date}.npz')
    results_dict = {
        'true_labels': true_labels,
        'preds': preds,
        'source_accuracy': source_accuracy,
        'spd_preds': spd_preds,
        'spd_accuracy': spd_accuracy,
        'deep_preds': deep_preds,
        'deep_accuracy': deep_accuracy,
    }
    np.savez(results_path, **results_dict)
    logger.info(f"Saved source validation results to {results_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--source_dates', nargs='+', default=['20250325'])
    # parser.add_argument('--source_dates', nargs='+', default=['20250707_20250708_20250709'])
    parser.add_argument('--source_dates', nargs='+', default=['20250325_20250326_20250327'])
    # parser.add_argument('--source_dates', nargs='+', default=['T'])

    
    # parser.add_argument('--data_path', default='/media/ubuntu/Storage1/ecog_data/new_day_data/C07_BDY-S01_CF-8Fr/daily/beida_s01_angle_filtered.h5',
    parser.add_argument('--data_path', default='/media/ubuntu/Storage1/ecog_data/daily_bdy_20250908',
    # parser.add_argument('--data_path', default='bcic_1',
                        type=str, help='Path to the dataset')
    parser.add_argument('--output_path', default='/media/ubuntu/Storage1/ecog_data/Multi-feature_results',
                        type=str, help='Path to save outputs and models')
    # parser.add_argument('--checkpoint_root', type=str, default='/media/ubuntu/Storage1/ecog_data/OTTA_results/20260107_213624',
    parser.add_argument('--checkpoint_root', type=str, default='/media/ubuntu/Storage1/ecog_data/OTTA_results/20260303_171635',
                        help='Path to pre-trained deep model')
    # parser.add_argument('--indices_path', type=str, default='/media/ubuntu/Storage1/ecog_data/HOTL_results/20251211_094853/source_model_20250325.npz',
    # parser.add_argument('--indices_path', type=str, default='/media/ubuntu/Storage1/ecog_data/HOTL_results/20251210_135405/source_model_20250325_20250326_20250327.npz',
    parser.add_argument('--indices_path', type=str, default='/media/ubuntu/Storage1/ecog_data/OTTA_results/20260303_171635/training_history_20250325_20250326_20250327.json',
    # parser.add_argument('--indices_path', type=str, default='/media/ubuntu/Storage1/ecog_data/OTTA_results/20260203_145056/training_history_20250707_20250708_20250709.json',
                        help='Path to the npz file containing selected indices for training/validation split')
    
    parser.add_argument('--ecog_step', type=int, default=256, help='Step size for windowing ECoG data')

    parser.add_argument('--similarity', type=str, default='tangent_DM', choices=['tangent_cosine', 'SPD', 'tangent_DM', 'SPD_DM', 'cosine', 'DM'], 
                        help='Similarity metric for hypergraph construction')
    parser.add_argument('--alignment', type=str, default='Riemannian', choices=['Riemannian', 'Euclidean'],
                        help='Data alignment method before similarity computation')
    parser.add_argument('--normalize', default=False, help='NOT IMPLEMENTED. Whether to normalize deep features')
    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for deep feature extraction')
    parser.add_argument('--k', type=int, default=5, help='The number of nearest neighber')
    parser.add_argument('--lambda_hg', type=float, default=0.2, help='Regularization parameter lambda for spd hypergraph')
    parser.add_argument('--miu', type=float, default=0.2, help='Regularization parameter miu for spd hypergraph')
    parser.add_argument('--eta', type=float, default=1000000, help='Parameter eta for feature weighting')
    
    parser.add_argument('--device', type=str, default='cuda:1', help='Device for computation (cuda or cpu).')
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