import os
import random
import numpy as np
from scipy.spatial.distance import cdist, pdist
from itertools import combinations

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from prefetch_generator import BackgroundGenerator
from typing import Dict, List, Union, Sequence, Optional, Iterator, Callable

DatasetType = Union[Dataset, IterableDataset]
PathType = Union[str, os.PathLike]


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, time_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(time_size, dtype=np.float32)
    grid_w = np.arange(time_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, time_size, time_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: tools list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def ch2index(ch_names: list) -> np.array:
    ch2idx = {
        'Fp1': 0,
        'Fp2': 1,
        'F3': 2,
        'F4': 3,
        'C3': 4,
        'C4': 5,
        'P3': 6,
        'P4': 7,
        'O1': 8,
        'O2': 9,
        'F7': 10,
        'F8': 11,
        'T7': 12,
        'T8': 13,
        'P7': 14,
        'P8': 15,
        'Fz': 16,
        'Cz': 17,
        'Pz': 18,
        'Oz': 19,
        'FC1': 20,
        'FC2': 21,
        'CP1': 22,
        'CP2': 23,
        'FC5': 24,
        'FC6': 25,
        'CP5': 26,
        'CP6': 27,
        'FT9': 28,
        'FT10': 29,
        'TP9': 30,
        'TP10': 31,
        'POz': 32,
    }
    idx = []
    for i, ch in enumerate(ch_names):
        idx.append(ch2idx[ch])
    return np.array(idx)


class SyncFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
                  for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[torch.distributed.get_rank()]
        return grad_out


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor


def get_mapping(root,mode='all'):
    import pandas as pd
    ele_mapping = pd.read_excel(os.path.join(root, "electrode_mapping.xlsx"), index_col=None, header=None)
    ele_mapping = ele_mapping.to_numpy()
    y_pos = np.arange(ele_mapping.shape[0])
    x_pos = np.arange(ele_mapping.shape[1])
    xx, yy = np.meshgrid(x_pos, y_pos)

    # drop 4 corners
    ele_mapping[0, 0] = -1
    ele_mapping[0, -1] = -1
    ele_mapping[-1, 0] = -1
    ele_mapping[-1, -1] = -1
    ele_mapping = ele_mapping.astype(np.int16) - 1

    # 根据mode选择索引（包含对角线）
    if mode == 'm1':
        select = xx >= yy  # 右上三角（含对角线）
    elif mode == 's1':
        select = xx <= yy # 左下三角（含对角线）
    else:
        select = np.ones_like(xx, dtype=bool)  # 全部选择
    xx = torch.from_numpy(xx).long()
    yy = torch.from_numpy(yy).long()
    select = select & (ele_mapping >=0 )
    ele_mapping = torch.from_numpy(ele_mapping[select]).long()
    return ele_mapping, xx, yy


class DataLoaderX(DataLoader):
    # A custom data loader class that inherits from DataLoader
    def __iter__(self, max_prefetch=10):
        # Overriding the __iter__ method of DataLoader to return a BackgroundGenerator
        # This is to enable data loading in the background to improve training performance
        return BackgroundGenerator(super().__iter__(), max_prefetch=max_prefetch)


class StatefulDistributedSampler(DistributedSampler):
    """
    Stateful distributed sampler for multi-stage training.
    """

    def __init__(
            self,
            dataset: DatasetType,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
            seed: int = 0,
            drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.start_index = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index:]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index


def prepare_dataloader(
        dataset,
        batch_size,
        shuffle=False,
        seed=1024,
        drop_last=False,
        pin_memory=False,
        num_workers=0,
        process_group: Optional[ProcessGroup] = None,
        **kwargs,
):
    r"""
    Prepare a dataloader for distributed training. The dataloader will be wrapped by
    `torch.utils.data.DataLoader` and `StatefulDistributedSampler`.


    Args:
        dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    _kwargs = kwargs.copy()
    process_group = process_group or _get_default_group()

    # sampler = StatefulDistributedSampler(
    #     dataset, num_replicas=process_group.size(), rank=process_group.rank(), shuffle=shuffle
    # )

    # Deterministic dataloader
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return DataLoaderX(
        dataset,
        batch_size=batch_size,
        # sampler=sampler,
        worker_init_fn=seed_worker,
        drop_last=drop_last,
        pin_memory=pin_memory,
        num_workers=num_workers,
        **_kwargs,
    )


def setup_distributed_dataloader(
        dataset: DatasetType,
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int = 1024,
        drop_last: bool = False,
        pin_memory: bool = False,
        num_workers: int = 0,
        collate_fn: Callable[[Sequence[Dict[str, Union[str, List[int]]]]], Dict[str, torch.Tensor]] = None,
        process_group: Optional[ProcessGroup] = None,
        **kwargs,
) -> DataLoader:
    """
    Setup dataloader for distributed training.
    """
    _kwargs = kwargs.copy()
    process_group = process_group or _get_default_group()
    sampler = StatefulDistributedSampler(
        dataset=dataset,
        num_replicas=process_group.size(),
        rank=process_group.rank(),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )

    # Deterministic dataloader
    def seed_worker(worker_id: int) -> None:
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return DataLoaderX(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        **_kwargs,
    )


class SyncFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor


def pairwise_mahalanobis_distance(x: torch.Tensor, y: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    """
    计算两组向量之间的马氏距离

    参数:
        x: 形状为 [n, d] 的张量，表示第一组向量
        y: 形状为 [m, d] 的张量，表示第二组向量
        covariance: 形状为 [d, d] 的张量，表示协方差矩阵

    返回:
        形状为 [n, m] 的张量，表示两两向量之间的马氏距离
    """
    # 计算协方差矩阵的逆
    covariance_inv = torch.inverse(covariance + torch.eye(covariance.size(0), device=covariance.device) * 1e-6)

    # 计算差值矩阵
    diff = x.unsqueeze(1) - y.unsqueeze(0)  # [n, m, d]

    # 计算马氏距离
    # 方法一：使用矩阵乘法手动实现
    # diff_reshaped = diff.reshape(-1, diff.size(-1))  # [n*m, d]
    # left = torch.matmul(diff_reshaped, covariance_inv)  # [n*m, d]
    # dist_squared = torch.sum(left * diff_reshaped, dim=1)  # [n*m]
    # dist = dist_squared.sqrt().reshape(diff.size(0), diff.size(1))  # [n, m]

    # 方法二：使用二次型形式（效率更高）
    dist = torch.sqrt(torch.einsum('ijk,kl,ijl->ij', diff, covariance_inv, diff))

    return dist


def compute_labels_mahalanobis(
        features: np.ndarray,
        labels: np.ndarray,
        device: str = None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unique = np.unique(labels)
    labels = torch.from_numpy(labels).long().to(device)
    dis = {}
    features = torch.from_numpy(features).float().to(device)
    # Normalize features
    combs = list(combinations(unique, 2))
    for comb in combs:
        # Calculate the Mahalanobis distance for each pair of unique labels
        u1, u2 = comb
        idx1 = torch.where(labels == u1)[0]
        idx2 = torch.where(labels == u2)[0]
        if len(idx1) == 0 or len(idx2) == 0:
            continue
        features1 = features[idx1]
        features2 = features[idx2]
        # pair_dis = cdist(features1, features2, metric='mahalanobis')
        data = torch.cat([features1, features2], dim=0)
        data = data - data.mean(dim=0, keepdim=True)
        covariance = torch.matmul(data.T, data) / (data.shape[0] - 1)
        pair_dis = pairwise_mahalanobis_distance(features1, features2, covariance)
        dis['{}_{}'.format(u1, u2)] = pair_dis.cpu().numpy().mean()
    dis['mean'] = np.mean([value for value in dis.values()], axis=0)
    return dis


def gen_new_aug(sample: torch.Tensor) -> torch.Tensor:
    """
    Generate new augmented samples by mixing the absolute values of the FFTs of two samples.
    Parameters
    ----------
    sample

    Returns
    -------

    """
    fftsamples = torch.fft.rfft(sample, dim=-1, norm='ortho')
    index = torch.randperm(sample.size(0)).to(fftsamples.device)  # Randomly select an index to mix with
    mixing_coeff = (0.9 - 1) * torch.rand(1, device=fftsamples.device) + 1
    abs_fft = torch.abs(fftsamples)
    phase_fft = torch.angle(fftsamples)
    mixed_abs = abs_fft * mixing_coeff + (1 - mixing_coeff) * abs_fft[index]
    z =  torch.polar(mixed_abs, phase_fft) # Go back to fft
    mixed_samples_time = torch.fft.irfft(z, dim=-1, norm='ortho')
    return mixed_samples_time


