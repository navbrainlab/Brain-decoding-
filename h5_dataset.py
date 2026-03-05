#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H5数据集类
用于从清洗好的h5数据中提取训练和测试数据，支持多种选择条件
在构建过程中进行滑窗操作，并支持保存处理后的数据
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
from data_pipline import OptimizedH5DailyDatasetController
import hashlib
import json
import h5py
from torch.nn.utils.rnn import pad_sequence


def rolling_window(a, window_size, step=1):
    """
    高效的滚动窗口函数，使用stride_tricks避免数据复制
    适用于2D数据: (n_channels, time)

    Parameters:
    -----------
    a : np.ndarray
        输入数据，形状为 (n_channels, time)
    window_size : int
        窗口大小
    step : int, optional
        窗口移动步长，默认为1

    Returns:
    --------
    np.ndarray
        滚动窗口数据，形状为 (n_windows, n_channels, window_size)
    """
    if a.shape[-1] < window_size:
        return np.array([]).reshape(0, a.shape[0], window_size)

    # 计算时间维度需要跨越的步数
    total_steps = a.shape[-1] - window_size
    n_windows = (total_steps // step) + 1

    # 使用as_strided创建窗口
    shape = (n_windows, window_size, a.shape[0])  # (窗口数, 窗长, 通道数)
    strides = (a.strides[-1] * step,  # 每个窗口在时间维度步进
               a.strides[-1],  # 窗内时间步进
               a.strides[0])  # 通道步进

    windowed = np.lib.stride_tricks.as_strided(
        a,
        shape=shape,
        strides=strides
    )
    return windowed.transpose(0, 2, 1)  # 转换为 (窗口数, 通道数, 窗长)


def slide_window(data: list, label: list, windows_size: int = 500, step: int = 100):
    """
    滑动窗口函数，使用优化的rolling_window实现
    """
    temp_slices = []
    labels_slices = []

    for i, temp in enumerate(data):
        # 使用rolling_window进行高效滑窗
        windows = rolling_window(temp, windows_size, step)

        # 根据步长选择窗口
        temp_slices.extend(windows)
        labels_slices.extend([label[i]] * len(windows))

    return np.array(temp_slices), np.array(labels_slices)


def slide_window_original(data: list, label: list, windows_size: int = 500, step: int = 100):
    """
    原始滑动窗口函数，用于对比
    """
    temp_slices = []
    labels_slices = []
    for i, temp in enumerate(data):
        for j in range(0, temp.shape[-1] - windows_size, step):
            temp_slices.append(temp[..., j:j + windows_size])
            labels_slices.append(label[i])
    return np.array(temp_slices), np.array(labels_slices)


class H5ECoGContinuesDataset(Dataset):
    """
    ECoG连续数据集类（仅处理movement数据）
    直接继承PyTorch Dataset，不依赖H5ECoGDataset
    特点：
    1. 不进行滑窗处理，保持数据的连续性
    2. 每个trial作为一个完整的样本
    3. 支持变长数据（每个trial的长度可能不同）
    4. 仅处理movement数据
    5. 支持随机截取最大长度（random_crop_max_length参数控制）
    6. 支持batch_size参数用于数据批处理
    """

    def __init__(self,
                 h5_file_path: str,
                 dates: Optional[List[str]] = None,
                 session_ids: Optional[List[int]] = None,
                 directions: Optional[List[float]] = None,
                 loop_types: Optional[List[str]] = None,
                 assist_size_range: Optional[Tuple[float, float]] = None,
                 success_only: Optional[bool] = None,
                 normalize: bool = True,
                 min_length: int = 100,  # 最小数据长度
                 max_length: Optional[int] = None,  # 最大数据长度，None表示无限制
                 enable_cache: bool = True,  # 是否启用缓存
                 cache_size: int = 500,  # 缓存大小
                 name: str = "H5ECoGContinuesDataset",
                 random_crop_max_length: Optional[int] = 5,  # 随机截取的最大长度
                 batch_size: int = 1,  # 批处理大小，默认为1
                 is_test: bool = False,  # 是否为测试集
                 return_rest: bool = False,
                 ):
        """
        初始化连续数据集（仅处理movement数据）

        Parameters:
        -----------
        h5_file_path : str
            h5文件路径
        dates : List[str], optional
            指定日期列表，None表示所有日期
        session_ids : List[int], optional
            指定session ID列表，None表示所有session
        directions : List[float], optional
            指定方向列表，None表示所有方向
        loop_types : List[str], optional
            指定循环类型列表，None表示所有类型
        assist_size_range : Tuple[float, float], optional
            辅助大小范围 (min, max)，None表示所有范围
        success_only : bool, optional
            是否只包含成功的trial，None表示所有trial
        normalize : bool
            是否对数据进行标准化
        min_length : int
            最小数据长度，短于此长度的trial将被过滤
        max_length : int, optional
            最大数据长度，长于此长度的trial将被截断
        enable_cache : bool
            是否启用缓存
        cache_size : int
            缓存大小
        name : str
            数据集名称
        random_crop_max_length : int, optional
            随机截取的最大长度，None表示不进行随机截取
        batch_size : int
            批处理大小，用于数据批处理
        """
        self.h5_file_path = h5_file_path
        self.dates = dates
        self.session_ids = session_ids
        self.directions = directions
        self.loop_types = loop_types
        self.assist_size_range = assist_size_range
        self.success_only = success_only
        self.normalize = normalize
        self.min_length = min_length
        self.max_length = max_length
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.name = name
        self.random_crop_max_length = random_crop_max_length
        self.batch_size = batch_size
        self.is_test = is_test  # 是否为测试集
        self.max_length = self.min_length + self.random_crop_max_length
        self.return_rest = return_rest

        # 加载数据
        self.data, self.labels, self.trajectory_list, self.rest, self.rotated_velocity, self.trial_info = self._load_data()

        print(f"连续数据集 '{self.name}' 加载完成:")
        print(f"  数据文件: {self.h5_file_path}")
        print(f"  样本数量: {len(self.data)}")
        print(f"  唯一标签: {np.unique(self.labels).tolist()}")
        print(f"  最小长度: {self.min_length}")
        print(f"  最大长度: {self.max_length}")
        if len(self.data) > 0:
            data_lengths = [data.shape[0] for data in self.data]
            print(f"  实际长度范围: {min(data_lengths)}-{max(data_lengths)}, 平均={np.mean(data_lengths):.1f}")
            self.max_data_lengths = max(data_lengths)

    def _load_data(self) -> Tuple[
        List[np.ndarray], np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        加载连续数据（仅处理movement数据）

        Returns:
        --------
        Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], List[Dict]]
            - data_list: movement数据列表
            - labels_array: 标签数组
            - trajectory_list: 轨迹数据列表
            - info_list: trial信息列表
        """
        print(f"正在加载连续数据从: {self.h5_file_path}")

        ctrl = OptimizedH5DailyDatasetController(self.h5_file_path, mode='r')
        all_dates = ctrl.list_days()

        # 确定要处理的日期
        if self.dates is not None:
            available_dates = [date for date in self.dates if date in all_dates]
            if len(available_dates) != len(self.dates):
                missing_dates = set(self.dates) - set(all_dates)
                print(f"警告: 以下日期不存在: {missing_dates}")
        else:
            available_dates = all_dates
        print(f"处理日期: {available_dates}")

        # 存储数据
        data_list = []
        labels_list = []
        rest_list = []
        trajectory_list = []  # 新增：单独的轨迹列表
        rotated_velocity_list = []  # 新增：单独的轨迹列表
        info_list = []

        for date in available_dates:
            print(f"  处理日期: {date}")
            trials = ctrl.get_trials(date)
            date_trial_count = 0
            date_trajectory_count = 0

            for trial_idx, trial in enumerate(trials):
                if not self._check_trial_conditions(trial):
                    continue

                # 只处理movement数据
                movement_data = trial['movement_data']
                trajectory_data = trial['trajectory']  # 读取轨迹数据
                rest_data = trial['rest_data']

                if movement_data is not None and movement_data.size > 0:
                    # 检查数据长度
                    data_length = movement_data.shape[0]

                    # 检查轨迹数据
                    if trajectory_data is not None and trajectory_data.size > 0:
                        trajectory_length = trajectory_data.shape[0]
                        date_trajectory_count += 1
                    else:
                        trajectory_data = np.zeros((1, 2), dtype=np.float32)  # 默认空轨迹
                        trajectory_length = 0

                    date_trial_count += 1
                    data_list.append(movement_data.astype(np.float32))
                    labels_list.append(int(trial['direction']))
                    rotated_velocity_list.append(trial['rotated_velocity'].astype(np.float32))
                    trajectory_list.append(trajectory_data.astype(np.float32))  # 添加到轨迹列表
                    rest_list.append(rest_data.astype(np.float32))
                    info_list.append({
                        'date': date,
                        'trial_idx': trial_idx,
                        'session_id': trial['session_id'],
                        'original_length': data_length,
                        'trajectory_length': trajectory_length
                    })

            print(f"    处理完成: {date_trial_count} 个trial，其中 {date_trajectory_count} 个有轨迹数据")

        ctrl.close()

        labels_array = np.array(labels_list, dtype=np.int64)

        print(f"  加载完成: {len(data_list)} 个样本")
        if len(data_list) > 0:
            data_lengths = [data.shape[0] for data in data_list]
            trajectory_lengths = [traj.shape[0] for traj in trajectory_list]
            print(
                f"  数据长度统计: 最小={min(data_lengths)}, 最大={max(data_lengths)}, 平均={np.mean(data_lengths):.1f}")
            print(
                f"  轨迹长度统计: 最小={min(trajectory_lengths)}, 最大={max(trajectory_lengths)}, 平均={np.mean(trajectory_lengths):.1f}")

        return data_list, labels_array, trajectory_list, rest_list, rotated_velocity_list, info_list

    def _check_trial_conditions(self, trial: Dict) -> bool:
        """
        检查trial是否满足过滤条件
        """
        # 检查session_ids
        if self.session_ids is not None and trial['session_id'] not in self.session_ids:
            return False

        # 检查directions
        if self.directions is not None and trial['direction'] not in self.directions:
            return False

        # 检查loop_types
        if self.loop_types is not None:
            loop_type = trial.get('loop_type', 'open')
            if isinstance(loop_type, bytes):
                loop_type = loop_type.decode('utf-8')
            if loop_type not in self.loop_types:
                return False

        # 检查assist_size_range
        if self.assist_size_range is not None:
            assist_size = trial.get('assist_size', 0.0)
            min_assist, max_assist = self.assist_size_range
            if not (min_assist <= assist_size <= max_assist):
                return False

        # 检查success_only
        if self.success_only is not None:
            trial_success = trial.get('trial_success', False)
            if self.success_only and not trial_success:
                return False
            elif not self.success_only and trial_success:
                return False

        return True

    def __getitem__(self, idx: int):
        """
        获取数据项，返回(data, label, date, trial_idx, session_id, trajectory)
        """
        # 获取数据
        data = torch.from_numpy(self.data[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # 标准化
        if self.normalize:
            # 对于2D数据 (channels, time)
            data = data - data.mean(-2, keepdim=True)

        # 获取trial信息
        trial_info = self.trial_info[idx]

        # 获取轨迹数据
        trajectory = torch.from_numpy(self.trajectory_list[idx]).float()
        velocity = torch.from_numpy(self.rotated_velocity[idx]).float()

        if trajectory.shape[0] != velocity.shape[0]:
            print('warning: trajectory.shape[0] != velocity.shape[0]')

        # 将原点 [0, 0] 插入到轨迹的起始位置
        # origin = torch.zeros((1, trajectory.shape[1]), dtype=trajectory.dtype, device=trajectory.device)
        # trajectory_origin = torch.cat([origin, trajectory], dim=0)
        # velocity_norm = torch.norm(torch.diff(trajectory_origin, dim=0), dim=0)
        # velocity = velocity * velocity_norm * 1.5

        # 根据min_length和random_crop_max_length，随机截取data的序列长度
        data_length = data.shape[0]
        crop_length = torch.randint(0, self.random_crop_max_length + 1, (1,)).item() + self.min_length

        # 如果数据长度不足min_length，则补零
        if not self.is_test:
            if crop_length < data_length:
                start_idx = np.random.randint(0, data_length - crop_length + 1)
                data = data[start_idx:start_idx + crop_length]
                velocity = velocity[start_idx:start_idx + crop_length]  # 轨迹长度比数据少1
                trajectory = trajectory[start_idx:start_idx + crop_length]
                data_length = data.shape[0]
        # 按照第一维pad到max_length长度，并给出mask
        pad_len = max(0, self.max_length - data.shape[0]) if not self.is_test else self.max_data_lengths - data.shape[0]

        label_length = self.max_length if not self.is_test else self.max_data_lengths
        label = torch.stack([label] * label_length, dim=0).long()
        if pad_len > 0:
            # 对data在第一维进行pad
            velocity = torch.nn.functional.pad(velocity, (0, 0, 0, pad_len), mode='constant', value=0.0)
            trajectory = torch.nn.functional.pad(trajectory, (0, 0, 0, pad_len), mode='constant', value=0.0)

            if self.return_rest:
                rest = self.rest[idx]
                if self.normalize:
                    # 对于2D数据 (channels, time)
                    rest = rest - rest.mean(-2, keepdims=True)

                if not self.is_test and rest.shape[0] > pad_len:
                    rest_length = torch.randint(0, rest.shape[0] - pad_len, (1,)).item()
                    rest = rest[rest_length:rest_length + pad_len]
                else:
                    rest = rest[:pad_len]
                data = torch.from_numpy(np.concatenate((data, rest), axis=0)).float()
                if pad_len > rest.shape[0]:
                    data = torch.nn.functional.pad(data.permute(1, 0, 2), (0, 0, 0, pad_len - rest.shape[0]),
                                                   mode='constant',
                                                   value=0.0).permute(1, 0, 2)
                label[-pad_len:] = -1
            else:
                data = torch.nn.functional.pad(data.permute(1, 0, 2), (0, 0, 0, pad_len), mode='constant',
                                               value=0.0).permute(1, 0, 2)

        # mask: 有效数据为1，pad为0
        mask = torch.arange(label_length) < data_length

        return data, label, velocity, trajectory, mask, trial_info

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)

    def get_length_statistics(self) -> Dict[str, float]:
        """
        获取数据长度统计信息
        """
        if len(self.data) == 0:
            return {
                'count': 0,
                'min_length': 0,
                'max_length': 0,
                'mean_length': 0,
                'std_length': 0,
                'median_length': 0
            }

        lengths = [data.shape[-1] for data in self.data]

        return {
            'count': len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'median_length': np.median(lengths)
        }

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        if not self.enable_cache:
            return {'cache_enabled': False}

        return {
            'cache_enabled': True,
            'cache_size': len(self._cache) if self._cache else 0,
            'max_cache_size': self.cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) if (
                                                                                              self._cache_hits + self._cache_misses) > 0 else 0
        }

    def clear_cache(self):
        """清空缓存"""
        if self.enable_cache and self._cache is not None:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0

    def __str__(self) -> str:
        """返回数据集的字符串表示"""
        return f"{self.name}(samples={len(self.data)}, min_length={self.min_length}, max_length={self.max_length})"


def create_continues_train_test_datasets(h5_file_path: str,
                                         train_dates: list,
                                         test_dates: list,
                                         train_session_ids: Optional[list] = None,
                                         test_session_ids: Optional[list] = None,
                                         directions: Optional[list] = None,
                                         loop_types: Optional[list] = None,
                                         assist_size_range: Optional[Tuple[float, float]] = None,
                                         success_only: Optional[bool] = None,
                                         normalize: bool = True,
                                         min_length: int = 100,
                                         max_length: Optional[int] = None,
                                         random_crop_max_length: int = 10,
                                         enable_cache: bool = True,
                                         cache_size: int = 1000,
                                         batch_size: int = 1,
                                         return_rest: bool = False) -> Tuple[
    H5ECoGContinuesDataset, H5ECoGContinuesDataset]:
    """
    创建连续训练和测试数据集

    Parameters:
    -----------
    h5_file_path : str
        h5文件路径
    train_dates : list
        训练日期列表
    test_dates : list
        测试日期列表
    train_session_ids : list, optional
        训练session ID列表
    test_session_ids : list, optional
        测试session ID列表
    directions : list, optional
        方向列表
    loop_types : list, optional
        循环类型列表
    assist_size_range : Tuple[float, float], optional
        辅助大小范围
    success_only : bool, optional
        是否只包含成功的trial
    normalize : bool
        是否标准化
    min_length : int
        最小数据长度
    max_length : int, optional
        最大数据长度
    enable_cache : bool
        是否启用缓存
    cache_size : int
        缓存大小
    batch_size : int
        批处理大小

    Returns:
    --------
    Tuple[H5ECoGContinuesDataset, H5ECoGContinuesDataset]
        训练数据集和测试数据集
    """
    print(f"创建连续训练和测试数据集:")
    print(f"  数据文件: {h5_file_path}")
    print(f"  训练日期: {train_dates}")
    print(f"  测试日期: {test_dates}")
    print(f"  批处理大小: {batch_size}")

    # 创建训练数据集
    train_dataset = H5ECoGContinuesDataset(
        h5_file_path=h5_file_path,
        dates=train_dates,
        session_ids=train_session_ids,
        directions=directions,
        loop_types=loop_types,
        assist_size_range=assist_size_range,
        success_only=success_only,
        normalize=normalize,
        min_length=min_length,
        max_length=max_length,
        random_crop_max_length=random_crop_max_length,
        enable_cache=enable_cache,
        cache_size=cache_size,
        batch_size=batch_size,
        return_rest=return_rest,
        name="H5ECoGContinuesTrainDataset"
    )

    # 创建测试数据集
    test_dataset = H5ECoGContinuesDataset(
        h5_file_path=h5_file_path,
        dates=test_dates,
        session_ids=test_session_ids,
        directions=directions,
        loop_types=loop_types,
        assist_size_range=assist_size_range,
        success_only=success_only,
        normalize=normalize,
        min_length=min_length,
        max_length=max_length,
        random_crop_max_length=random_crop_max_length,
        enable_cache=enable_cache,
        cache_size=cache_size,
        batch_size=batch_size,
        return_rest=return_rest,
        name="H5ECoGContinuesTestDataset"
    )

    return train_dataset, test_dataset


if __name__ == '__main__':
    # 测试代码
    print("=== H5ECoGDataset 测试 ===")

    # 测试文件路径
    h5_file = '/media/ubuntu/Storage1/ecog_data/new_day_data/C07_BDY-S01_CF-8Fr/daily/beida_s01_angle_filtered.h5'
    train_dataset, test_dataset = create_continues_train_test_datasets(
        h5_file_path='/media/ubuntu/Storage1/ecog_data/new_day_data/C07_BDY-S01_CF-8Fr/daily/beida_s01_angle_filtered.h5' ,
        train_dates=['20250701', '20250702'],
        test_dates=[],
    )

    print(len(train_dataset[0]))

    if os.path.exists(h5_file):
        print(f"找到测试文件: {h5_file}")
        ctrl = OptimizedH5DailyDatasetController(h5_file)
        dates = ctrl.list_days()
        ctrl.close()
