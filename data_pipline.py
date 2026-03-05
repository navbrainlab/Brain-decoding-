###############################################
# 简化版：每天一个group，支持不等长数据的trial存储
###############################################
import h5py
import numpy as np
import os
import time
import pickle

import sys
sys.path.append('/media/ubuntu/Storage1/ecog_data/new_day_data/C07_BDY-S01_CF-8Fr/python_load_intan_rhd_wireless_20241215')
# from load_intan_rhd_format import hunman_center_out
# from assis import assis
# from utils.tools import compute_angle, compute_distance


class H5DailyDatasetController:
    """
    简化版h5py控制器：每天一个group，支持不等长的trial数据存储。
    使用pickle序列化不等长数组，支持高效批量操作。
    """

    def __init__(self, file_path, mode='a', dtype=np.float64):
        """
        初始化控制器
        :param file_path: h5文件路径
        :param mode: 文件打开模式
        :param dtype: 数据类型
        """
        self.file = h5py.File(file_path, mode)
        self.dtype = dtype

        # 确保days组存在
        if 'days' not in self.file:
            self.file.create_group('days')

    def add_day(self, date, meta=None):
        """
        新建一天，创建可扩展的dataset
        :param date: 日期字符串，如'20240420'
        :param meta: 可选的元信息字典
        """
        if str(date) in self.file['days']:  # type: ignore
            raise ValueError(f"日期 {date} 已存在")

        day_group = self.file['days'].create_group(str(date))  # type: ignore

        # 存储元信息
        if meta:
            for k, v in meta.items():
                day_group.attrs[k] = v

        # 创建空的可扩展dataset
        # 使用变长字节数组存储序列化的不等长数组
        dt = h5py.special_dtype(vlen=np.dtype('uint8'))

        # 优化chunking和压缩设置
        day_group.create_dataset(
            'movement_data',
            shape=(0,),
            maxshape=(None,),
            dtype=dt,
            chunks=(100,),  # 每100个trial一个chunk
        )
        day_group.create_dataset(
            'rest_data',
            shape=(0,),
            maxshape=(None,),
            dtype=dt,
            chunks=(100,),  # 每100个trial一个chunk
        )
        day_group.create_dataset(
            'direction',
            shape=(0,),
            maxshape=(None,),
            dtype=self.dtype,
            chunks=(1000,)  # 每1000个trial一个chunk
        )
        day_group.create_dataset(
            'trajectory',
            shape=(0,),
            maxshape=(None,),
            dtype=dt,
            chunks=(100,),  # 每100个trial一个chunk
        )
        day_group.create_dataset(
            'session_id',
            shape=(0,),
            maxshape=(None,),
            dtype='i4',
            chunks=(1000,)  # 每1000个trial一个chunk
        )
        day_group.create_dataset(
            'assist_size',
            shape=(0,),
            maxshape=(None,),
            dtype=self.dtype,
            chunks=(1000,)  # 每1000个trial一个chunk
        )
        day_group.create_dataset(
            'loop_type',
            shape=(0,),
            maxshape=(None,),
            dtype='S10',  # 字符串类型，最大10个字符
            chunks=(1000,)  # 每1000个trial一个chunk
        )
        day_group.create_dataset(
            'trial_success',
            shape=(0,),
            maxshape=(None,),
            dtype='bool',  # 布尔类型
            chunks=(1000,)  # 每1000个trial一个chunk
        )
        # 存储形状信息
        day_group.create_dataset(
            'movement_shapes',
            shape=(0, 3),
            maxshape=(None, 3),
            dtype='i4',
            chunks=(1000, 3)  # 每1000个trial一个chunk
        )
        day_group.create_dataset(
            'rest_shapes',
            shape=(0, 3),
            maxshape=(None, 3),
            dtype='i4',
            chunks=(1000, 3)  # 每1000个trial一个chunk
        )
        day_group.create_dataset(
            'trajectory_shapes',
            shape=(0, 3),
            maxshape=(None, 3),
            dtype='i4',
            chunks=(1000, 3)  # 每1000个trial一个chunk
        )
        day_group.create_dataset(
            'trajectory_angle',
            shape=(0,),
            maxshape=(None,),
            dtype=self.dtype,
            chunks=(1000,)  # 每1000个trial一个chunk
        )
        day_group.create_dataset(
            'trajectory_distance',
            shape=(0,),
            maxshape=(None,),
            dtype=self.dtype,
            chunks=(1000,)  # 每1000个trial一个chunk
        )
        day_group.create_dataset(
            'rotated_velocity',
            shape=(0,),
            maxshape=(None,),
            dtype=dt,
            chunks=(100,)  # 每100个trial一个chunk
        )
        day_group.create_dataset(
            'rotated_velocity_shapes',
            shape=(0, 3),
            maxshape=(None, 3),
            dtype='i4',
            chunks=(1000, 3)  # 每1000个trial一个chunk
        )

    def _serialize_array(self, arr):
        """序列化数组为字节数组"""
        return np.frombuffer(pickle.dumps(arr.astype(self.dtype)), dtype=np.uint8)

    def _deserialize_array(self, data_bytes):
        """反序列化字节数组为数组"""
        return pickle.loads(data_bytes.tobytes())

    def append_trial(self, date, movement_data, rest_data, direction, trajectory, session_id=0, assist_size=0.0,
                     loop_type='open', trial_success=False, trajectory_angle=0.0, trajectory_distance=0.0,
                     rotated_velocity=None):
        """
        向某天追加一个trial数据
        :param date: 日期
        :param movement_data: 运动数据（不等长）
        :param rest_data: 静息数据（不等长）
        :param direction: 方向
        :param trajectory: 轨迹（不等长）
        :param session_id: session编号
        :param assist_size: 辅助大小（0-1浮点数）
        :param loop_type: 循环类型（'open'或'closed'）
        :param trial_success: trial是否成功
        :param trajectory_angle: 轨迹角度（度）
        :param trajectory_distance: 轨迹距离（归一化）
        :param rotated_velocity: 旋转后的速度向量（不等长）
        """
        day_group = self.file['days'][str(date)]  # type: ignore
        n = day_group['movement_data'].shape[0]  # type: ignore

        # 扩展所有dataset
        day_group['movement_data'].resize((n + 1,))  # type: ignore
        day_group['rest_data'].resize((n + 1,))  # type: ignore
        day_group['direction'].resize((n + 1,))  # type: ignore
        day_group['trajectory'].resize((n + 1,))  # type: ignore
        day_group['session_id'].resize((n + 1,))  # type: ignore
        day_group['assist_size'].resize((n + 1,))  # type: ignore
        day_group['loop_type'].resize((n + 1,))  # type: ignore
        day_group['trial_success'].resize((n + 1,))  # type: ignore
        day_group['movement_shapes'].resize((n + 1, 3))  # type: ignore
        day_group['rest_shapes'].resize((n + 1, 3))  # type: ignore
        day_group['trajectory_shapes'].resize((n + 1, 3))  # type: ignore
        day_group['trajectory_angle'].resize((n + 1,))  # type: ignore
        day_group['trajectory_distance'].resize((n + 1,))  # type: ignore
        day_group['rotated_velocity'].resize((n + 1,))  # type: ignore
        day_group['rotated_velocity_shapes'].resize((n + 1, 3))  # type: ignore

        # 序列化并写入数据
        day_group['movement_data'][n] = self._serialize_array(movement_data)  # type: ignore
        day_group['rest_data'][n] = self._serialize_array(rest_data)  # type: ignore
        day_group['direction'][n] = direction  # type: ignore
        day_group['trajectory'][n] = self._serialize_array(trajectory)  # type: ignore
        day_group['session_id'][n] = session_id  # type: ignore
        day_group['assist_size'][n] = assist_size  # type: ignore
        day_group['loop_type'][n] = loop_type.encode('utf-8')  # type: ignore
        day_group['trial_success'][n] = trial_success  # type: ignore

        # 存储形状信息（自动适配2维和3维）
        ms = list(movement_data.shape)
        if len(ms) == 2:
            ms = [1] + ms
        day_group['movement_shapes'][n] = ms  # type: ignore
        rs = list(rest_data.shape)
        if len(rs) == 2:
            rs = [1] + rs
        day_group['rest_shapes'][n] = rs  # type: ignore
        ts = list(trajectory.shape)
        if len(ts) == 2:
            ts = [1] + ts
        day_group['trajectory_shapes'][n] = ts  # type: ignore
        day_group['trajectory_angle'][n] = trajectory_angle  # type: ignore
        day_group['trajectory_distance'][n] = trajectory_distance  # type: ignore

        # 处理旋转速度向量
        if rotated_velocity is not None:
            day_group['rotated_velocity'][n] = self._serialize_array(rotated_velocity)  # type: ignore
            rvs = list(rotated_velocity.shape)
            if len(rvs) == 2:
                rvs = [1] + rvs
            day_group['rotated_velocity_shapes'][n] = rvs  # type: ignore
        else:
            # 如果没有旋转速度数据，存储空数组
            empty_velocity = np.zeros((1, 2), dtype=self.dtype)
            day_group['rotated_velocity'][n] = self._serialize_array(empty_velocity)  # type: ignore
            day_group['rotated_velocity_shapes'][n] = [1, 1, 2]  # type: ignore

    def append_trials_batch(self, date, trials_data):
        """
        批量追加多个trial数据，性能更优
        :param date: 日期
        :param trials_data: 列表，每个元素为dict包含movement_data, rest_data, direction, trajectory, session_id, assist_size, loop_type, trajectory_angle, trajectory_distance
        """
        if not trials_data:
            return

        day_group = self.file['days'][str(date)]  # type: ignore
        n = day_group['movement_data'].shape[0]  # type: ignore
        new_n = n + len(trials_data)

        # 一次性扩展所有dataset
        day_group['movement_data'].resize((new_n,))  # type: ignore
        day_group['rest_data'].resize((new_n,))  # type: ignore
        day_group['direction'].resize((new_n,))  # type: ignore
        day_group['trajectory'].resize((new_n,))  # type: ignore
        day_group['session_id'].resize((new_n,))  # type: ignore
        day_group['assist_size'].resize((new_n,))  # type: ignore
        day_group['loop_type'].resize((new_n,))  # type: ignore
        day_group['trial_success'].resize((new_n,))  # type: ignore
        day_group['movement_shapes'].resize((new_n, 3))  # type: ignore
        day_group['rest_shapes'].resize((new_n, 3))  # type: ignore
        day_group['trajectory_shapes'].resize((new_n, 3))  # type: ignore
        day_group['trajectory_angle'].resize((new_n,))  # type: ignore
        day_group['trajectory_distance'].resize((new_n,))  # type: ignore
        day_group['rotated_velocity'].resize((new_n,))  # type: ignore
        day_group['rotated_velocity_shapes'].resize((new_n, 3))  # type: ignore

        # 批量写入数据
        for i, trial in enumerate(trials_data):
            idx = n + i
            day_group['movement_data'][idx] = self._serialize_array(trial['movement_data'])  # type: ignore
            day_group['rest_data'][idx] = self._serialize_array(trial['rest_data'])  # type: ignore
            day_group['direction'][idx] = trial['direction']  # type: ignore
            day_group['trajectory'][idx] = self._serialize_array(trial['trajectory'])  # type: ignore
            day_group['session_id'][idx] = trial.get('session_id', 0)  # type: ignore
            day_group['assist_size'][idx] = trial.get('assist_size', 0.0)  # type: ignore
            day_group['loop_type'][idx] = trial.get('loop_type', 'open').encode('utf-8')  # type: ignore
            day_group['trial_success'][idx] = trial.get('trial_success', False)  # type: ignore

            # 存储形状信息（自动适配2维和3维）
            ms = list(trial['movement_data'].shape)
            if len(ms) == 2:
                ms = [1] + ms
            day_group['movement_shapes'][idx] = ms  # type: ignore
            rs = list(trial['rest_data'].shape)
            if len(rs) == 2:
                rs = [1] + rs
            day_group['rest_shapes'][idx] = rs  # type: ignore
            ts = list(trial['trajectory'].shape)
            if len(ts) == 2:
                ts = [1] + ts
            day_group['trajectory_shapes'][idx] = ts  # type: ignore
            day_group['trajectory_angle'][idx] = trial.get('trajectory_angle', 0.0)  # type: ignore
            day_group['trajectory_distance'][idx] = trial.get('trajectory_distance', 0.0)  # type: ignore

            # 处理旋转速度向量
            rotated_velocity = trial.get('rotated_velocity', None)
            if rotated_velocity is not None:
                day_group['rotated_velocity'][idx] = self._serialize_array(rotated_velocity)  # type: ignore
                rvs = list(rotated_velocity.shape)
                if len(rvs) == 2:
                    rvs = [1] + rvs
                day_group['rotated_velocity_shapes'][idx] = rvs  # type: ignore
            else:
                # 如果没有旋转速度数据，存储空数组
                empty_velocity = np.zeros((1, 2), dtype=self.dtype)
                day_group['rotated_velocity'][idx] = self._serialize_array(empty_velocity)  # type: ignore
                day_group['rotated_velocity_shapes'][idx] = [1, 1, 2]  # type: ignore

    def get_trial(self, date, idx):
        """
        获取某天第idx个trial所有数据
        :param date: 日期
        :param idx: trial索引
        :return: dict包含所有trial数据
        """
        day_group = self.file['days'][str(date)]
        return {
            'movement_data': self._deserialize_array(day_group['movement_data'][idx]),
            'rest_data': self._deserialize_array(day_group['rest_data'][idx]),
            'direction': day_group['direction'][idx],
            'trajectory': self._deserialize_array(day_group['trajectory'][idx]),
            'session_id': day_group['session_id'][idx],
            'assist_size': day_group['assist_size'][idx],
            'loop_type': day_group['loop_type'][idx].decode('utf-8'),  # 字节解码为字符串
            'trial_success': day_group['trial_success'][idx],
            'movement_shape': day_group['movement_shapes'][idx],
            'rest_shape': day_group['rest_shapes'][idx],
            'trajectory_shape': day_group['trajectory_shapes'][idx],
            'trajectory_angle': day_group['trajectory_angle'][idx],
            'trajectory_distance': day_group['trajectory_distance'][idx],
            'rotated_velocity': self._deserialize_array(day_group['rotated_velocity'][idx]),
            'rotated_velocity_shape': day_group['rotated_velocity_shapes'][idx]
        }

    def get_trials(self, date, idxs=None):
        """
        批量获取某天多个trial数据
        :param date: 日期
        :param idxs: 索引列表，None表示获取全部
        :return: 列表，每个元素为trial数据dict
        """
        day_group = self.file['days'][str(date)]
        if idxs is None:
            idxs = range(day_group['movement_data'].shape[0])

        return [self.get_trial(date, i) for i in idxs]

    def get_trials_by_session(self, date, session_id):
        """
        获取某天指定session的所有trial
        :param date: 日期
        :param session_id: session编号
        :return: 列表，每个元素为trial数据dict
        """
        day_group = self.file['days'][str(date)]
        session_ids = day_group['session_id'][:]
        session_mask = session_ids == session_id
        session_indices = np.where(session_mask)[0]

        return self.get_trials(date, session_indices)

    def list_days(self):
        """返回所有日期列表"""
        return list(self.file['days'].keys())

    def get_day_meta(self, date):
        """获取某天的元信息"""
        day_group = self.file['days'][str(date)]
        meta = dict(day_group.attrs)
        print(f"日期 {date} 的元信息:")
        for key, value in meta.items():
            print(f"  {key}: {value}")
        return meta

    def get_day_stats(self, date):
        """获取某天的统计信息"""
        day_group = self.file['days'][str(date)]
        n_trials = day_group['movement_data'].shape[0]
        unique_sessions = np.unique(day_group['session_id'][:])

        # 获取当天的元信息
        day_meta = dict(day_group.attrs)

        # 获取所有trial的详细信息
        directions = day_group['direction'][:]
        assist_sizes = day_group['assist_size'][:]
        loop_types = [lt.decode('utf-8') if isinstance(lt, bytes) else lt for lt in day_group['loop_type'][:]]
        trial_successes = day_group['trial_success'][:]
        trajectory_angles = day_group['trajectory_angle'][:]
        trajectory_distances = day_group['trajectory_distance'][:]

        # 统计方向分布
        unique_directions = np.unique(directions)
        direction_counts = {int(d): int(np.sum(directions == d)) for d in unique_directions}

        # 统计成功率
        success_rate = float(np.mean(trial_successes))

        # 统计辅助大小分布
        unique_assist_sizes = np.unique(assist_sizes)
        assist_size_counts = {float(asize): int(np.sum(assist_sizes == asize)) for asize in unique_assist_sizes}

        # 统计loop类型分布
        unique_loop_types = list(set(loop_types))
        loop_type_counts = {lt: int(loop_types.count(lt)) for lt in unique_loop_types}

        # 统计轨迹角度和距离
        trajectory_angle_mean = float(np.mean(trajectory_angles))
        trajectory_angle_std = float(np.std(trajectory_angles))
        trajectory_distance_mean = float(np.mean(trajectory_distances))
        trajectory_distance_std = float(np.std(trajectory_distances))

        return {
            'n_trials': n_trials,
            'n_sessions': len(unique_sessions),
            'sessions': unique_sessions.tolist(),
            'day_meta': day_meta,
            'direction_distribution': direction_counts,
            'success_rate': success_rate,
            'assist_size_distribution': assist_size_counts,
            'loop_type_distribution': loop_type_counts,
            'unique_directions': unique_directions.tolist(),
            'unique_assist_sizes': unique_assist_sizes.tolist(),
            'unique_loop_types': unique_loop_types,
            'trajectory_angle_mean': trajectory_angle_mean,
            'trajectory_angle_std': trajectory_angle_std,
            'trajectory_distance_mean': trajectory_distance_mean,
            'trajectory_distance_std': trajectory_distance_std
        }

    def get_file_stats(self):
        """获取整个文件的统计信息"""
        stats = {
            'total_days': len(self.list_days()),
            'total_trials': 0,
            'total_sessions': 0,
            'file_size_mb': 0
        }

        for date in self.list_days():
            day_stats = self.get_day_stats(date)
            stats['total_trials'] += day_stats['n_trials']
            stats['total_sessions'] += day_stats['n_sessions']

        # 获取文件大小
        self.file.flush()
        stats['file_size_mb'] = self.file.id.get_filesize() / (1024 * 1024)

        return stats

    def close(self):
        """关闭文件"""
        self.file.close()


class OptimizedH5DailyDatasetController(H5DailyDatasetController):
    """
    优化版本的H5DailyDatasetController，通过缓存和预加载提高读取速度
    """

    def __init__(self, file_path, mode='a', dtype=np.float64, cache_size=500):
        """
        初始化优化控制器
        :param file_path: h5文件路径
        :param mode: 文件打开模式
        :param dtype: 数据类型
        :param cache_size: 缓存大小（trial数量）
        """
        super().__init__(file_path, mode, dtype)
        self.cache_size = cache_size

        # 缓存系统
        self._trial_cache = {}  # 缓存trial数据
        self._meta_cache = {}  # 缓存元数据

        # 预加载元数据
        self._preload_metadata()

    def _preload_metadata(self):
        """预加载所有元数据到内存"""
        for date in self.list_days():
            day_group = self.file['days'][str(date)]
            self._meta_cache[date] = {
                'session_ids': day_group['session_id'][:],
                'directions': day_group['direction'][:],
                'assist_sizes': day_group['assist_size'][:],
                'loop_types': day_group['loop_type'][:],
                'trial_successes': day_group['trial_success'][:],
                'trajectory_angles': day_group['trajectory_angle'][:],
                'trajectory_distances': day_group['trajectory_distance'][:],
                'n_trials': day_group['movement_data'].shape[0]
            }

    def _get_cache_key(self, date, idx):
        """生成缓存键"""
        return f"{date}_{idx}"

    def _add_to_cache(self, date, idx, trial_data):
        """添加到缓存"""
        cache_key = self._get_cache_key(date, idx)
        self._trial_cache[cache_key] = trial_data

        # 如果缓存满了，删除最旧的
        if len(self._trial_cache) > self.cache_size:
            oldest_key = next(iter(self._trial_cache))
            del self._trial_cache[oldest_key]

    def _get_from_cache(self, date, idx):
        """从缓存获取"""
        cache_key = self._get_cache_key(date, idx)
        return self._trial_cache.get(cache_key)

    def get_trial(self, date, idx):
        """
        获取指定trial（带缓存）
        """
        # 先检查缓存
        cached_data = self._get_from_cache(date, idx)
        if cached_data is not None:
            return cached_data

        # 缓存未命中，从文件读取
        trial_data = super().get_trial(date, idx)

        # 添加到缓存
        self._add_to_cache(date, idx, trial_data)

        return trial_data

    def get_trials_by_session(self, date, session_id):
        """
        优化的按session读取
        """
        # 使用预加载的元数据快速找到session的trial索引
        session_ids = self._meta_cache[date]['session_ids']
        session_mask = session_ids == session_id
        session_indices = np.where(session_mask)[0]

        return self.get_trials(date, session_indices)

    def get_trials_by_direction_range(self, date, min_direction, max_direction):
        """
        按方向范围快速查询trial
        """
        directions = self._meta_cache[date]['directions']
        direction_mask = (directions >= min_direction) & (directions <= max_direction)
        direction_indices = np.where(direction_mask)[0]

        return self.get_trials(date, direction_indices)

    def get_trials_by_assist_size_range(self, date, min_assist_size, max_assist_size):
        """
        按辅助大小范围快速查询trial
        """
        assist_sizes = self._meta_cache[date]['assist_sizes']
        assist_mask = (assist_sizes >= min_assist_size) & (assist_sizes <= max_assist_size)
        assist_indices = np.where(assist_mask)[0]

        return self.get_trials(date, assist_indices)

    def get_trials_by_loop_type(self, date, loop_type):
        """
        按循环类型快速查询trial
        """
        loop_types = self._meta_cache[date]['loop_types']
        # 解码字节数组为字符串进行比较
        loop_mask = np.array([lt.decode('utf-8') == loop_type for lt in loop_types])
        loop_indices = np.where(loop_mask)[0]

        return self.get_trials(date, loop_indices)

    def get_trials_by_success(self, date, success=True):
        """
        按trial成功与否快速查询
        """
        trial_successes = self._meta_cache[date]['trial_successes']
        success_mask = trial_successes == success
        success_indices = np.where(success_mask)[0]

        return self.get_trials(date, success_indices)

    def get_trials_by_trajectory_angle_range(self, date, min_angle, max_angle):
        """
        按轨迹角度范围快速查询trial
        """
        trajectory_angles = self._meta_cache[date]['trajectory_angles']
        angle_mask = (trajectory_angles >= min_angle) & (trajectory_angles <= max_angle)
        angle_indices = np.where(angle_mask)[0]

        return self.get_trials(date, angle_indices)

    def get_trials_by_trajectory_distance_range(self, date, min_distance, max_distance):
        """
        按轨迹距离范围快速查询trial
        """
        trajectory_distances = self._meta_cache[date]['trajectory_distances']
        distance_mask = (trajectory_distances >= min_distance) & (trajectory_distances <= max_distance)
        distance_indices = np.where(distance_mask)[0]

        return self.get_trials(date, distance_indices)

    def get_trials_by_trajectory_distance_range_and_angle_range(self, date, min_distance, max_distance, min_angle,
                                                                max_angle):
        """
        按轨迹距离范围快速查询trial
        """
        trajectory_distances = self._meta_cache[date]['trajectory_distances']
        trajectory_angles = self._meta_cache[date]['trajectory_angles']
        distance_mask = (trajectory_distances >= min_distance) & (trajectory_distances <= max_distance) & (
                trajectory_angles >= min_angle) & (trajectory_angles <= max_angle)
        distance_indices = np.where(distance_mask)[0]

        return self.get_trials(date, distance_indices)

    def clear_cache(self):
        """清空缓存"""
        self._trial_cache.clear()

    def get_cache_stats(self):
        """获取缓存统计信息"""
        return {
            'cache_size': len(self._trial_cache),
            'max_cache_size': self.cache_size
        }


# target_pos = {
#     0: [1, 0],
#     2: [0, 1],
#     4: [-1, 0],
#     6: [0, -1],
#     1: [0.7071, 0.7071],
#     3: [-0.7071, 0.7071],
#     5: [-0.7071, -0.7071],
#     7: [0.7071, -0.7071],
#     8: [0, 0]
# }
target_pos = {
    0: [1, 0],
    2: [0, 1],
    4: [-1, 0],
    6: [0, -1],
    1: [0.7071, 0.7071],
    3: [-0.7071, 0.7071],
    5: [-0.7071, -0.7071],
    7: [0.7071, -0.7071],
    8: [0, 0]
}

if __name__ == '__main__':
    # test_basic_functionality()
    # test_performance_comparison()

    # 创建控制器
    test_file = '/media/ubuntu/Storage1/ecog_data/new_day_data/C07_BDY-S01_CF-8Fr/daily/beida_s01_trajectory.h5'
    ctrl = OptimizedH5DailyDatasetController(test_file)
    # ctrl.get_day_stats('20250620')

    root = r'/media/ubuntu/Storage1/ecog_data/new_day_data/C07_BDY-S01_CF-8Fr/raw'
    dailies = [os.path.join(root, i) for i in os.listdir(root) if os.path.isdir(os.path.join(root, i))]
    dailies.sort()
    # 仅选择用户指定的日期范围：
    # - 在 20250701 之后 且 在 20250715 之前（含边界）
    # - 在 20250804 之后 且 在 20250825 之前（含边界）
    def _in_selected_ranges(path):
        name = os.path.basename(path)
        try:
            date_val = int(name)
        except ValueError:
            return False
        return (20250701 <= date_val <= 20250713) or (20250804 <= date_val <= 20250814) or (20250816 <= date_val <= 20250825)

    dailies = [d for d in dailies if _in_selected_ranges(d)]
    print(f"选中的daily日期: {[os.path.basename(d) for d in dailies]}")
    # dailies = ['/media/ubuntu/Storage1/ecog_data/new_day_data/C07_BDY-S01_CF-8Fr/raw/20250815']
    # 处理每个daily文件夹
    for daily in dailies:
        daily_name = os.path.basename(daily)
        print(f"处理daily文件夹: {daily_name}")

        # 为这一天创建h5组
        try:
            ctrl.add_day(daily_name, {
                'experimenter': 'tiantan',
                'subject': 'S02',
                'notes': f'Daily data from {daily_name}'
            })
        except ValueError:
            print(f"日期 {daily_name} 已存在，跳过创建")
            continue

        if daily_name == '20250827':
            a = 1

        # 处理该daily文件夹下的所有实验
        for exp_folder in os.listdir(daily):

            exp_path = os.path.join(daily, exp_folder)
            if not os.path.isdir(exp_path):
                continue

            exp_folder_wo_date = '_'.join(exp_folder.split('_')[:3])

            session_id, loop_type = exp_folder.split('_')[:2]

            print(f"  处理实验文件夹: {exp_folder}")

            # 查找.rhd文件
            rhd_files = [f for f in os.listdir(exp_path) if f.endswith('.rhd')]
            if not rhd_files:
                print(f"    未找到.rhd文件，跳过")
                continue

            rhd_file = rhd_files[0]
            rhd_path = os.path.join(exp_path, rhd_file)

            # 查找notes.txt文件
            notes_path = os.path.join(exp_path, 'notes.txt')
            if not os.path.exists(notes_path):
                print(f"    未找到notes.txt文件，跳过")
                continue

            # 查找.log文件（可选）
            log_files = [f for f in os.listdir(exp_path) if f.endswith('statereader.log')]
            log_path = log_files[0] if log_files else None
            log_path = None if log_path is None else os.path.join(exp_path, log_path)

            # 调用hunman_center_out函数处理数据
            trial_data, labels, pos, trial_rest_data, rest_labels, success_failure = hunman_center_out(
                rhd_path, notes_path, log=log_path
            )

            print(f"    成功处理数据: {len(trial_data)} 个trial, {len(trial_rest_data)} 个rest trial")

            # 准备批量写入的数据
            trials_data = []

            num_directions = list((set(labels)))
            if len(num_directions) != 4 and len(num_directions) != 8:
                print(f"    不支持的label数量: {num_directions}")
            # 处理运动trial和rest数据
            for i, (data, label) in enumerate(zip(trial_data, labels)):
                # 计算方向（从label映射到角度）
                direction = int(label * 90 if max(num_directions) == 3 else label * 45)  # 假设每个label对应45度的方向

                # 获取轨迹数据（如果有的话）
                trajectory = np.zeros((1, 1))  # 默认轨迹
                trajectory_angle = 0.0
                trajectory_distance = 0.0
                target = np.array(target_pos[label]) * 16

                origin = np.array([[0, 0]])
                if pos is not None and i < len(pos):
                    trajectory = np.array(pos[i])
                    trajectory_with_origin = np.concatenate((origin, trajectory), axis=0)
                    velocity = np.diff(trajectory_with_origin, axis=0)
                    velocity_norm = np.linalg.norm(velocity, axis=-1, keepdims=True)

                    target_batch = np.array([target] * len(velocity))
                    diff_degree = np.arctan2(target_batch[:, 1] - trajectory[:, 1], target_batch[:, 0] - trajectory[:, 0])
                    direction_vector = np.array([np.cos(diff_degree), np.sin(diff_degree)]).T
                    rotated_velocity = direction_vector * velocity_norm

                    # 计算轨迹角度和距离
                    if trajectory.shape[0] > 1 and trajectory.shape[1] == 2:  # 确保轨迹有足够的点和正确的维度
                        trajectory_angle = compute_angle(trajectory, label)
                        trajectory_distance = compute_distance(trajectory)
                else:
                    # 如果没有轨迹数据，设置默认值
                    rotated_velocity = np.zeros((1, 2), dtype=np.float64)
                    trajectory_angle = 0.0
                    trajectory_distance = 0.0

                assist_size = 0.0
                if daily_name in assis.keys():
                    if exp_folder_wo_date in assis[daily_name].keys():
                        assist_size = assis[daily_name][exp_folder_wo_date]['assistan_prob']

                if trajectory.shape[0] != rotated_velocity.shape[0]:
                    print('warning: trajectory.shape[0] != velocity.shape[0]')

                trials_data.append({
                    'movement_data': data.astype(np.float64),
                    'rest_data': trial_rest_data[i].astype(np.float64) if trial_rest_data[i] is not None else np.zeros(
                        (1, 1)).astype(np.float64),
                    'direction': direction,
                    'trajectory': trajectory.astype(np.float64),
                    'session_id': session_id,
                    'assist_size': assist_size,
                    'loop_type': loop_type,
                    'trial_success': success_failure[i],
                    'trajectory_angle': trajectory_angle,
                    'trajectory_distance': trajectory_distance,
                    'rotated_velocity': rotated_velocity.astype(np.float64)
                })

            # 批量写入h5文件
            if trials_data:
                ctrl.append_trials_batch(daily_name, trials_data)
                print(f"    成功写入 {len(trials_data)} 个trial到h5文件")

        # 显示该天的统计信息
        try:
            stats = ctrl.get_day_stats(daily_name)
            print(f"  {daily_name} 统计信息: {stats}")
        except Exception as e:
            print(f"  获取统计信息时出错: {e}")

    # 显示整个文件的统计信息
    try:
        file_stats = ctrl.get_file_stats()
        print(f"\n文件统计信息: {file_stats}")
    except Exception as e:
        print(f"获取文件统计信息时出错: {e}")

    # 关闭文件
    ctrl.close()
    print("数据处理完成！")
