import h5py
import os
import numpy as np
import torch
import pickle
from collections import Counter
from scipy import signal
from torch.nn.utils.rnn import pad_sequence



def slide_window(data: list, label: list, windows_size: int = 500,
                 step: int = 100, start_from: int = 0):
    temp_slices = []
    labels_slices = []
    for i, temp in enumerate(data):
        for j in range(start_from, temp.shape[-1] - windows_size, step):
            temp_slices.append(temp[..., j:j + windows_size])
            labels_slices.append(label[i])
    return np.array(temp_slices), np.array(labels_slices)


def filter_pipline(data, sfreq=600, l_freq=4, h_freq=200, order=4):
    sos_bp = signal.butter(order, [l_freq, h_freq], 'bandpass', output='sos', fs=sfreq)
    sos_bs_100 = signal.butter(order, [99, 101], 'bandstop', output='sos', fs=sfreq)
    sos_bs_200 = signal.butter(order, [199, 201], 'bandstop', output='sos', fs=sfreq)
    data = signal.sosfiltfilt(sos_bp, data)
    data = signal.sosfiltfilt(sos_bs_100, data)
    data = signal.sosfiltfilt(sos_bs_200, data)
    return data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label, name, decorrelate=False):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long()
        self.name = name
        self.decorrelate = decorrelate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        # data = data[:, -1]
        if self.decorrelate:
            if data.ndim == 2:
                data = data - data.mean(1, keepdim=True)
                U, S, Vt = torch.linalg.svd(data, full_matrices=False)
                data = U.T @ data
            else:
                raise ValueError('data shape is not correct')                
        else:
            if data.ndim == 2:
                data = data - data.mean(0, keepdim=True)
            elif data.ndim == 3:
                data = data - data.mean(1, keepdim=True)
        return data, self.label[idx]

    def __str__(self):
        return self.name


class ContinuesDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, pos, mask, name):
        self.data = data
        self.label = label
        self.pos = pos
        self.mask = mask
        self.name = name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        # data = data[:, -1]
        if data.ndim == 2:
            data = data - data.mean(0, keepdim=True)
        elif data.ndim == 3:
            data = data - data.mean(1, keepdim=True)
        return data, self.label[idx], self.pos[idx], self.mask[idx]

    def __str__(self):
        return self.name


class StreamDataset(torch.utils.data.Dataset):
    def __init__(self, data, batch=128, seq_len=768, max_length=None):
        self.data = data
        self.batch = batch
        self.seq_len = seq_len
        self.length = max_length

    def __len__(self):
        return self.batch

    def __getitem__(self, idx):
        data = self.data[idx]
        length = self.length[idx] - self.seq_len
        data = data - data.mean(0, keepdims=True)
        i = torch.randint(0, length, (1,)).item()
        data = data[..., i:i + self.seq_len]
        return data


class StreamDatasetwithSession(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=768, session=None, max_length=None):
        self.data = data
        self.seq_len = seq_len
        self.length = max_length
        self.session = session
        self.split = seq_len // 256
        self.trial_length = len(self.data)

    def __len__(self):
        return self.trial_length * 5

    def __getitem__(self, idx):
        idx = idx % self.trial_length
        data = self.data[idx]
        length = self.length - self.seq_len
        i = torch.randint(0, length, (1,)).item()
        data = data[..., i:i + self.seq_len]
        data = data - data.mean(0, keepdims=True)
        data = np.array(np.split(data, self.split, axis=-1)).transpose(1, 0, 2)
        return data, self.session[idx]


class StreamDataset2(torch.utils.data.Dataset):
    def __init__(self, data, labels, batch=128, seq_len=20, max_length=None, name=None, session=None):
        self.data = data
        self.labels = labels
        self.batch = batch
        self.seq_len = seq_len
        self.length = max_length
        self.name = name
        self.session = session

    def __len__(self):
        return len(self.data) * 4

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        data = self.data[idx]
        labels = self.labels[idx]
        length = self.length[idx] - self.seq_len
        session = self.session[idx]
        if length < 0:
            length = 1
        data = data - data.mean(0, keepdims=True)
        i = torch.randint(0, length, (1,)).item()
        data = data[..., i:i + self.seq_len]
        # data = data[:, -1]
        return data, labels, session

    def __str__(self):
        return self.name


def read_pkl(root, session=0, sfreq=256, include_failure=True):
    data_path = [os.path.join(root, file_name) for file_name in os.listdir(root)]
    data_path.sort()
    data_paths = []
    for file_name in data_path:
        try:
            a = pickle.load(open(file_name, 'rb'))
            data_paths.append(a)
        except:
            print(f'Error loading {file_name}')
            continue

    data = []
    labels = []
    data = [d['data'] for d in data_paths]
    labels = [d['label'] for d in data_paths]
    
    filtered_data = []
    filtered_labels = []
    sessions = []
    for d, l in zip(data, labels):
        temp_d = []
        temp_l = []
        s = []
        flag = False

        for i in range(len(d)):
            if l[i] == 4:
                continue
            flag = True
            temp = filter_pipline(d[i], sfreq=600, l_freq=4, h_freq=int(sfreq / 2.56))
            # temp = temp[:, :-600]
            temp = signal.resample_poly(temp, sfreq, 600, axis=-1)  # [..., sfreq:-sfreq]
            temp_d.append(temp)
            temp_l.append(l[i])
            s.append(session)

        if flag:
            sessions.append(s)
            filtered_data += temp_d
            filtered_labels += temp_l
    return filtered_data, filtered_labels, sessions


def read_continuous_pkl(root, windows_size=1, step=0.1, sfreq=256, radius=12, dis_threshold=1.5):
    data_path = [os.path.join(root, file_name) for file_name in os.listdir(root)]
    data_path.sort()
    data_paths = []
    for file_name in data_path:
        try:
            a = pickle.load(open(file_name, 'rb'))
            data_paths.append(a)
        except:
            print(f'Error loading {file_name}')
            continue
    data = []
    labels = []
    positions = []
    lengths = []
    distances = []
    for data_path in data_paths:
        if data_path['pos'] is not None:
            d = data_path['data']
            label = data_path['label']
            pos = list(data_path['pos'])
            rest = len(Counter(label)) - 1
            for i, l in enumerate(label):
                if l != rest:
                    temp_pos = np.array(pos[i])
                    length = temp_pos.shape[0]
                    temp_pos = temp_pos[:length]

                    diffs = np.diff(temp_pos, axis=0)
                    distance = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))) / radius

                    if distance > dis_threshold:
                        continue

                    positions.append(torch.from_numpy(temp_pos / radius).float())
                    dd = filter_pipline(d[i], sfreq=sfreq, l_freq=4, h_freq=100)
                    dd, ll = slide_window([dd], [l], windows_size=int(windows_size * sfreq), step=int(step * sfreq))
                    dd = signal.resample_poly(dd, 256, sfreq, axis=-1)
                    dd = dd[:length]
                    data.append(torch.from_numpy(dd).float())
                    labels.append(torch.from_numpy(ll[:length]).long())
                    lengths.append(length)
                    distances.append(distance)
    return data, labels, positions, lengths, distances


class GeneratorDataset(object):
    def __init__(self, root, mode='drop_old', sfreq=256, session=0):
        filtered_data, filtered_labels, sessions = read_pkl(root, session=session, sfreq=sfreq)
        self.data = filtered_data
        self.labels = filtered_labels
        self.sessions = sessions
        self.length = len(self.data)
        self.mode = mode
        self.count = 0
        self.name = os.path.split(root)[-1]
        self.sfreq = 256

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.count == 0:
            test = self.data[self.count:self.count + 1]
            test_labels = np.concatenate(self.labels[self.count:self.count + 1])
            test = [i for d in test for i in d]
            test, test_labels = slide_window(test, list(test_labels), windows_size=self.sfreq, step=32)
            test_dataset = Dataset(test, test_labels, name=f'{self.name}_{self.count + 1}')
            self.count += 1
            return None, test_dataset
        elif self.count < self.length:
            if self.mode != 'drop_old':
                train = self.data[self.count - 1:self.count]
                train_labels = np.concatenate(self.labels[self.count - 1:self.count])
                train_session = np.concatenate(self.sessions[self.count - 1:self.count])
            else:
                train = self.data[:self.count]
                train_labels = np.concatenate(self.labels[:self.count])
                train_session = np.concatenate(self.sessions[:self.count])
            test = self.data[self.count:self.count + 1]
            test_labels = np.concatenate(self.labels[self.count:self.count + 1])
            train = [i for d in train for i in d]
            test = [i for d in test for i in d]
            train_length = [d.shape[-1] - self.sfreq for d in train]
            test, test_labels = slide_window(test, list(test_labels), windows_size=self.sfreq, step=32)
            train_dataset = StreamDataset2(train, train_labels, batch=0, seq_len=self.sfreq, max_length=train_length,
                                           name=f'{self.name}_{self.count}', session=train_session)
            test_dataset = Dataset(test, test_labels, name=f'{self.name}_{self.count + 1}')
            self.count += 1
            return train_dataset, test_dataset
        else:
            if self.mode != 'drop_old':
                train = self.data[self.count - 1:self.count]
                train_labels = np.concatenate(self.labels[self.count - 1:self.count])
                train_session = np.concatenate(self.sessions[self.count - 1:self.count])
            else:
                train = self.data[:self.count]
                train_labels = np.concatenate(self.labels[:self.count])
                train_session = np.concatenate(self.sessions[:self.count])
            train = [i for d in train for i in d]
            train_length = [d.shape[-1] for d in train]
            train_dataset = StreamDataset2(train, train_labels, batch=0, seq_len=256, max_length=train_length,
                                           name=f'{self.name}_{self.count}', session=train_session)
            return train_dataset, None


def load_stream(t_len=3, windows_size=256):
    import h5py
    data = h5py.File('/home/xzd_lab/wangruopeng/DATA/s01.h5', 'r')
    stream = data['data']
    session = data['session']
    train_stream = StreamDatasetwithSession(stream, session=session, seq_len=windows_size * t_len,
                                            max_length=10 * windows_size)
    return train_stream


def load_daily_dataset(test_path, windows_size=256, step=32, session=16, include_failure=True, decorrelate=False, svd=None):
    """
    Load daily dataset from pkl files and apply sliding window.
    Parameters
    ----------
    step
    test_path
    windows_size
    session

    Returns
    -------

    """
    test, test_labels, test_session = read_pkl(test_path, sfreq=windows_size, session=session, include_failure=include_failure)
    if decorrelate:
        print('Conduct channel decorrelation')
        test = decorrelate_channel_by_day(test)
    elif svd:
        print(f"Conduct channel SVD keeping top {svd['rank']} components")
        for i in range(len(test)):
            test[i] -= np.mean(test[i], axis=1, keepdims=True)
            test[i] = svd['U'][:, :svd['rank']].T @ test[i]
    test, test_labels = slide_window(test, list(test_labels), windows_size=windows_size, step=step)
    test_path = os.path.split(test_path)[-1]
    test_dataset = Dataset(test, test_labels, name=f'{test_path}')
    return test_dataset


def load_trail(test_path, session=16):
    """
    Load daily dataset from pkl files and apply sliding window.
    Parameters
    ----------
    test_path
    windows_size
    session

    Returns
    -------

    """
    test, test_labels, test_session = read_pkl(test_path, session=session)
    test_path = os.path.split(test_path)[-1]
    return test, test_labels, test_path

def decorrelate_channel(trails):
    decorrelated_trails = []
    for trail in trails:
        trail = trail - np.mean(trail, axis=1, keepdims=True)
        U, S,Vt = np.linalg.svd(trail, full_matrices=False)
        trail = U.T @ trail
        decorrelated_trails.append(trail)
    return decorrelated_trails

def decorrelate_channel_by_day(trials):
    n_trials = len(trials)
    n_time_list = [trial.shape[1] for trial in trials]
    trials = np.hstack(trials)
    # n_trials, n_channels, n_time = decorrelated_trials.shape
    trials -= np.mean(trials, axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(trials, full_matrices=False)
    trials = U.T @ trials
    n = 0
    decorrelated_trials = []
    for i in range(n_trials):
        decorrelated_trials.append(trials[:, n:n+n_time_list[i]])
        n += n_time_list[i]
    return decorrelated_trials



def load_train_dataset(train_path, windows_size=256, step=64, session=16, include_failure=True, decorrelate=False, svd=None):
    """
    Load daily dataset from pkl files and apply sliding window.
    Parameters
    ----------
    step
    train_path
    windows_size
    session

    Returns
    -------

    """
    train_trail = []
    train_labels_trail = []
    train_session = []
    for path in train_path:
        t, t_l, t_s = read_pkl(path, sfreq=windows_size, session=session, include_failure=include_failure)
        train_trail += t
        train_labels_trail += t_l
        train_session += t_s

    if decorrelate:
        print('Conduct channel decorrelation')
        train_trail = decorrelate_channel_by_day(train_trail)
    elif svd:
        print(f"Conduct channel SVD keeping top {svd['rank']} components")
        for i in range(len(train_trail)):
            train_trail[i] -= np.mean(train_trail[i], axis=1, keepdims=True)
            train_trail[i] = svd['U'][:, :svd['rank']].T @ train_trail[i]

    idx = np.arange(len(train_trail))
    np.random.shuffle(idx)
    train = [train_trail[i] for i in idx]
    train_labels = [train_labels_trail[i] for i in idx]
    # train_session = [train_session[i] for i in idx]

    valid_size = int(len(train) * 0.9)
    train_valid = train[:valid_size]
    train_valid_labels = train_labels[:valid_size]
    valid = train[valid_size:]
    valid_labels = train_labels[valid_size:]

    train, train_labels = slide_window(train, list(train_labels), windows_size=windows_size, step=step)
    train_valid, train_valid_labels = slide_window(train_valid, list(train_valid_labels), windows_size=windows_size,
                                                   step=step)
    valid, valid_labels = slide_window(valid, list(valid_labels), windows_size=windows_size, step=step)

    train_path = '_'.join([os.path.split(i)[-1] for i in train_path])
    train_dataset = Dataset(train, train_labels, name=f'{train_path}')
    valid_dataset = Dataset(valid, valid_labels, name=f'{train_path}')
    train_valid_dataset = Dataset(train_valid, train_valid_labels, name=f'{train_path}')
    return train_dataset, train_valid_dataset, valid_dataset, train_trail, train_labels_trail


def load_continuous_daily_dataset(train_path, windows_size=1, step=0.1, sfreq=600):
    """
    Load daily dataset from pkl files and apply sliding window.
    Parameters
    ----------
    train_path

    Returns
    -------

    """
    train_trail = []
    train_labels_trail = []
    train_pos = []
    train_seq_len = []
    train_distance = []
    t, t_l, t_s, t_len, t_dis = read_continuous_pkl(train_path, windows_size=windows_size, step=step, sfreq=sfreq)
    train_trail += t
    train_labels_trail += t_l
    train_pos += t_s
    train_seq_len += t_len
    train_distance += t_dis

    train_distance = np.array(train_distance)
    train_trail = pad_sequence(train_trail, batch_first=True, padding_value=0.0)
    length = train_trail.shape[1]
    train_labels_trail = pad_sequence(train_labels_trail, batch_first=True, padding_value=-1)
    train_pos = pad_sequence(train_pos, batch_first=True, padding_value=0.0)[:, :length]
    train_mask = torch.ones_like(train_labels_trail, dtype=torch.bool)
    train_mask = train_mask & (train_labels_trail != -1)  # Mask out -1 labels

    train_path = '_'.join([os.path.split(i)[-1] for i in train_path])
    train_dataset = ContinuesDataset(train_trail, train_labels_trail, train_pos, train_mask, name=f'{train_path}')
    return train_dataset


def load_continuous_train_dataset(train_path, windows_size=1, step=0.1, sfreq=600):
    """
    Load daily dataset from pkl files and apply sliding window.
    Parameters
    ----------
    step
    train_path
    windows_size
    session

    Returns
    -------

    """
    train_trail = []
    train_labels_trail = []
    train_pos = []
    train_seq_len = []
    train_distance = []
    for path in train_path:
        t, t_l, t_s, t_len, t_dis = read_continuous_pkl(path, windows_size=windows_size, step=step, sfreq=sfreq)
        train_trail += t
        train_labels_trail += t_l
        train_pos += t_s
        train_seq_len += t_len
        train_distance += t_dis

    train_distance = np.array(train_distance)
    train_trail = pad_sequence(train_trail, batch_first=True, padding_value=0.0)
    length = train_trail.shape[1]
    train_labels_trail = pad_sequence(train_labels_trail, batch_first=True, padding_value=-1)
    train_pos = pad_sequence(train_pos, batch_first=True, padding_value=0.0)[:, :length]
    train_mask = torch.ones_like(train_labels_trail, dtype=torch.bool)
    train_mask = train_mask & (train_labels_trail != -1)  # Mask out -1 labels

    train_path = '_'.join([os.path.split(i)[-1] for i in train_path])
    train_dataset = ContinuesDataset(train_trail, train_labels_trail, train_pos, train_mask, name=f'{train_path}')
    return train_dataset





