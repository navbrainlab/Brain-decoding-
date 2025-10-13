import h5py
import os
import numpy as np
import torch
import pickle
from collections import Counter
from scipy import signal
from torch.nn.utils.rnn import pad_sequence


def slide_window(data: list, label: list, windows_size: int = 500, step: int = 100):
    temp_slices = []
    labels_slices = []
    for i, temp in enumerate(data):
        for j in range(0, temp.shape[-1] - windows_size, step):
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
    def __init__(self, data, label, name):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long()
        self.name = name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        # data = data[:, -1]
        # remove subtract mean
        if data.ndim == 2:
            data = data - data.mean(0, keepdim=True)# xzd
            # data = data - data.mean(1, keepdim=True)
            # min_vals = data.min(axis=0, keepdims=True).values  # 每个时间点的最小值 shape [1, t]
            # max_vals = data.max(axis=0, keepdims=True).values  # 每个时间点的最大值 shape [1, t]
            # data = (data - min_vals) / (max_vals - min_vals + 1e-8)
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


def read_pkl(root, session=0, sfreq=256):
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

    data = [d['data'] for d in data_paths]
    labels = [d['label'] for d in data_paths]
    filtered_data = []
    filtered_labels = []
    sessions = []
    for d, l in zip(data, labels):
        temp_d = []
        temp_l = []
        s = []
        for i in range(len(d)):
            temp = filter_pipline(d[i], sfreq=600, l_freq=4, h_freq=int(sfreq / 2.56))
            # temp = temp[:, :-600]
            temp = signal.resample_poly(temp, sfreq, 600, axis=-1)  # [..., sfreq:-sfreq]
            temp_d.append(temp[:,:])
            temp_l.append(l[i])
            s.append(session)
        sessions.append(s)

        filtered_data += temp_d
        filtered_labels += temp_l
    return filtered_data, filtered_labels, sessions


def read_pkl_wo_resample(root,):# modify
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

    data = [d['data'] for d in data_paths]
    labels = [d['label'] for d in data_paths]
    filtered_data = []
    filtered_labels = []
    sessions = []
    for d, l in zip(data, labels):
        temp_d = []
        temp_l = []
        s = []
        for i in range(len(d)):
            temp = filter_pipline(d[i], sfreq=600, l_freq=4, h_freq=100)
            temp_d.append(temp)
            temp_l.append(l[i])
            s.append([])
        sessions.append(s)

        filtered_data += temp_d
        filtered_labels += temp_l
    return filtered_data, filtered_labels, sessions




    #             dd = filter_pipline(d[i], sfreq=sfreq, l_freq=4, h_freq=100)
    #             dd, ll = slide_window([dd], [l], windows_size=int(windows_size * sfreq), step=int(step * sfreq))
    #             dd = signal.resample_poly(dd, 256, sfreq, axis=-1)
    #             # dd = dd[:length]
    #             # data.append(torch.from_numpy(dd).float())
    #             # labels.append(torch.from_numpy(ll[:length]).long())
    #             # lengths.append(length)
    #             # distances.append(distance)
    # return data, labels, positions, lengths, distances


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


def load_daily_dataset(test_path, windows_size=256, step=32, session=16):
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
    test, test_labels, test_session = read_pkl(test_path, sfreq=windows_size, session=session)
    idx = [i for i,j in enumerate(test_labels) if j<4]
    test = [test[i] for i in idx]
    test_labels = [test_labels[i] for i in idx]
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


def load_train_dataset(train_path, windows_size=256, step=32, session=16):
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
        t, t_l, t_s = read_pkl(path, sfreq=windows_size, session=session)
        train_trail += t
        train_labels_trail += t_l
        train_session += t_s

    idx = [i for i,j in enumerate(train_labels_trail) if j<4]
    # idx = np.arange(len(train_trail))
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


def load_daily_dataset_post_resample(test_path, windows_size=600, step=60):
    """
    Load daily dataset from pkl files and apply sliding window.
    Parameters
    ----------
    train_path

    Returns
    -------

    """
    test, test_labels, test_session = read_pkl_wo_resample(test_path)
    test, test_labels = slide_window(test, list(test_labels), windows_size=windows_size, step=step)
    # idx = np.arange(len(test))
    # np.random.shuffle(idx)
    # test = test[idx][:1000]
    # test_labels = test_labels[idx][:1000]
    test = signal.resample_poly(test, 256, 600, axis=-1)
    test_path = os.path.split(test_path)[-1]
    test_dataset = Dataset(test, test_labels, name=f'{test_path}')
    return test_dataset

def load_train_dataset_post_resample(train_path, windows_size=600, step=60, session=16): # modify
    train_trail = []
    train_labels_trail = []
    train_session = []
    for path in train_path:
        t, t_l, t_s = read_pkl_wo_resample(path)
        train_trail += t
        train_labels_trail += t_l
        train_session += t_s

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
    train = signal.resample_poly(train, 256, 600, axis=-1)
    train_valid, train_valid_labels = slide_window(train_valid, list(train_valid_labels), windows_size=windows_size,
                                                   step=step)
    train_valid = signal.resample_poly(train_valid, 256, 600, axis=-1)
    valid, valid_labels = slide_window(valid, list(valid_labels), windows_size=windows_size, step=step)
    valid = signal.resample_poly(valid, 256, 600, axis=-1)
    train_path = '_'.join([os.path.split(i)[-1] for i in train_path])
    train_dataset = Dataset(train, train_labels, name=f'{train_path}')
    valid_dataset = Dataset(valid, valid_labels, name=f'{train_path}')
    train_valid_dataset = Dataset(train_valid, train_valid_labels, name=f'{train_path}')
    return train_dataset, train_valid_dataset, valid_dataset, train_trail, train_labels_trail





