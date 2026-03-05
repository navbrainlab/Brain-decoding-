from scipy import signal
import pickle
import os
import numpy as np
import torch
import scipy.io
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
    scale,
)
from h5_dataset import *

class XZD_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label, name):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long()
        self.name = name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = data - data.mean(0, keepdim=True)
        return data, self.label[idx]

    def __str__(self):
        return self.name

class Stanford_imagery_basic_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = data - data.mean(0, keepdim=True)
        return data, self.label[idx]

def slide_window(data: list, label: list, windows_size: int = 500,
                 step: int = 100, date: list=None):
    temp_slices = []
    labels = []
    dates = []
    for i, temp in enumerate(data):
        for j in range(0, temp.shape[-1] - windows_size + 1, step):
            temp_slices.append(temp[..., j:j + windows_size])
            labels.append(label[i])
            if date is not None:
                dates.append(date[i])
    if date is not None:
        return np.array(temp_slices), np.array(labels), np.array(dates)
    else:
        return np.array(temp_slices), np.array(labels)

def filter_pipline(data, sfreq=600, l_freq=4, h_freq=200, order=4):
    sos_bp = signal.butter(order, [l_freq, h_freq], 'bandpass', output='sos', fs=sfreq)
    sos_bs_100 = signal.butter(order, [99, 101], 'bandstop', output='sos', fs=sfreq)
    sos_bs_200 = signal.butter(order, [199, 201], 'bandstop', output='sos', fs=sfreq)
    data = signal.sosfiltfilt(sos_bp, data)
    data = signal.sosfiltfilt(sos_bs_100, data)
    data = signal.sosfiltfilt(sos_bs_200, data)
    return data

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


def load_daily_dataset(data_path, windows_size=256, step=32, train_ratio=0.8, shuffle='sample'):
    """
    Load daily dataset from pkl files, apply sliding window
    and split train set and test set randomly.

    Parameters
    ----------
    step
    data_path
    windows_size
    session

    Returns
    -------

    """
    
    data, labels, test_session = read_pkl(data_path, sfreq=windows_size)

    if shuffle == 'trial':
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        data = [data[i] for i in indices]
        labels = [labels[i] for i in indices]    

    data, labels = slide_window(data, list(labels), windows_size=windows_size, step=step)

    # Shuffle the data
    if shuffle == 'sample':
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

    if shuffle == 'none':
        pass

    if shuffle != 'sample' and shuffle != 'trial'  and shuffle != 'none':
        raise ValueError(f'Invalid shuffle type: {shuffle}')

    # Split the data
    train_ratio = train_ratio
    split_index = int(data.shape[0] * train_ratio)    
    train_data = data[:split_index]
    train_labels = labels[:split_index]
    test_data = data[split_index:]
    test_labels = labels[split_index:]

    data_path = os.path.split(data_path)[-1]
    train_dataset = XZD_Dataset(train_data, train_labels, name=f'{data_path}_train')
    test_dataset = XZD_Dataset(test_data, test_labels, name=f'{data_path}_test')
    return train_dataset, test_dataset


def load_multiday_dataset(train_path, windows_size=256, step=64, train_ratio=0.8, shuffle='sample'):
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
    trial = []
    trial_labels = []
    for path in train_path:
        t, t_l, t_s = read_pkl(path, sfreq=windows_size)
        # Repeat path to form a list with same length as t
        trial += t
        trial_labels += t_l

    if shuffle == 'trial':
        idx = np.arange(len(trial))
        np.random.shuffle(idx)
        trial = [trial[i] for i in idx]
        trial_labels = [trial_labels[i] for i in idx]

        train_size = int(len(trial) * train_ratio)
        train = trial[:train_size]
        train_labels = trial_labels[:train_size]
        valid = trial[train_size:]
        valid_labels = trial_labels[train_size:]

        train, train_labels = slide_window(train, list(train_labels), windows_size=windows_size, step=step)
        valid, valid_labels = slide_window(valid, list(valid_labels), windows_size=windows_size, step=step)

    trial, trial_labels = slide_window(trial, list(trial_labels), windows_size=windows_size, step=step)

    if shuffle == 'sample':
        indices = np.arange(len(trial))
        np.random.shuffle(indices)
        trial = trial[indices]
        trial_labels = trial_labels[indices]    

        train_size = int(len(trial) * train_ratio)
        train = trial[:train_size]
        train_labels = trial_labels[:train_size]
        valid = trial[train_size:]
        valid_labels = trial_labels[train_size:]

    if shuffle == 'none':
        pass

    if shuffle != 'sample' and shuffle != 'trial' and shuffle != 'none':
        raise ValueError(f'Invalid shuffle type: {shuffle}')
    

    train_path = '_'.join([os.path.split(i)[-1] for i in train_path])
    all_dataset = XZD_Dataset(trial, trial_labels, name=f'{train_path}')
    valid_dataset = XZD_Dataset(valid, valid_labels, name=f'{train_path}')
    train_dataset = XZD_Dataset(train, train_labels, name=f'{train_path}')
    return train_dataset, valid_dataset


def load_Stanford_imagery_basic(data_path, windows_size=1000, step=100, train_ratio=0.8, shuffle='sample'):
    data = scipy.io.loadmat(data_path)
    # Split data['stim'] into groups of consecutive same numbers
    stim = data['stim'].flatten()
    signal = data['data']
    stim_trial = []
    signal_trial = []
    start = 0

    for i in range(1, len(stim)):
        if (stim[i] != stim[i-1]):
            if stim[i-1] != 0:
                stim_trial.append(stim[start])
                signal_trial.append(signal[start:i])
            start = i
    if stim[-1] != 0:
        stim_trial.append(stim[start])  # The last group
        signal_trial.append(signal[start:])  # The last group

    stim_trial = np.array(stim_trial) # [trial]
    signal_trial = np.array(signal_trial) # [trial, time, channel]  

    signal_sample, stim_sample = slide_window(signal_trial.transpose(0, 2, 1), stim_trial, windows_size=windows_size, step=step)
    # signal_sample: [sample, channel, time]
    # stim_sample: [sample]

    # Shuffle the data
    if shuffle == 'sample':
        indices = np.arange(signal_sample.shape[0])
        np.random.shuffle(indices)
        signal_sample = signal_sample[indices]
        stim_sample = stim_sample[indices]
    else:
        raise ValueError('Not supported shuffle type')
    
    # Split the data
    train_ratio = train_ratio
    split_index = int(signal_sample.shape[0] * train_ratio)    
    train_data = signal_sample[:split_index]
    train_labels = stim_sample[:split_index]
    test_data = signal_sample[split_index:]
    test_labels = stim_sample[split_index:]

    data_path = os.path.split(data_path)[-1]
    train_dataset = Stanford_imagery_basic_Dataset(train_data, train_labels)
    test_dataset = Stanford_imagery_basic_Dataset(test_data, test_labels)
    return train_dataset, test_dataset  


def load_Stanford_imagery_feedback(data_paths, windows_size=1000, step=100, train_ratio=0.8, shuffle='sample'):
    '''
    Can only classify rest or not
    '''
    # Load imagery_feedback data
    stim_sample = []
    signal_sample = []    
    for data_path in data_paths:
        data = scipy.io.loadmat(data_path)
        # Split data['stim'] into groups of consecutive same numbers
        if 'TargetCode' in data.keys(): # feedback
            iti = data['ITI'].flatten()
            result = data['Result'].flatten()
            stim = data['TargetCode'].flatten()
            signal = data['data']
            stim = stim[(iti==0)*(result==0)] # Remove inter trial interval and reward period
            signal = signal[(iti==0)*(result==0), :]
            fb = True
        elif 'StimulusCode' in data.keys(): # imagery or motor
            stim = data['StimulusCode'].flatten()
            signal = data['data']
            fb = False
        else:
            raise ValueError('Unrecognized stimulation')

        j = 0
        while j<(len(stim) - windows_size):
            indices = np.where(stim[j:j+windows_size] != stim[j])
            if len(indices[0]) > 0: # stim[j:j+windows_size] includes different values
                j += indices[0][0]
            else:
                if fb and (stim[j]==0): # No target
                    j += windows_size
                else:
                    signal_sample.append(signal[j:j + windows_size, :])
                    if fb and (stim[j]==2): # Lower target with smaller cursor location number, Passive, TargetCode is 2
                        stim_sample.append(0)
                    elif (not fb) and (stim[j]!=0): # Every condition that is not rest
                        stim_sample.append(1)
                    else:
                        stim_sample.append(stim[j])
                    j += step            

    stim_sample = np.array(stim_sample) # [sample]
    signal_sample = np.array(signal_sample).transpose(0, 2, 1) # [sample, channel, time]
        # if fb:
        #     stim_sample[stim_sample==2] = 0 # Lower target with smaller cursor location number, Passive, TargetCode is 2
        # else:
        #     stim_sample[stim_sample!=0] = 1


    # Shuffle the data
    if shuffle == 'sample':
        indices = np.arange(signal_sample.shape[0])
        np.random.shuffle(indices)
        signal_sample = signal_sample[indices]
        stim_sample = stim_sample[indices]
    else:
        raise ValueError('Not supported shuffle type')

    # Split the data
    train_ratio = train_ratio
    split_index = int(signal_sample.shape[0] * train_ratio)    
    train_data = signal_sample[:split_index]
    train_labels = stim_sample[:split_index]
    test_data = signal_sample[split_index:]
    test_labels = stim_sample[split_index:]

    data_path = os.path.split(data_path)[-1]
    train_dataset = Stanford_imagery_basic_Dataset(train_data, train_labels)
    test_dataset = Stanford_imagery_basic_Dataset(test_data, test_labels)
    return train_dataset, test_dataset  


def load_hypergraph_data(train_path, windows_size=256, step=256, train_ratio=0.8, return_date=False):
    """
    Load daily dataset from pkl files, apply sliding window and compute covariance matrices.
    Parameters
    ----------
    step
    train_path
    windows_size
    session

    Returns
    -------

    """
    trial = []
    trial_labels = []
    date = []
    # Repeat path to form a list
    for path in train_path:
        t, t_l, t_s = read_pkl(path, sfreq=windows_size)
        date += [os.path.split(path)[-1]] * len(t)
        trial += t
        trial_labels += t_l

    trial, trial_labels, date = slide_window(trial, list(trial_labels), windows_size=windows_size, step=step, date=date)
    # trial: [sample, channel, time], numpy array; trial_labels: [sample], numpy array

    # compute covariance matrix for each sample
    cov_matrices = []
    for sample in trial:
        cov = np.cov(sample)
        cov = cov.flatten() # reshape to 1D vector
        cov_matrices.append(cov)
    trial_cov = np.array(cov_matrices) # [sample, channel*channel]

    # one-hot encode the labels
    num_classes = len(np.unique(trial_labels))
    trial_labels_onehot = np.zeros((len(trial_labels), num_classes))
    for i, label in enumerate(trial_labels):
        trial_labels_onehot[i, label] = 1
    trial_labels = trial_labels_onehot
    
    # shuffle the data
    indices = np.arange(len(trial_cov))
    np.random.shuffle(indices)
    trial_cov = trial_cov[indices]
    trial_labels = trial_labels[indices]
    date = np.array(date)[indices]
    # split the data
    train_size = int(len(trial_cov) * train_ratio)
    train = trial_cov[:train_size]
    train_labels = trial_labels[:train_size]
    train_date = date[:train_size]
    valid = trial_cov[train_size:]
    valid_labels = trial_labels[train_size:]
    valid_date = date[train_size:]
    if return_date:
        return train, train_labels, valid, valid_labels, train_date, valid_date, indices
    else:
        return train, train_labels, valid, valid_labels, indices

def load_centered_data_for_otta(train_path, windows_size=256, step=256, train_ratio=0.8, indices=None, return_date=False):
    """
    Load daily dataset from pkl files and apply sliding window. Do NOT compute covariance matrices.
    Parameters
    ----------
    step
    train_path
    windows_size
    session

    Returns
    -------

    """
    trial = []
    trial_labels = []
    date = []
    # Repeat path to form a list
    for path in train_path:
        t, t_l, t_s = read_pkl(path, sfreq=windows_size)
        date += [os.path.split(path)[-1]] * len(t)
        trial += t
        trial_labels += t_l

    trial, trial_labels, date = slide_window(trial, list(trial_labels), windows_size=windows_size, step=step, date=date)
    # trial: [sample, channel, time], numpy array; trial_labels: [sample], numpy array

    # Channel_wise centering
    trial = trial - trial.mean(axis=2, keepdims=True)

    # one-hot encode the labels
    num_classes = len(np.unique(trial_labels))
    trial_labels_onehot = np.zeros((len(trial_labels), num_classes))
    for i, label in enumerate(trial_labels):
        trial_labels_onehot[i, label] = 1
    trial_labels = trial_labels_onehot
    
    # shuffle the data
    if indices is None:
        indices = np.arange(len(trial))
        np.random.shuffle(indices)
    trial = trial[indices]
    trial_labels = trial_labels[indices]
    date = np.array(date)[indices]
    # split the data
    train_size = int(len(trial) * train_ratio)
    train = trial[:train_size]
    train_labels = trial_labels[:train_size]
    train_date = date[:train_size]
    valid = trial[train_size:]
    valid_labels = trial_labels[train_size:]
    valid_date = date[train_size:]
    if return_date:
        return train, train_labels, valid, valid_labels, train_date, valid_date, indices
    else:
        return train, train_labels, valid, valid_labels, indices

def load_splitted_target_dataset(path):
    dataset = np.load(path, allow_pickle=True).item()
    data = dataset['target_data']
    labels = dataset['target_labels']
    return data, labels
    
def load_splitted_source_dataset(path):
    dataset = np.load(path, allow_pickle=True).item()
    train_data = dataset.get('train_data')
    train_labels = dataset.get('train_labels')
    valid_data = dataset.get('valid_data')
    valid_labels = dataset.get('valid_labels')
    train_date = dataset.get('train_date', None)
    valid_date = dataset.get('valid_date', None)
    return train_data, train_labels, valid_data, valid_labels, train_date, valid_date

def _load_bcic(subject_ids: list[int], dataset: str = "2a",
              preprocessing_dict: dict = None, verbose: str = "WARNING"):
    dataset_name = "BNCI2014001" if dataset == "2a" else "BNCI2014004"
    dataset = MOABBDataset(dataset_name, subject_ids=subject_ids)

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose=verbose),
        Preprocessor(scale, factor=1e6, apply_on_array=True),
        Preprocessor("resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose)
    ]

    l_freq, h_freq = preprocessing_dict["low_cut"], preprocessing_dict["high_cut"]
    if l_freq is not None or h_freq is not None:
        preprocessors.append(Preprocessor("filter", l_freq=l_freq, h_freq=h_freq,
                                          verbose=verbose))

    preprocess(dataset, preprocessors)

    # create windows
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    trial_start_offset_samples = int(preprocessing_dict["start"] * sfreq)
    trial_stop_offset_samples = int(preprocessing_dict["stop"] * sfreq)
    windows_dataset = create_windows_from_events(
        dataset, trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples, preload=True
    )

    return windows_dataset, dataset.datasets[0].raw.info

def load_centered_bcic(subject_id: int, 
                       dataset: str = "2a", 
                       session: str = "T",
                       train_ratio: float = 0.8, 
                       windows_size: int = 1000, 
                       step: int = 1000,
                       preprocessing_dict: dict = None, 
                       return_date: bool = False,
                       indices = None,
                       verbose: str = "WARNING"):
    if preprocessing_dict is None:
        preprocessing_dict = {"sfreq":250,"low_cut":0,"high_cut":40,"start":0.0,"stop":0.0}
    dataset, info = _load_bcic(subject_ids=[subject_id], dataset=dataset,
                                preprocessing_dict=preprocessing_dict, verbose=verbose)    
    # split the data
    splitted_ds = dataset.split("session")
    dataset = splitted_ds[f"session_{session}"]

    # load the data
    X = np.concatenate([run.windows.load_data()._data for run in dataset.datasets], axis=0)
    y = np.concatenate([run.y for run in dataset.datasets], axis=0)

    # Slide window
    X_windows, y_windows = slide_window(X, y, windows_size=windows_size, step=step)

    # Channel-wise centering
    X_windows = X_windows - X_windows.mean(axis=2, keepdims=True)

    # One-hot encode the labels
    y_windows = np.eye(np.max(y_windows) + 1)[y_windows]

    # shuffle the data
    if indices is None:
        indices = np.arange(len(X_windows))
        np.random.shuffle(indices)
    X_windows = X_windows[indices]
    y_windows = y_windows[indices]

    # split the data
    train_size = int(len(X_windows) * train_ratio)
    X_train = X_windows[:train_size]
    y_train = y_windows[:train_size]
    X_valid = X_windows[train_size:]
    y_valid = y_windows[train_size:]

    if return_date:
        session_train = np.array([f"{subject_id}_{session}"] * len(X_train))
        session_valid = np.array([f"{subject_id}_{session}"] * len(X_valid))
        return X_train, y_train, X_valid, y_valid, session_train, session_valid, indices
    else:
        return X_train, y_train, X_valid, y_valid, indices



def _dates_per_trial_samples(data_list, trial_info):
    """
    For each trial, create a 1D array of dates with length equal to
    the number of samples in that trial.

    Returns:
    - List[np.ndarray], each shaped (samples_i,) of dtype str
    """
    if len(data_list) != len(trial_info):
        raise ValueError(f"Length mismatch: {len(data_list)} trials vs {len(trial_info)} trial_info")

    out = []
    for i, arr in enumerate(data_list):
        date_str = str(trial_info[i]['date'])
        n = arr.shape[0]  # samples_i
        # Use a Unicode dtype sized to the date string length
        out.append(np.full(n, date_str, dtype=f'U{len(date_str)}'))
    return out

def _dates_concatenated_per_sample(data_list, trial_info):
    """
    Create a single 1D array of dates with length equal to the total number of samples
    across all trials, repeating each trial's date by its sample count.
    """
    parts = _dates_per_trial_samples(data_list, trial_info)
    return np.concatenate(parts, axis=0) if parts else np.array([], dtype='U1')

def _concat_trials_prealloc(trial_list):
    if not trial_list:
        raise ValueError("trial_list is empty.")

    c, t = trial_list[0].shape[1], trial_list[0].shape[2]
    total_samples = sum(arr.shape[0] for arr in trial_list)
    out = np.empty((total_samples, c, t), dtype=trial_list[0].dtype)

    offset = 0
    for arr in trial_list:
        s = arr.shape[0]
        out[offset:offset + s] = arr
        offset += s
    return out

def load_centered_new_data_for_otta(train_path, train_dates, train_ratio=0.8, indices=None, return_date=False, class_num=8):
    """
    Load daily dataset from pkl files and apply sliding window. Do NOT compute covariance matrices.
    Parameters
    ----------
    train_path
    train_dates
    train_ratio
    indices
    return_date
    class_num

    Returns
    -------

    """

    train_dataset, _ = create_continues_train_test_datasets(
        h5_file_path=train_path,
        train_dates=train_dates,
        test_dates=[],
    )    

    trial = _concat_trials_prealloc(train_dataset.data)
    trial_labels = np.concatenate([
        np.full(arr.shape[0], lab, dtype=np.int64)
        for arr, lab in zip(train_dataset.data, train_dataset.labels)
    ])
    # trial: [sample, channel, time], numpy array; trial_labels: [sample], numpy array
    date = _dates_concatenated_per_sample(train_dataset.data, train_dataset.trial_info)
    if class_num == 4:
        # Remove the data with labels 45, 135, 225, 315
        valid_indices = np.where((trial_labels == 0) | (trial_labels == 90) | (trial_labels == 180) | (trial_labels == 270))[0]
        trial = trial[valid_indices]
        trial_labels = trial_labels[valid_indices]
        date = date[valid_indices]
    elif class_num == 8:
        pass
    else:
        raise ValueError('Only support class_num 4 or 8')
    
    # Channel_wise centering
    trial = trial - trial.mean(axis=2, keepdims=True)

    # one-hot encode the labels (robust to non-contiguous labels like angles)
    unique_labels = np.unique(trial_labels)
    label_to_index = {lab: idx for idx, lab in enumerate(unique_labels)}
    trial_labels_idx = np.array([label_to_index[int(lab)] for lab in trial_labels], dtype=np.int64)
    trial_labels_onehot = np.eye(len(unique_labels), dtype=np.float32)[trial_labels_idx]
    trial_labels = trial_labels_onehot
    
    # shuffle the data
    if indices is None:
        indices = np.arange(len(trial))
        np.random.shuffle(indices)
    trial = trial[indices]
    trial_labels = trial_labels[indices]
    date = date[indices]
    # split the data
    train_size = int(len(trial) * train_ratio)
    train = trial[:train_size]
    train_labels = trial_labels[:train_size]
    train_date = date[:train_size]
    valid = trial[train_size:]
    valid_labels = trial_labels[train_size:]
    valid_date = date[train_size:]
    if return_date:
        return train, train_labels, valid, valid_labels, train_date, valid_date, indices
    else:
        return train, train_labels, valid, valid_labels, indices
    

if __name__ == "__main__":

    # file_path = '/home/ubuntu/ECoG_datasets/imagery_feedback/data/hh/hh_fb_tongue.mat'
    # file_path = '/home/ubuntu/ECoG_datasets/imagery_feedback/data/hh/hh_mot_t.mat'
    # file_path = '/media/ubuntu/Storage1/ecog_data/daily_bdy_20250908/20250325'
    # train_dataset, test_dataset = load_hypergraph_data([file_path], windows_size=256, step=256, train_ratio=0.8)
    # print(train_dataset)
    
    subject_id = 1
    preprocessing_dict = {
        "sfreq": 250,
        "low_cut": 0,
        "high_cut": 40,
        "start": 0.0,
        "stop": 0.0
    }
    dataset, info = _load_bcic(subject_ids=[subject_id], dataset="2a",
                                preprocessing_dict=preprocessing_dict)    
    # split the data
    splitted_ds = dataset.split("session")
    train_dataset, test_dataset = splitted_ds["session_T"], splitted_ds["session_E"]

    # load the data
    X = np.concatenate(
        [run.windows.load_data()._data for run in train_dataset.datasets], axis=0)
    y = np.concatenate([run.y for run in train_dataset.datasets], axis=0)
    X_test = np.concatenate(
        [run.windows.load_data()._data for run in test_dataset.datasets], axis=0)
    y_test = np.concatenate([run.y for run in test_dataset.datasets], axis=0)

    print(X.shape, y.shape, X_test.shape, y_test.shape)