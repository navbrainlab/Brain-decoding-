import glob
import os
import numpy as np
import torch
from joblib import delayed, Parallel
import pickle
from scipy.interpolate import interp1d
from pyriemann.estimation import Covariances
from pyriemann.channelselection import ElectrodeSelection
from pyriemann.utils.distance import distance
from pyriemann.classification import MDM

class ElectrodeSelectionLoop(ElectrodeSelection):
    def fit(self, X, y=None, sample_weight=None):
        """Find the optimal subset of electrodes.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None | ndarray, shape (n_matrices,), default=None
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : ElectrodeSelection instance
            The ElectrodeSelection instance.
        """
        if y is None:
            y = np.ones((X.shape[0]))

        mdm = MDM(metric=self.metric, n_jobs=self.n_jobs)
        mdm.fit(X, y, sample_weight=sample_weight)
        self.covmeans_ = mdm.covmeans_

        n_classes, n_channels, _ = self.covmeans_.shape

        self.dist_ = []
        self.subelec_ = list(range(n_channels))
        self.loop_subelec_ = []
        while (len(self.subelec_)) > self.nelec:
            di = np.zeros((len(self.subelec_), 1))
            for idx in range(len(self.subelec_)):
                sub = self.subelec_[:]
                sub.pop(idx)
                di[idx] = 0
                for i in range(n_classes):
                    for j in range(i + 1, n_classes):
                        di[idx] += distance(
                            self.covmeans_[i][:, sub][sub, :],
                            self.covmeans_[j][:, sub][sub, :],
                            metric=mdm.metric_dist,
                        )

            torm = di.argmax()
            self.dist_.append(di.max())
            self.loop_subelec_.insert(0, self.subelec_[torm])
            self.subelec_.pop(torm)
            
        self.loop_subelec_.insert(0, self.subelec_[0])
        return self


def obtain_data_path(paths: str, path_list: list, task: str) -> list:
    """
    Loop to convert files from all subdirectories.
    """
    if len(glob.glob(paths + "/notes.txt")):
        path_list.append(paths)
    else:
        for child_path in os.listdir(paths):
            if not os.path.isdir(os.path.join(paths, child_path)):
                continue
            path_list = obtain_data_path(
                os.path.join(paths, child_path), path_list, task
            )
    return path_list


def interp(data, windows_size, method='interp1d'):
    f = interp1d(np.linspace(0, 1, data.shape[-1]), data, kind='cubic', fill_value="extrapolate")
    return f(np.linspace(0, 1, windows_size))


def slide_window(data: list, label: list, windows_size: int=500, step: int=100, max_trial_length: int=2000, trial_handle_method: str='interp') -> tuple[list, list, list]:
    temp_slices = []
    labels_slices = []
    trial_padding_mask = []
    for i, temp in enumerate(data):
        padding_mask = []
        if temp.shape[-1] < windows_size:
            if trial_handle_method == 'padding':
                padding = max_trial_length - temp.shape[-1]
                temp = np.pad(temp, ((0, 0), (0, padding)), 'constant')
                padding_mask = np.zeros(max_trial_length)
                padding_mask[-padding:] = 1
            elif trial_handle_method == 'interp':
                interp(temp, windows_size)
            temp_slices.append(temp)
            labels_slices.append(label[i])
            trial_padding_mask.append(padding_mask)
            continue
        for j in range(0, temp.shape[-1] - windows_size, step):
            temp_slices.append(temp[..., j:j + windows_size])
            labels_slices.append(label[i])
    data = torch.tensor(np.array(temp_slices), dtype=torch.float32)
    labels = torch.tensor(np.array(labels_slices), dtype=torch.int32)
    return data, labels

def channel_contribution(neu, label=None, channel_num=36):
    '''
    Used to calcuate the contribution of channels. 

    Args:
        neu: The neu data shape as Trial X Channel X Time
        label: The label shape as Trial X 1
        channel_num: The number of channels to be selected
    '''
    neu = Covariances(estimator='lwf').fit_transform(np.array(neu))
    selectChs = ElectrodeSelectionLoop(nelec=1, n_jobs=-1)
    selectChs.fit(neu, label)
    chs = np.array(selectChs.loop_subelec_)
    return chs[:channel_num]

def _load_data(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['data'], data['label']

def read_data_motion_session(root, session_used=[]):
    file_list = []
    for session_file in session_used:
        print('Searching session: ', session_file)
        filepath = os.path.join(root, session_file)
        files = [os.path.join(filepath, f) for f in os.listdir(filepath) if f.endswith('.pkl')]
        file_list.extend(files)
    neu_all = []
    label_all = []
    print('Loading data...')
    # out = Parallel(n_jobs=-1)(delayed(_load_data)(file) for file in file_list)
    out = [_load_data(file) for file in file_list]
    for data in out:
        neu, label = data
        label_temp = []
        neu_temp = []
        for i in range(len(label)):
            if label[i]!=4:
                label_temp.append(label[i])
                neu_temp.append(neu[i])
        if label_temp:
            neu_all.append(neu_temp)
            label_all.append(np.array(label_temp))
    try:
        neu_all = np.concatenate(neu_all)
        print('data size is', neu_all.shape)
    except ValueError:
        neu = []
        for i in range(len(neu_all)):
            neu += neu_all[i]
        neu_all = neu
    label_all = np.concatenate(label_all)
    return neu_all, label_all