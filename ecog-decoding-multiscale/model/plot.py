import umap
import seaborn as sns
import numpy as np
import json

import matplotlib.pyplot as plt

try:
    from PyQt6.QtCore import QThread, pyqtSignal, QObject, pyqtSlot
except:
    from PyQt5.QtCore import QThread, pyqtSignal, QObject, pyqtSlot


def plot_confusion_matrix(matrix, labels, title, figsize=(10, 10), label_fontsize=20, fig_path=None):
    fig, ax = plt.subplots(figsize=figsize)
    # set fontsize
    label_fontsize = int(140 / len(labels))  # set suitable size
    sns.heatmap(np.round(matrix, 3), annot=True, cmap='Reds', fmt='g', xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={"fontsize": label_fontsize})
    plt.title(title)
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('True', fontsize=20)
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_original_fft_angle(fft, angle, fft_rec, angle_rec, title, save_path=None):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title)
    axs[0, 0].plot(fft.cpu().numpy())
    axs[0, 0].set_title('Original FFT')
    axs[1, 0].plot(angle.cpu().numpy())
    axs[1, 0].set_title('Original Time')
    axs[0, 1].plot(fft_rec.cpu().numpy())
    axs[0, 1].set_title('Reconstructed FFT')
    axs[1, 1].plot(angle_rec.cpu().numpy())
    axs[1, 1].set_title('Reconstructed Time')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_center_out_trajectory(positions, groud_truth, r2_x, r2_y, save_path):
        time_steps = np.linspace(0, groud_truth.shape[0], groud_truth.shape[0])
        fig = plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        plt.plot(time_steps, groud_truth[:, 0], color='g', label='Hand_grnd_x')
        plt.plot(time_steps, positions[:, 0], color='r', label='Hand_prctd_x')
        plt.title(f'Predicted vs True Labels for R2_x: {r2_x:.2f} R2_y: {r2_y:.2f} ')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(time_steps, groud_truth[:, 1], color='g', label='Hand_grnd_y')
        plt.plot(time_steps, positions[:, 1], color='r', label='Hand_prctd_y')
        plt.legend()
        plt.xlabel("Time [s]")
        plt.legend()

        if save_path:
            # 保存图像到指定路径
            fig.savefig(save_path, bbox_inches='tight')
        plt.close('all')


def plot_position(positions, labels, pred, acc, r2, save_path, color=None, action_id=None):
    if color is None:
        color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    fontsize = 20
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'Position Plot val acc {acc}  R2 {r2}')
    for i, label in enumerate(np.unique(labels)):
        axes[0].scatter(positions[labels == label, 0], positions[labels == label, 1], c=color[i], s=5,
                        label=action_id[str(label)])
    for i, label in enumerate(np.unique(pred)):
        axes[1].scatter(positions[pred == label, 0], positions[pred == label, 1], c=color[i], s=5,
                        label=action_id[str(label)])
    axes[0].set_title('True Labels', fontsize=fontsize)
    axes[1].set_title('Predicted Labels', fontsize=fontsize)
    axes[1].set_xlabel('X', fontsize=fontsize)
    axes[1].set_ylabel('Y', fontsize=fontsize),
    axes[0].set_xlabel('X', fontsize=fontsize)
    axes[0].set_ylabel('Y', fontsize=fontsize)
    # legend
    axes[0].legend()
    axes[1].legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_embedding(x, labels, preds, acc, distance, save_path, color=None, action_id=None):
    if color is None:
        color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    fontsize = 20
    embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    embedding.fit(x)
    embedding_point = embedding.transform(x)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'UMAP embedding val acc {acc} mahalanobis distance {distance}')
    for i, label in enumerate(np.unique(labels)):
        axes[0].scatter(embedding_point[labels == label, 0], embedding_point[labels == label, 1], c=color[i], s=5,
                        label=action_id[str(label)])
        axes[1].scatter(embedding_point[preds == label, 0], embedding_point[preds == label, 1], c=color[i], s=5,
                        label=action_id[str(label)])
    axes[0].set_title('True Labels', fontsize=fontsize)
    axes[1].set_title('Predicted Labels', fontsize=fontsize)
    axes[1].set_xlabel('Latent 1', fontsize=fontsize)
    axes[1].set_ylabel('Latent 2', fontsize=fontsize),
    axes[0].set_xlabel('Latent 1', fontsize=fontsize)
    axes[0].set_ylabel('Latent 2', fontsize=fontsize)
    # legend
    axes[0].legend()
    axes[1].legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_accuracy_curve(val_acc, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(val_acc, marker='o', linestyle='-', color='blue')
    ax.set_title('Validation Accuracy Curve')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_bands(freqs, psd, labels, low_freqs, band_widths, save_path=None, action_id=None, epoch=None):
    # compute mean psd
    psd_labels = {}
    labels_set = np.unique(labels)
    for l in labels_set:
        psd_labels[l] = np.mean(psd[labels == l], axis=(0, 1))

    # plot psd with bands in one figure

    fig, ax = plt.subplots(figsize=(10, 5))
    for l in labels_set:
        ax.plot(freqs, psd_labels[l], label=action_id[str(int(l))])
    ax.set_title(f'Epoch {epoch} Power Spectral Density')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power/Frequency (dB/Hz)')
    ax.set_ylim(0, 20)
    ax.grid(True)
    ax.legend()
    # plot bands
    for i, (low_freq, band_width) in enumerate(zip(low_freqs, band_widths)):
        ax.axvspan(float(low_freq), float(low_freq + band_width), color='gray', alpha=0.5)
    ax.legend(loc='upper right')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_filter_frequency_response(filters, fs, save_path=None, epoch=None):
    from scipy.signal import freqz
    freq_list = []
    resp_list = []

    for i in range(filters.shape[0]):
        w, h = freqz(filters[i], worN=512, fs=fs)  # worN=512点采样，fs=256Hz → 得到真实频率（Hz）
        freq_list.append(w)  # 频率 [0, fs/2]
        resp_list.append(np.abs(h))  # 幅度响应

    # 转为 [16, 512]
    resp_array = np.array(resp_list)

    # 平均频率响应
    mean_resp = np.mean(resp_array, axis=0)

    # 绘图
    plt.figure(figsize=(10, 6))
    for i in range(resp_array.shape[0]):
        plt.plot(freq_list[i], resp_array[i], color='gray', alpha=0.4)
    plt.plot(freq_list[0], mean_resp, color='red', linewidth=2.5, label='Mean Response')

    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    plt.title(f'Epoch {epoch} Frequency Responses of Filters and Their Mean', fontsize=16)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def combine_animations(animations):
    """
    Combine multiple animations into a single animation.
    :param animations: List of paths to the individual animations.
    :param save_path: Path to save the combined animation.
    """
    from PIL import Image

    for i, (key, pngs) in enumerate(animations.items()):
        images = []
        for png in pngs:
            img = Image.open(png)
            images.append(img)
        save_path = pngs[0].replace('.png', '.gif')
        images[0].save(save_path, save_all=True, append_images=images[1:], duration=100, loop=0)


def plot_trajectory(trials, fixed_colors, session, mode, fig_path=None):
    length = len(trials[0])

    for i in range(length):
        plt.figure(figsize=(16, 12))

        # 存储已经绘制过的标签，避免重复图例
        plotted_labels = set()

        for marker, trial in trials.items():
            trial = trial[i]
            # 提取轨迹点
            pos = trial['pos']
            x_coords = pos[:, 0]
            y_coords = pos[:, 1]

            # 获取颜色
            color = fixed_colors[trial["labels"][0]]

            # 只在图例中添加新标签
            if trial["labels"][0] not in plotted_labels:
                label = trial["labels"][0]
                plotted_labels.add(trial["labels"][0])
            else:
                label = None

            # 绘制轨迹
            plt.plot(x_coords, y_coords, color=color, marker='o', label=label, markersize=1, linestyle='solid')

            # 绘制平滑轨迹
            for size, (alpha, ewma) in enumerate(trial[mode].items()):
                smoothing = trial[mode][alpha]
                x_coords = smoothing[f'{mode}_pos'][:, 0]
                y_coords = smoothing[f'{mode}_pos'][:, 1]
                plt.plot(x_coords, y_coords, color=color, marker='+', label=label, markersize=int(size + 1) * 4)

            # 单独绘制起始点(0,0)，确保显示
            plt.scatter(x_coords[0], y_coords[0], color=color, marker='o')
            # break

        # 添加图例（自动去重）
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # 去重图例项
        plt.legend(by_label.values(), by_label.keys())
        # plt.legend(loc='upper right', fontsize=12)

        # 添加标题和坐标轴标签
        plt.title(session + f"-{mode}-Trajectories")
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])

        # 显示图形
        plt.grid(True)
        if fig_path:
            plt.savefig(fig_path.replace('.png', f'_{i}.png'), bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close()


def plot_sequence(sequence, mask, close_fig=None,
                  nrows=12, ncols=11, figsize=(100, 50), fontsize=40, cmap="RdBu_r", gridspec_kw=None,
                  fig_path=None, title=None):
    if gridspec_kw is None:
        gridspec_kw = {'wspace': 0.1, 'hspace': 0.1}

    if close_fig is None:
        close_fig = [[0, 0], [0, 10], [11, 0], [11, 10]]

    idx = 0
    # 设置子图之间间距
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, squeeze=True, gridspec_kw=gridspec_kw,
                            figsize=figsize)

    for i in range(nrows):
        for j in range(ncols):
            if [i, j] not in close_fig:
                ax = axs[i, j]
                if i == nrows - 1 or j == 0:
                    # if mask[idx] == 1:
                    #     ax.plot(sequence[idx], color='red')
                    # else:
                    ax.plot(sequence[idx], color='black')
                else:
                    # if mask[idx] == 1:
                    #     ax.plot(sequence[idx], color='red')
                    # else:
                    ax.plot(sequence[idx], color='black')
                idx += 1
            else:
                ax = axs[i, j]
                ax.grid(False)
                ax.axis('off')

    for i in range(nrows):
        for j in range(ncols):
            if [i, j] not in close_fig:
                if i == nrows - 1 and j != 0:
                    axs[i, j].set_yticklabels([])
                    axs[i, j].tick_params(labelsize=fontsize)
                elif j == 0 and i != nrows - 1:
                    axs[i, j].set_xticklabels([])
                    axs[i, j].tick_params(labelsize=fontsize)
                elif i == nrows - 1 and j == 0:
                    axs[i, j].tick_params(labelsize=fontsize)
                    pass
                else:
                    axs[i, j].set_xticklabels([])
                    axs[i, j].set_yticklabels([])

    fig.supxlabel('Times (s)', fontsize=fontsize)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05)

    sm = plt.cm.ScalarMappable(cmap=cmap)

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cax, fraction=0.5, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize)
    fig.suptitle(title, fontsize=fontsize)

    if fig_path:
        fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    plt.clf()


class WorkerBase(QObject):
    def __init__(self, parent: QObject = None):
        super().__init__()

        self._parent = parent


class Plot(WorkerBase):
    def __init__(self, parent: QObject = None, ):
        super().__init__(parent)

    @pyqtSlot(dict)
    def doing(self, things):
        """
        Plot the sequence with a grid of subplots.
        :param sequence: The sequence to plot.
        :param fig_path: The path to save the figure.
        :param title: The title of the figure.
        """
        plot_sequence(things['sequence'], things['mask'], fig_path=things['fig_path'], title=things['title'])


class PlotEmbedding(WorkerBase):
    def __init__(self, parent: QObject = None, ):
        super().__init__(parent)

    @pyqtSlot(dict)
    def doing(self, things):
        """
        Plot the sequence with a grid of subplots.
        :param sequence: The sequence to plot.
        :param fig_path: The path to save the figure.
        :param title: The title of the figure.
        """
        plot_embedding(things['sequence'], labels=things['labels'], preds=things['preds'],
                       acc=things['acc'], save_path=things['fig_path'])


class WorkerThread(QThread):
    signal = pyqtSignal(dict)

    def __init__(self, parent: QObject = None, worker: WorkerBase = None):
        super().__init__()

        self._parent = parent
        self._worker = worker

        self._worker.moveToThread(self)

        self.signal.connect(self._worker.doing)

    @property
    def worker(self):
        return self._worker

    def send_2_thread(self, things):
        self.signal.emit(things)


if __name__ == '__main__':
    # test confusion matrix
    from pyriemann.utils.distance import distance_riemann
    from pyriemann.estimation import Covariances
    from pyriemann.embedding import TSNE

    x = np.random.randn(100, 128, 600)
    x = Covariances().fit_transform(x)
    embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric=distance_riemann)
    embedding.fit(x)
