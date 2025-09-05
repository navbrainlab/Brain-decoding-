import argparse
import os
from dataset import read_pkl, slide_window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
from matplotlib import gridspec

def cluster(matrix,link='average',aff='cosine'):
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None,linkage=link,metric=aff)
    model = model.fit(matrix)
    idx=[]
    R=plot_dendrogram(model, no_plot=True)
    idx=np.array(R['leaves'])
    return idx

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = model.labels_.shape[0]
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]
    ).astype(float)

    R=dendrogram(linkage_matrix,**kwargs)
    return R

def linear_correlation_plot():
    electrode_mapping_df = pd.read_excel(os.path.join(args.data_path, '..', 'electrode_mapping.xlsx'), header=None)
    electrode_mapping = electrode_mapping_df.values
    upper = []
    lower = []
    for i in range(electrode_mapping.shape[0]):
        for j in range(electrode_mapping.shape[1]):
            if isinstance(electrode_mapping[i, j], int):
                if i <= j:
                    upper.append(electrode_mapping[i, j])
                else:
                    lower.append(electrode_mapping[i, j])
    electrode_idx = np.array(upper+lower) - 1

    os.makedirs(args.save_path, exist_ok=True)
    for date in args.dates:
        date_path = os.path.join(args.data_path, date)
        data, data_labels, data_session = read_pkl(date_path, session=16)
        data, data_labels = slide_window(data, list(data_labels), windows_size=args.window_size, step=args.window_stride, start_from=args.start_from)
        n_sample, n_channel, n_time_point = data.shape
        # print(data.shape)
        # print(data_labels.shape)

        avg_corr_matrix = np.zeros((n_channel, n_channel))
        for i in range(n_sample):
            corr_matrix = np.corrcoef(data[i])
            avg_corr_matrix += corr_matrix
        avg_corr_matrix /= n_sample

        high_corr_channels = np.argwhere((avg_corr_matrix > 0.9) & (avg_corr_matrix < 1)) + 1
        # print(high_corr_channels)
        with open(os.path.join(args.save_path,'high_corrcoef_channel.txt'), 'a') as f:
            f.write(f"\n{date}: {len(high_corr_channels)}\n")     
            for coord in high_corr_channels:
                f.write(f"{coord[0]:3d}, {coord[1]:3d}\n")     

        # electrode_idx = cluster(avg_corr_matrix)
        avg_corr_matrix = avg_corr_matrix[electrode_idx][:, electrode_idx]
        # 128 * 11 * 12

        plt.figure(figsize=(12, 10))
        plt.imshow(avg_corr_matrix, 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1, 
                interpolation='none')
        plt.hlines(len(upper), 0, 127, linestyles='dashed', colors='black')
        plt.vlines(len(upper), 0, 127, linestyles='dashed', colors='black')
        # Add colorbar
        cbar = plt.colorbar(aspect=30)
        cbar.set_label('Average Correlation Coefficient')
        # Add labels and title
        plt.xlabel('Channel Index')
        plt.xticks(range(0,n_channel, 10), electrode_idx[::10] + 1)
        plt.ylabel('Channel Index')
        plt.yticks(range(0,n_channel, 10), electrode_idx[::10] + 1)
        plt.title(f'{date} Average Channel Correlation Matrix Sort by Space')
        # Adjust layout and show
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_path, f'corrcoef_{date}_space.png'))

def matrix_correlation_plot():
    electrode_mapping_df = pd.read_excel(os.path.join(args.data_path, '..', 'electrode_mapping.xlsx'), header=None)
    electrode_mapping = electrode_mapping_df.values
    electrode_mapping[0,0] = 129
    electrode_mapping[0,-1] = 130
    electrode_mapping[-1,0] = 131
    electrode_mapping[-1,-1] = 132
    electrode_mapping = electrode_mapping.astype(int)
    electrode_mapping -= 1

    for date in args.dates:
        os.makedirs(os.path.join(args.save_path, date), exist_ok=True)
        date_path = os.path.join(args.data_path, date)
        data, data_labels, data_session = read_pkl(date_path, session=16)
        data, data_labels = slide_window(data, list(data_labels), windows_size=args.window_size, step=args.window_stride, start_from=args.start_from)
        n_sample, n_channel, n_time_point = data.shape
        # print(data.shape)
        # print(data_labels.shape)

        avg_corr_matrix = np.zeros((n_channel, n_channel))
        for i in range(n_sample):
            corr_matrix = np.corrcoef(data[i])
            avg_corr_matrix += corr_matrix
        avg_corr_matrix /= n_sample

        high_corr_channels = np.argwhere((avg_corr_matrix > 0.9) & (avg_corr_matrix < 1)) + 1
        # print(high_corr_channels)
        with open(os.path.join(args.save_path,'high_corrcoef_channel.txt'), 'a') as f:
            f.write(f"\n{date}: {len(high_corr_channels)}\n")     
            for coord in high_corr_channels:
                f.write(f"{coord[0]:3d}, {coord[1]:3d}\n")     

        # electrode_idx = cluster(avg_corr_matrix)
        avg_corr_matrix = np.concatenate((avg_corr_matrix, np.zeros((n_channel,4))), axis=1)
        avg_corr_matrix = avg_corr_matrix.reshape(n_channel, -1)
        avg_corr_matrix = avg_corr_matrix[:, electrode_mapping.reshape(-1)]
        avg_corr_matrix = avg_corr_matrix.reshape(n_channel,11,12) # 128*11*12
        i = 0
        for c in electrode_mapping[electrode_mapping < n_channel].reshape(-1):
            plt.figure(figsize=(12, 10))
            plt.imshow(avg_corr_matrix[c], 
                    cmap='coolwarm', 
                    vmin=-1, 
                    vmax=1, 
                    interpolation='none')
            # Add colorbar
            cbar = plt.colorbar(aspect=30)
            cbar.set_label('Average Correlation Coefficient')
            # Add labels and title
            plt.title(f'{date} Channel{c+1} Average Channel Correlation Matrix')
            plt.yticks(range(11), range(1,12))
            plt.xticks(range(12), range(1,13))
            # Adjust layout and show
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_path, date, f'corrcoef_{date}_{i+1}.png'))    
            plt.close()
            i += 1

def signal_plot():
    for date in args.dates:
        os.makedirs(os.path.join(args.save_path, date), exist_ok=True)
        date_path = os.path.join(args.data_path, date)
        data, data_labels, data_session = read_pkl(date_path, session=16)
        sample = data[50]
        print(sample.shape)
        n_channel, n_time_point = sample.shape
        sample -= np.mean(sample, axis=1, keepdims=True)
        sample /= np.std(sample, axis=1, keepdims=True)

        n_channel = 5

        fig = plt.figure(figsize=(10, 2*n_channel))
        # 使用GridSpec创建网格
        gs = gridspec.GridSpec(n_channel, 1, figure=fig)

        # 创建子图
        axes = []
        for c in range(n_channel):
            ax = fig.add_subplot(gs[c, 0], sharex=axes[0] if c > 0 else None)
            axes.append(ax)
            
            # 绘制数据
            ax.plot(sample[c, :])
            ax.axis('off')
            
            # 只为最底部的子图添加x轴标签和刻度
            # ax.set_ylabel('Amplitude')
            # if c < n_channel - 1:
            #     ax.set_xticks([])
            #     ax.set_xlabel('')        
            # else:
            #     ax.set_xlabel('Time points')
            #     ax.set_xticks(range(0, len(sample[1]), 100))

        # 添加标题
        plt.suptitle('Sample signal from 5 channels')
        plt.tight_layout()
        plt.show()
        break

def parse_args():
    """参数配置"""
    parser = argparse.ArgumentParser(description='ResNet1D训练系统')

    # 数据参数
    parser.add_argument('--data_path', type=str,
                        default='/media/ubuntu/Storage/ecog_data/preprocessed',
                        help='脑电数据根目录')
    parser.add_argument('--save_path', type=str, default='/home/ubuntu/ecog_proj/channel_correlation')
    parser.add_argument('--dates', nargs='+',
                        default=[
                                 '20250319',
                                 '20250320',
                                 '20250321',
                                 '20250323',
                                 '20250324',
                                 '20250325',
                                 '20250326',
                                 '20250327',
                                 '20250329',
                                 '20250331',
                                 '20250401',
                                 '20250402',
                                 '20250409'
                                 ],                   
                        help='实验日期列表，按时间顺序排列')

    parser.add_argument('--window_size', type=int, default=768)
    parser.add_argument('--window_stride', type=int, default=1024)
    parser.add_argument('--start_from', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    signal_plot()
