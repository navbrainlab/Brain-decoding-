"""
基于ResNet的深度神经网络分类训练系统
整合功能：
1. 模块化分层架构
2. 超参数全局配置
3. 混合精度训练
4. 分布式数据并行
5. 学习率策略
6. 动态数据增强
"""

import os
import json
import time
import torch
import logging
import argparse
from tqdm import tqdm
from datetime import datetime
import numpy as np
from logging.handlers import TimedRotatingFileHandler
import psutil
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from model.tools import get_mapping, DataLoaderX, compute_labels_mahalanobis
from model.ResNet import *
# from model.nt_xent import SupConLoss
import pynvml as nv
from model.plot import plot_confusion_matrix, plot_embedding
from dataset import *
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  # 导入带重启的调度器

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from dataset import *
from pathlib import Path
import glob
        

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import torch
import torch.nn as nn   
from model.ResNet import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist
import seaborn as sns
def plot_channel_cluster(data, save_path=None, 
                           method='average', metric='correlation', 
                           threshold=0.7, vmin=-1, vmax=1, 
                           cmap='jet', level=60):
    """
    绘制卷积核的聚类热图与树状图
    :param weight_tensor: 权重张量 [out_channels, in_channels, kernel_length]
    :param save_path: 保存路径
    :param method: 聚类方法 ('ward', 'average', 'complete', 'single')
    :param metric: 距离度量 ('euclidean', 'cosine', 'correlation')
    :param threshold: 聚类距离阈值
    :param vmin: 热图颜色最小值
    :param vmax: 热图颜色最大值
    :param cmap: 热图颜色映射
    :param level: 树状图截断级别
    """
    # 准备数据
    out_channels= data.shape[0]
    flattened_kernels = data
    
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(flattened_kernels)
    
    # 层次聚类
    model = AgglomerativeClustering(
        distance_threshold=0, 
        n_clusters=None,
        linkage=method,
        metric=metric
    )
    model = model.fit(corr_matrix)
    
    # 获取聚类结果和排序索引
    Z = linkage(corr_matrix, method=method, metric=metric)
    clusters = fcluster(Z, t=threshold, criterion='distance')
    idx = leaves_list(Z)
    sorted_corr = corr_matrix[idx, :][:, idx]

    # 获取dendrogram颜色映射
    dendro = dendrogram(Z, color_threshold=threshold, no_plot=True)
    leaf_colors = {}
    for leaf, color in zip(dendro['leaves'], dendro['leaves_color_list']):
        leaf_colors[leaf] = color
    # 计算每个cluster的主颜色
    cluster_color_map = {}
    for c in np.unique(clusters):
        # 找到属于该cluster的leaf
        leaves = np.where(clusters == c)[0]
        # 取第一个leaf的颜色（同cluster颜色应一致）
        cluster_color_map[c] = leaf_colors[leaves[0]]
    
    # 计算聚类边界
    bound = []
    current_cluster = clusters[idx[0]]
    count = 1
    for i in range(1, len(clusters)):
        if clusters[idx[i]] == current_cluster:
            count += 1
        else:
            bound.append(count)
            current_cluster = clusters[idx[i]]
            count = 1
    bound.append(count)
    bound = np.array(bound)
    
    # 创建图形
    fig = plt.figure(figsize=(7, 6))
    ax = plt.subplot(1,1,1)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 创建子图网格
    gs00 = GridSpecFromSubplotSpec(1, 2, subplot_spec=ax.get_subplotspec(), 
                                 wspace=0.02, hspace=0.02, 
                                 width_ratios=[3., 0.5])
    ax0 = fig.add_subplot(gs00[0, 0])  # 热图
    ax1 = fig.add_subplot(gs00[0, 1])  # 树状图
    
    # 绘制树状图
    dendrogram(linkage(corr_matrix, method=method, metric=metric),
              truncate_mode="level", p=level,
              no_labels=True, orientation='right', 
              color_threshold=threshold, ax=ax1)
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.invert_yaxis()
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    # 绘制热图
    ax0 = sns.heatmap(sorted_corr, ax=ax0, cmap=cmap, 
                     vmin=vmin, vmax=vmax, cbar=False)
    
    # 添加颜色条
    cbar_ax = fig.add_axes([-0.02, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ax0.collections[0], cax=cbar_ax)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    
    # 设置刻度
    ax_list = list(np.linspace(0, out_channels, 24, dtype=int))
    ax0.set_xticks(ax_list, ax_list, fontsize=10)
    ax0.set_yticks(ax_list, ax_list, fontsize=10)
    ax0.set_xlabel('Input Channels', fontsize=16)
    ax0.set_ylabel('Input Channels', fontsize=16)
    
    # 添加聚类边界线
    boundary = 0
    idx_bound = np.zeros(len(bound)+1)
    for i in range(len(bound)-1):
        boundary += bound[i]
        idx_bound[i+1] = boundary
        ax0.axhline(y=boundary, color='white', linestyle='--')
    idx_bound[-1] = out_channels
    
    # 保存结果
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    return idx_bound.astype(int), fig, clusters, cluster_color_map, idx


def draw_signal(data, data_info, show=False):
    """
    绘制ECoG信号的时序图像，标题中的name部分使用不同颜色展示
    
    :param data: ECoG信号数据，形状为 (128, 时间点数)
    :param name: 标题名称（格式如"001_250325_094004-label:0-trial:0"）
    """
    plt.figure(figsize=(16, 3))
    
    # 绘制ECoG信号
    img = plt.imshow(data, 
                    cmap='jet', 
                    aspect='auto', 
                    interpolation='none',
                    vmin=-40, 
                    vmax=40)
    
    # 添加颜色条和轴标签
    plt.colorbar(img, label='Intensity')
    plt.xlabel('Time')
    plt.ylabel('Channel')
    
    session = data_info['session']
    
    # 提取标签值（格式为"label:X"）
    label = data_info['label']
    trial = data_info['trial']
    
    # 创建多颜色标题组件
    title_parts = [
        ('session-','black'),
        (session, 'orange'),          # 文件ID-青色
        ('-trial:', 'black'),        # 试验前缀-灰色
        (trial, 'green'),     # 试验值-绿色
        ('-label:', 'black'),        # 标签前缀-灰色
        (label, 'red'),       # 标签值-红色
        (' ECoG visualization', 'blue')  # 固定描述-蓝色
    ]
    
    # 创建文本区域组件
    text_boxes = []
    for text, color in title_parts:
        ta = TextArea(text, textprops={
            'color': color,
            'size': 14,          # 标题字体大小
            'weight': 'bold',     # 加粗标题
        })
        text_boxes.append(ta)
    
    # 水平组装文本组件
    hpacker = HPacker(children=text_boxes, pad=0, sep=0, align="center")
    
    # 获取当前坐标轴
    ax = plt.gca()
    
    # 将标题置于图表上方居中
    anchored_box = AnchoredOffsetbox(
        loc='upper center', 
        child=hpacker,
        pad=0.5,   # 与图表的间距
        frameon=False,
        bbox_to_anchor=(0.5, 1.25),  # 在常规标题位置稍上方
        bbox_transform=ax.transAxes
    )
    ax.add_artist(anchored_box)
    
    # 调整布局并显示
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # 为标题留出空间
    if show:
        plt.show()
    else:
        plt.savefig(f'{save_root}/{session}-{trial}-{label}-cluster_channels.png')

        
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

electrod_index = np.array([
    [np.inf, 102, 120, 80, 97, 73, 56, 33, 48, 9, 23, np.inf],
    [65, 104, 118, 67, 95, 72, 57, 36, 63, 11, 24, 62],
    [96, 92, 114, 127, 91, 75, 54, 40, 2, 15, 35, 32],
    [99, 86, 111, 125, 89, 74, 55, 44, 4, 18, 39, 30],
    [101, 84, 110, 123, 79, 107, 27, 51, 6, 19, 43, 28],
    [103, 82, 100, 121, 76, 98, 31, 49, 8, 25, 52, 26],
    [116, 81, 105, 119, 66, 94, 37, 64, 10, 22, 47, 13],
    [117, 128, 93, 115, 69, 90, 41, 61, 14, 34, 1, 12],
    [109, 126, 87, 113, 71, 88, 45, 60, 16, 38, 3, 20],
    [108, 124, 85, 112, 68, 77, 53, 59, 17, 42, 5, 21],
    [np.inf, 122, 83, 106, 70, 78, 46, 58, 29, 50, 7, np.inf],
])

valid_pos = (electrod_index!=np.inf)


def restore2d(x):
    C, T = x.shape
    # x_2d = np.zeros((132, T))
    x_2d = np.full((132, T), np.nan)
    x_2d[valid_pos.flatten()] = x[(electrod_index[valid_pos]-1).astype(np.int16)]
    return x_2d.reshape(11, 12, T)


import imageio
from tqdm import tqdm

from termcolor import colored

from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker

def draw_signal(data, data_info, show=False, save_root='viz1d_signal_pic'):
    """
    绘制ECoG信号的时序图像，标题中的name部分使用不同颜色展示
    
    :param data: ECoG信号数据，形状为 (128, 时间点数)
    :param name: 标题名称（格式如"001_250325_094004-label:0-trial:0"）
    """
    plt.figure(figsize=(16, 3))
    
    # 绘制ECoG信号
    img = plt.imshow(data, 
                    cmap='jet', 
                    aspect='auto', 
                    interpolation='none',
                    vmin=-40, 
                    vmax=40)
    
    # 添加颜色条和轴标签
    plt.colorbar(img, label='Intensity')
    plt.xlabel('Time')
    plt.ylabel('Channel')
    
    session = data_info['session']
    
    # 提取标签值（格式为"label:X"）
    label = data_info['label']
    trial = data_info['trial']
    
    # 创建多颜色标题组件
    title_parts = [
        ('session-','black'),
        (session, 'orange'),          # 文件ID-青色
        ('-trial:', 'black'),        # 试验前缀-灰色
        (trial, 'green'),     # 试验值-绿色
        ('-label:', 'black'),        # 标签前缀-灰色
        (label, 'red'),       # 标签值-红色
        (' ECoG visualization', 'blue')  # 固定描述-蓝色
    ]
    
    # 创建文本区域组件
    text_boxes = []
    for text, color in title_parts:
        ta = TextArea(text, textprops={
            'color': color,
            'size': 14,          # 标题字体大小
            'weight': 'bold',     # 加粗标题
        })
        text_boxes.append(ta)
    
    # 水平组装文本组件
    hpacker = HPacker(children=text_boxes, pad=0, sep=0, align="center")
    
    # 获取当前坐标轴
    ax = plt.gca()
    
    # 将标题置于图表上方居中
    anchored_box = AnchoredOffsetbox(
        loc='upper center', 
        child=hpacker,
        pad=0.5,   # 与图表的间距
        frameon=False,
        bbox_to_anchor=(0.5, 1.25),  # 在常规标题位置稍上方
        bbox_transform=ax.transAxes
    )
    ax.add_artist(anchored_box)
    
    # 调整布局并显示
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # 为标题留出空间
    if show:
        plt.show()
    else:
        plt.savefig(f'{save_root}/{session}-{trial}-{label}-channel.png')

def load_data(data_p):
    datas = pickle.load(open(data_p, 'rb'))
    data = datas['data']
    labels = datas['label']
    return data, labels, datas

def draw_all_date(date_list):
    for date in date_list:
        data_paths = glob.glob(f'preprocessed_xzd/{date}/*.pkl')
        bar = tqdm(data_paths)
        for data_p in bar:
            bar.set_description(desc=f'processing {data_p}...')
            data, labels, datas = load_data(data_p)
            global save_root    
            save_root = f'viz1d_signal_pic/{date}/{Path(data_p).stem}'
            os.makedirs(save_root, exist_ok=True)
            for i,(d,l) in enumerate(zip(data, labels)):
                sorted_d = d[electrod_index-1][:,:]
                # sorted_d = d
                data_info = {'session':Path(data_p).stem, 'label':l, 'trial':i}
                draw_signal(sorted_d, data_info=data_info)

import matplotlib.colors as mcolors

def cluster_png(cluster_results, electrod_index, save_path='cluster_video.png', cluster_color_map=None, idx=None):
    """
    绘制并保存聚类2D图，颜色顺序与dendrogram一致。
    cluster_results: shape [128]，每个通道的聚类标签
    cluster_color_map: dict, cluster_label -> color (hex)
    idx: 通道排序（如leaves_list），用于颜色一致性
    """
    cluster_results = np.array(cluster_results)
    # 不要对cluster_results做idx排序，否则颜色会乱
    if cluster_color_map is not None:
        # 保证顺序一致
        cluster_labels = list(cluster_color_map.keys())  # 保持原始编号顺序
        colors = [cluster_color_map[c] for c in cluster_labels]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(np.arange(len(cluster_labels)+1)-0.5, len(cluster_labels))
        # 将cluster_results映射到0~n-1
        label_to_idx = {c: i for i, c in enumerate(cluster_labels)}
        mapped = np.vectorize(label_to_idx.get)(cluster_results)
    else:
        mapped = cluster_results
        cmap = 'tab20'
        norm = None
    cluster_2d = restore2d(mapped[:, np.newaxis])[:, :, 0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cluster_2d, cmap=cmap, norm=norm)
    ax.set_title(f"{Path(save_path).stem}")
    plt.axis('off')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved cluster  to {save_path}")

import numpy as np
import matplotlib.pyplot as plt
import imageio

def cluster_video(cluster_results, electrod_index, save_path='cluster_video.gif', fps=5, cmap='tab20'):
    """
    绘制并保存聚类随时间变化的2D视频。
    cluster_results: shape [n_timepoints, 128]，每一行为每个通道的聚类标签
    electrod_index: 11x12的二维数组，内容为1~128或np.inf
    save_path: gif保存路径
    fps: 帧率
    cmap: 颜色映射
    """
    n_timepoints = cluster_results.shape[0]
    frames = []
    valid_pos = (electrod_index != np.inf)
    n_clusters = int(np.max(cluster_results) + 1)
    color_map = plt.get_cmap(cmap, n_clusters)

    for t in range(n_timepoints):
        cluster_1d = cluster_results[t]  # shape: [128]
        # 还原到2d
        # low-efficiency version--------
        # cluster_2d = np.full(electrod_index.shape, np.nan)
        # for i in range(electrod_index.shape[0]):
        #     for j in range(electrod_index.shape[1]):
        #         idx = electrod_index[i, j]
        #         if np.isfinite(idx):
        #             cluster_2d[i, j] = cluster_1d[int(idx) - 1]

        # high-efficiency version--------
        cluster_2d = restore2d(cluster_1d[:,np.newaxis])[:,:,0]  # shape: [11, 12]
        # 绘图
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cluster_2d, cmap=color_map, vmin=0, vmax=n_clusters-1)
        ax.set_title(f"Time {t}")
        plt.axis('off')
        # 为gif保存帧
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(image)
        plt.close(fig)
    # 保存为gif
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Saved cluster video to {save_path}")



m1_index = np.array([1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.,  16.,  18.,  19.,  20.,  21.,  22.,
        23.,  24.,  25.,  26.,  27.,  28.,  30.,  31.,  32.,  33.,  34.,
        35.,  36.,  37.,  38.,  39.,  40.,  42.,  43.,  44.,  48.,
        49.,  47.,  51.,  52.,  54.,  55.,  56.,  57.,  61.,  62.,  63.,
        64.,  67.,  72.,  73.,  74.,  75.,  79.,  80.,  89.,  91.,  95.,
        97.,  98., 102., 104., 107., 114., 118., 120., 125., 127.],
      dtype=np.int16)

ele_mapping = torch.tensor(m1_index, dtype=torch.long)
ele_mapping, _, _ = get_mapping('/mnt/c/gaochao/CODE/BCI/XZD4class/shared_BDY_S01')
if __name__ == '__main__':
    device = 'cuda'
    # model_path = 'daily_resnet_results/ResNet_1_3/20250725_130630/checkpoints/20250325/20250402/20250325-81.pt'
    # model = ResNetv2(
    #         in_channels=74,
    #         conv1_channels=512,
    #         ).to(device)

    # model_path = '/mnt/c/gaochao/CODE/BCI/XZD4class/daily_resnet_results/ResNet_1_3/20250818_104449/checkpoints/20250325_20250326_20250327/20250401/20250325_20250326_20250327-95.pt'
    model_path = 'daily_resnet_results/ResNet_1_3/20250818_104449/checkpoints/20250325/20250401/20250325-95.pt'
    
    model_path = 'daily_resnet_results/ResNet_1_3/20250904_143030/checkpoints/20250325/20250401/20250325-95.pt'
    model_path = 'daily_resnet_results/ResNet_1_3/20250904_143030/checkpoints/20250325_20250326_20250327/20250401/20250325_20250326_20250327-95.pt'
    model_path = 'daily_resnet_results/ResNet_1_3/20250928_110038/checkpoints/20250325_20250326_20250327/20250401/20250325_20250326_20250327-25.pt'
    model = ResNetv2(
            128, 128
            ).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'),strict=True)


    # model_path = 'daily_resnet_results/ResNet_1_3/20250905_151255/checkpoints/20250325/20250401/20250325-95.pt'
    # mapping_root = 'shared_BDY_S01'
    # s1_ele_mapping, _, _ = get_mapping(mapping_root, 's1')
    # m1_ele_mapping, _, _ = get_mapping(mapping_root, 'm1')
    # model = M1S1(
    #     m1_mapping=m1_ele_mapping,
    #     s1_mapping=s1_ele_mapping,
    #     conv1_channels=512
    #     ).to(device)
    # model.load_state_dict(torch.load(model_path, map_location='cpu'),strict=True)
    data_path = '/mnt/c/gaochao/CODE/BCI/daily_bdy_20250908'
    test_dates = [
                    # 20250325,
                    # 20250326,
                    # 20250327,
                    20250329,
                    20250331,
                    20250401,
                    # 20250402,
                ]
    test_dates = [str(i) for i in test_dates]
    test_datasets = [load_daily_dataset(d, windows_size=256, step=32) for d in [os.path.join(data_path, i) for i in test_dates]]
    test_loaders = {
        date:DataLoaderX(
            dataset,
            batch_size=256,
            shuffle=True,
            drop_last=False,
            num_workers=8) for date, dataset in zip(test_dates, test_datasets)
    }
    # train_dataset, train_valid_dataset, valid_dataset, train_trail, train_labels_trail = load_train_dataset_post_resample(test_dates, windows_size=600, step=60,)
    # test_loaders = {
    #     date:DataLoaderX(
    #         dataset,
    #         batch_size=256,
    #         shuffle=True,
    #         drop_last=False,
    #         num_workers=8) for date, dataset in zip(test_dates, test_datasets)
    # }
    total_val_loss = 0
    correct = 0
    total = 0
    all_logits = []
    all_labels = []
    all_features = []
    feature_maps = []

    # 前向推理，收集所有logits、标签和feature map

    with torch.enable_grad():
        for test_date in test_dates:
            test_loader = test_loaders[test_date]
            grads_list = []     
            y_val_all = []
            y_pred_all = []
            input_all = []
            for X_val, y_val in tqdm(test_loader, desc=f'Processing {test_date}...'):
                X_val = X_val.to(device)[:, ele_mapping.long()]
                y_val = y_val.to(device)
                X_val.requires_grad = True  # 关键：使输入可求导
                output, _ = model(X_val)    # output: [B, num_classes]
                y_pred_all.append(output.argmax(dim=1).detach().abs().cpu().numpy())
                input_all.append(X_val.detach().cpu().numpy())
                for i in range(X_val.shape[0]):
                    model.zero_grad()
                    y_val_all.append(y_val[i].item())
                    # 取该样本预测最大类别
                    pred_class = output[i].argmax().item()
                    # 对该logit做反向传播
                    output[i, :].sum().backward(retain_graph=True)
                    # 收集输入的梯度
                    grads_list.append(X_val.grad[i].detach().cpu().numpy().mean(1))
                    X_val.grad.zero_()  # 清空梯度，防止累积
            y_val_all = np.array(y_val_all)
            # grads_list: [N, C, T]，N为样本数，C为input channel数，T为时间长度
            orig_grads_arr = np.stack(grads_list)  # [N, C, T]
            input_all = np.concatenate(input_all, axis=0)  # [N, C, T]
            # 对batch和时间维度平均，得到每个input channel的重要性
            
            
            import os

            # grads_arr: [N, C, T]
            # y_val_all: [N]，所有样本的标签（你需要在循环外收集所有batch的y_val拼接到一起）
            for name in [
                        # 'input','grad','input*grad',
                         'input_var']:
                if name=='grad':
                    grads_arr = orig_grads_arr
                elif name=='input':
                    grads_arr = input_all.mean(2)
                elif name=='input*grad':
                    grads_arr = input_all.mean(2)*orig_grads_arr
                elif name=='input_var':
                    grads_arr = input_all.var(axis=2)
                labels_to_calc = [0, 1, 2, 3, 'all']
                for label in labels_to_calc:
                    if label == 'all':
                        mask = np.ones(len(y_val_all), dtype=bool)
                    else:
                        mask = (y_val_all == label)
                    if mask.sum() == 0:
                        continue  # 没有该类样本
                    grads_arr_label = grads_arr[mask]  # [n_label, C, T]
                    channel_importance = grads_arr_label.mean(axis=(0))  # [C]
                    channel_importance_softmax = torch.softmax(torch.tensor(channel_importance), dim=0).numpy()
                    # 保存   
                    
                    save_dir = f'(input)-(grad)-(input*grad)_channel_importance_origrad_simple1d_input-1s/{test_date}'
                    os.makedirs(save_dir, exist_ok=True)
                    np.save(f'{save_dir}/{name}-channel_importance_label{label}.npy', channel_importance)            
                    # 可视化
                    grid = np.full(electrod_index.shape, np.nan)
                    # 有效位置的 mask
                    valid_mask = (electrod_index != np.inf)
                    grid[valid_mask] = channel_importance
                    plt.figure(figsize=(6, 5))
                    im = plt.imshow(grid, cmap='jet')
                    plt.colorbar(im, label=f'{name} Channel Importance')
                    plt.title(f'{name} Channel Importance Map')

                    # 在每个有效格子上写数值
                    for i in range(grid.shape[0]):
                        for j in range(grid.shape[1]):
                            if not np.isnan(grid[i, j]):
                                plt.text(j, i, f"{grid[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)

                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(f'{save_dir}/{test_date}_label:{label}(num{mask.sum()})_{name}_channel_importance_by_gradcam.png')
            
            
            
            
            
            
            
            # channel_importance = grads_arr.sum(axis=(0))  # [C]
            # channel_importance_softmax = torch.softmax(torch.tensor(channel_importance), dim=0).numpy()
            # grid = np.full(electrod_index.shape, np.nan)
            # # 有效位置的 mask
            # valid_mask = (electrod_index != np.inf)
            # grid[valid_mask] = channel_importance

            # np.save(f'gradcam_channel_importance/{test_date}_channel_importance_by_gradcam.npy', channel_importance)
            # # 可视化
            # plt.figure(figsize=(6, 5))
            # im = plt.imshow(grid, cmap='jet')
            # plt.colorbar(im, label='Channel Importance')
            # plt.title('Channel Importance Map')

            # # 在每个有效格子上写数值
            # for i in range(grid.shape[0]):
            #     for j in range(grid.shape[1]):
            #         if not np.isnan(grid[i, j]):
            #             plt.text(j, i, f"{grid[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)

            # plt.axis('off')
            # plt.tight_layout()
            # plt.savefig(f'gradcam_channel_importance/{test_date}_channel_importance_by_gradcam.png')

    # import amethod as am
    # choose_index = 0
    # thresh = 5
    # links=['ward','average','average','complete']
    # affs=['euclidean','cosine','cityblock','cosine']

    # date_list = os.listdir('/mnt/c/gaochao/CODE/BCI/XZD4class/preprocessed_xzd')
    # interval = 25
    # for date in date_list:
    #     data_paths = glob.glob(f'preprocessed_xzd/{date}/*.pkl')
    #     bar = tqdm(data_paths)
    #     global save_root    
    #     save_root = f'viz_channel_daily-allclass-cluster_tree-png/{date}'
    #     daily_class_datas = []
    #     daily_class_lables = []
    #     for data_p in bar:
    #         bar.set_description(desc=f'processing {data_p}...')
    #         data, labels, datas = load_data(data_p)
    #         os.makedirs(save_root, exist_ok=True)
    #         bar = tqdm(enumerate(zip(data, labels)))
    #         for i,(d,l) in bar:
    #             if l<4:
    #                 # print(int(l), d.shape)
    #                 daily_class_datas.append(d)
    #                 daily_class_lables.append(l)

    #     datas = np.hstack(daily_class_datas)
    #     labels = np.array(daily_class_lables)
    #     data_info = {'session':date, 'label':'all','trial':'all'}
    #     idx_bound, fig, clusters, cluster_color_map, idx = plot_channel_cluster(datas[:,:], save_path=f"{save_root}/{date}-{data_info['label']}-channel-clustertree.png",method=links[choose_index], 
    #             metric=affs[choose_index], threshold=thresh, vmin=-1, vmax=1,)
    #     draw_signal(datas[clusters][:,:], data_info=data_info, save_root=save_root)

    #     # draw 2d cluster map
    #     # clusters, n_clusters, sorted_corr, idx = cluster(datas[:,:], method=links[choose_index], metric=affs[choose_index], threshold=thresh)
    #     cluster_results = np.array(clusters)  # shape [n_timepoints, 128
    #     cluster_png(cluster_results, electrod_index, save_path=f"{save_root}/{date}-{data_info['label']}-cluster_2d.png",cluster_color_map=cluster_color_map, idx=idx)
            


