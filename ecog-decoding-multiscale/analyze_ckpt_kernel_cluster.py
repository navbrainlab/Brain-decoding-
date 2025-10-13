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




def compute_kernel_correlations(weight_tensor, method='cosine'):
    """
    计算卷积核之间的相关性矩阵
    :param weight_tensor: 权重张量 [out_channels, in_channels, kernel_length]
    :param method: 相关性计算方法 ('cosine'或'pearson')
    :return: 相关性矩阵 [out_channels, out_channels]
    """
    kernels = weight_tensor.detach().cpu().numpy() if hasattr(weight_tensor, 'detach') else weight_tensor
    out_channels, in_channels, kernel_length = kernels.shape
    
    # 将每个卷积核展平为向量
    flattened_kernels = kernels.reshape(out_channels, -1)  # [out_channels, in_channels*kernel_length]
    
    if method == 'cosine':
        # 计算余弦相似度
        corr_matrix = cosine_similarity(flattened_kernels)
    elif method == 'pearson':
        # 计算皮尔逊相关系数
        corr_matrix = np.corrcoef(flattened_kernels)
    else:
        raise ValueError("Unsupported correlation method. Use 'cosine' or 'pearson'")
    
    # 取绝对值，因为我们关心相似性而非方向
    corr_matrix = np.abs(corr_matrix)
    
    return corr_matrix

def cluster_kernels(corr_matrix, n_clusters=None, method='hierarchical', threshold=None):
    """
    基于相关性矩阵对卷积核进行聚类
    :param corr_matrix: 相关性矩阵 [out_channels, out_channels]
    :param n_clusters: 聚类数量(仅用于K-means)
    :param method: 聚类方法 ('hierarchical'或'kmeans')
    :param threshold: 层次聚类的距离阈值(仅用于层次聚类)
    :return: 聚类标签数组 [out_channels]
    """
    # 将相似度转换为距离(1 - similarity)
    distance_matrix = 1 - corr_matrix
    
    if method == 'hierarchical':
        # 层次聚类
        linkage_matrix = linkage(distance_matrix, method='average')
        
        if threshold is None:
            # 如果没有指定阈值，使用默认的聚类数量
            threshold = 0.7 * np.max(linkage_matrix[:, 2])
        
        clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
        n_clusters = len(np.unique(clusters))
        print(f"Hierarchical clustering identified {n_clusters} clusters at threshold {threshold:.2f}")
        
        # 可视化树状图
        plt.figure(figsize=(12, 6))
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Filter Index')
        plt.ylabel('Distance')
        plt.show()
        
    elif method == 'kmeans':
        # K-means聚类
        if n_clusters is None:
            n_clusters = 5  # 默认聚类数量
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(distance_matrix)
        print(f"K-means clustering with {n_clusters} clusters")
        
    else:
        raise ValueError("Unsupported clustering method. Use 'hierarchical' or 'kmeans'")
    
    return clusters

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os

def visualize_conv1d_filters_long(layer_name, weight_tensor, save_dir="conv_filters", cols=16):
    """
    可视化1D卷积核并保存为长图（每行固定数量）
    :param layer_name: 卷积层名称(用于文件名)
    :param weight_tensor: 权重张量 [out_channels, in_channels, kernel_length]
    :param save_dir: 保存目录
    :param cols: 每行显示的卷积核数量
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    kernels = weight_tensor.detach().cpu().numpy() if hasattr(weight_tensor, 'detach') else weight_tensor
    out_channels, in_channels, kernel_length = kernels.shape
    
    # 计算行数
    rows = int(np.ceil(out_channels / cols))
    
    # 计算图形大小（保持原始比例）
    # 每个子图的宽度和高度（根据kernel_length和in_channels比例调整）
    subplot_width = 1.0  # 每个子图的相对宽度
    subplot_height = max(0.5, in_channels / kernel_length * 1.5)  # 保持原始比例
    
    # 整体图形大小（宽度固定，高度根据行数调整）
    fig_width = cols * subplot_width * 1.2  # 每列宽度 * 列数 * 边距系数
    fig_height = rows * subplot_height * 1.5  # 每行高度 * 行数 * 边距系数
    
    # 创建图形和网格
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
    gs = gridspec.GridSpec(rows, cols, wspace=0.1, hspace=0.3)
    
    for i in range(out_channels):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        
        # 组合输入通道 (H, W) = (in_channels, kernel_length)
        img = np.zeros((in_channels, kernel_length))
        for j in range(in_channels):
            # 归一化到[0,1]并保留原始值分布
            channel_data = kernels[i, j]
            norm_data = (channel_data - np.min(channel_data)) / (np.max(channel_data) - np.min(channel_data) + 1e-8)
            img[j, :] = norm_data
        
        # 使用原图的彩色映射
        ax.imshow(img, cmap='jet', aspect='auto', interpolation='nearest')
        
        # 装饰设置
        ax.set_title(f'F{i}', fontsize=8, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 保存并关闭
    plt.suptitle(f'{layer_name} Filters (Total {out_channels} Channels)', fontsize=14, y=0.99)
    plt.savefig(f'{save_dir}/{layer_name}_filters_long.png', 
                dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def analyze_and_visualize_kernels(layer_name, weight_tensor, n_clusters=None, method='hierarchical'):
    """
    完整的分析流程: 计算相关性 -> 聚类 -> 可视化
    :param layer_name: 卷积层名称
    :param weight_tensor: 权重张量
    :param n_clusters: 聚类数量(仅用于K-means)
    :param method: 聚类方法
    """
    # 1. 计算相关性矩阵
    corr_matrix = compute_kernel_correlations(weight_tensor)
    
    # 可视化相关性矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='viridis')
    plt.title(f'{layer_name} Kernel Correlation Matrix')
    plt.show()
    
    # 2. 聚类分析
    clusters = cluster_kernels(corr_matrix, n_clusters=n_clusters, method=method)
    
    # 3. 按聚类可视化   




if __name__ == "__main__":
    model = ResNetv2(
            in_channels=128,
            conv1_channels=128,
            kernel_size=3,
            n_classes=4,
            n_layers=1,
            first_kernel_size=25,
            drop_out=0.0,
            )
    model_path = '/mnt/c/gaochao/CODE/BCI/XZD4class/daily_resnet_results/ResNet_1_3/20250812_153709/checkpoints/20250325/20250401/20250325-45.pt'
    model.load_state_dict(torch.load(model_path, map_location='cpu'),strict=True)


    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, torch.nn.Conv1d)):  # PyTorch
            # visualize_conv1d_filters_by_sort(name, module.weight.data, save_dir=f"{save_root}/conv_filter_{model_name}_m1_512check_top100_no-norm")
            print(module.weight.data.mean().item(), module.weight.data.std().item())
            analyze_and_visualize_kernels(name, module.weight.data, n_clusters=5, method='kmeans')

