import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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
import random
try:
    import torch
except Exception:
    torch = None

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.cluster.hierarchy import linkage, leaves_list

import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.cluster.hierarchy import linkage, leaves_list

def cluster_visualize_conv1d_hw_t_cluster(weights, save_name="conv1d_hw_t_clustered.png", ncols=8, method="ward", metric="euclidean"):
    """
    可视化 1D 卷积核 (out_channels 个色块，每个色块大小 HW × T)
    对 HW 个向量 (长度 T) 进行聚类，重排 HW 维度顺序
    
    :param weights: torch.Tensor, 形状 (out_channels, H*W, T)
    :param save_name: 文件名
    :param ncols: 每行显示多少个卷积核
    :param method: 聚类方法
    :param metric: 距离度量
    """
    out_channels, HW, T = weights.shape
    w_np = weights.detach().cpu().numpy()  # (out, HW, T)

    # === 对 HW 维度的向量聚类 ===
    # 每行一个向量，长度 T
    # Z = linkage(w_np[0], method=method, metric=metric)  # 只用第一个 out_channel 聚类
    # idx = leaves_list(Z)  # HW 排序索引

    # # 所有 out_channel 都按照 idx 重排 HW
    # w_np = w_np[:, idx, :]

    # 绘图
    kernels = [w_np[oc] for oc in range(out_channels)]
    labels = [f"Ch {oc}" for oc in range(out_channels)]

    total = out_channels
    nrows = math.ceil(total / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    axes = np.array(axes).reshape(nrows, ncols)

    for oc, k in enumerate(kernels):
        Z = linkage(k, method=method, metric=metric)  # 只用第一个 out_channel 聚类
        idx = leaves_list(Z)
        r, c = divmod(oc, ncols)
        ax = axes[r, c]
        ax.imshow(k[idx,:], cmap="jet", aspect="auto")
        ax.axis("off")
        ax.set_title(labels[oc], fontsize=7)

    # 隐藏多余子图
    for idx_hide in range(total, nrows * ncols):
        r, c = divmod(idx_hide, ncols)
        axes[r, c].axis("off")

    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    plt.savefig(save_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"保存卷积核(HW×T 聚类)可视化到 {save_name}")




def visualize_conv1d_outchannels(weights, save_name="conv1d_outchannels.png", ncols=8):
    """
    可视化 Conv1D 卷积核 (out_channels 个色块，每个色块大小 in_channels × kernel_size)
    
    :param weights: torch.Tensor, 形状 (out_channels, in_channels, kernel_size)
    :param save_name: 保存图片的文件名
    :param ncols: 每行显示多少个卷积核
    """
    out_channels, in_channels, K = weights.shape
    w_np = weights.detach().cpu().numpy()  # (out, in, K)

    kernels = [w_np[oc] for oc in range(out_channels)]  # 每个是 (in_channels, K)
    labels = [f"Ch {oc}" for oc in range(out_channels)]

    total = out_channels
    nrows = math.ceil(total / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, k in enumerate(kernels):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ax.imshow(k, cmap="jet", aspect="auto")
        ax.axis("off")
        ax.set_title(labels[idx], fontsize=7)

    # 隐藏多余子图
    for idx in range(total, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    plt.savefig(save_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"保存卷积核可视化到 {save_name}")


# ===== 示例 =====
# weights = torch.randn(64, 128, 15)
# visualize_conv1d_outchannels(weights, save_name="conv1d_64x128x15.png", ncols=8)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# ---------------- 示例 ----------------
if __name__ == "__main__":
    # # ===== 示例 =====
    # weights = torch.randn(64, 1, 5, 5, 15)

    # # 每行 5 个 kernel，每个 kernel 大小 H×T (5×15)，标题 F#-W#
    # visualize_conv3d_kernels_grid(weights, mode="ht", save_name="conv3d_ht_grid.png")

    # # 每行 5 个 kernel，每个 kernel 大小 W×T (5×15)，标题 F#-H#
    # visualize_conv3d_kernels_grid(weights, mode="wt", save_name="conv3d_wt_grid.png")
    save_root = './viz_ckpt_1dkernel_results'
    model = MultiScale1DCNN_v2(128,out_channels=32, num_class=4, scales=[5,15,25])
    # ckpt_path = 'daily_resnet_results/ResNet_1_3/20250923_111145/checkpoints/20250325_20250326_20250327/20250401/20250325_20250326_20250327-75.pt'
    # model.load_state_dict(torch.load(ckpt_path),strict=True)
    # HT 可视化
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")
    w1 = model.branches[0][0].weight #([64, 1, 5, 5, 15])
    w2 = model.branches[1][0].weight #([64, 1, 8, 8, 15])
    w3 = model.branches[2][0].weight #([64, 1, 10, 10, 25])
    for w in tqdm([w1,w2,w3]):
        os.makedirs(f"{save_root}/{w.shape[2:]}", exist_ok=True)
        # cluster_visualize_conv1d_hw_t_cluster(w, save_name=f"{save_root}/{w.shape[2:]}/image_woc.png")
        visualize_conv1d_outchannels(w, save_name=f"{save_root}/{w.shape[2:]}/image.png")



    # WT 可视化
    # visualize_conv3d_kernels(demo_weight, mode='wt', out_channels_to_show=16, save_name="wt_demo.png")

    # 如果要可视化自己模型的卷积层:
    # import torch.nn as nn
    # conv_layer = your_model.conv1   # 替换为你的 conv3d 层
    # visualize_conv3d_kernels(conv_layer.weight, mode='ht')
    # visualize_conv3d_kernels(conv_layer.weight, mode='wt')
