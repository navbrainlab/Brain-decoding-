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
import math
from scipy.cluster.hierarchy import linkage, leaves_list
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
def visualize_conv3d_kernels_grid(weights, mode="ht", save_name="conv3d_kernels_grid.png"):
    """
    可视化 Conv3D 权重
    :param weights: torch.Tensor, 形状 (out_channels, in_channels, H, W, T)
    :param mode: 
        "ht"  -> 每行画 W 个 kernel (H×T)
        "wt"  -> 每行画 H 个 kernel (W×T)
        "hw"  -> 每行画 T 个 kernel (H×W)
        "hwt" -> 每行最多画 8 个 kernel (H*W × T)
    """
    out_channels, in_channels, H, W, T = weights.shape
    assert in_channels == 1, "当前仅支持 in_channel=1 的情况"

    w_np = weights.squeeze(1).detach().cpu().numpy()  # (out, H, W, T)

    if mode == "ht":
        ncols = W
        nrows = out_channels
        kernels = [[w_np[oc, :, wi, :] for wi in range(W)] for oc in range(out_channels)]
        col_labels = [f"W:{j}" for j in range(W)]
    elif mode == "wt":
        ncols = H
        nrows = out_channels
        kernels = [[w_np[oc, hi, :, :] for hi in range(H)] for oc in range(out_channels)]
        col_labels = [f"H:{j}" for j in range(H)]
    elif mode == "hw":
        ncols = T
        nrows = out_channels
        kernels = [[w_np[oc, :, :, ti] for ti in range(T)] for oc in range(out_channels)]
        col_labels = [f"T:{j}" for j in range(T)]
    elif mode == "hwt":
        ncols = 8  # 每行最多 8 个
        nrows = math.ceil(out_channels / ncols)
        kernels = []
        col_labels = []
        for oc in range(out_channels):
            kernels.append(w_np[oc].reshape(H*W, T))
            col_labels.append(f"Ch:{oc}")
    else:
        raise ValueError("mode 必须是 ht, wt, hw 或 hwt")

    if mode == "hwt":
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
        axes = np.array(axes).reshape(nrows, ncols)
        method="ward"
        metric="euclidean"
        for idx, k in enumerate(kernels):
            # Z = linkage(k, method=method, metric=metric)  # 只用第一个 out_channel 聚类
            # ids = leaves_list(Z)
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            # ax.imshow(k[ids,:], cmap="jet", aspect="auto")
            ax.imshow(k[:,:], cmap="jet", aspect="auto")
            ax.axis("off")
            ax.set_title(col_labels[idx], fontsize=7)

        # 隐藏多余的空 subplot
        for idx in range(len(kernels), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].axis("off")
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*1.5))

        if nrows == 1:
            axes = np.expand_dims(axes, 0)
        if ncols == 1:
            axes = np.expand_dims(axes, 1)

        for i in range(nrows):
            for j in range(ncols):
                ax = axes[i, j]
                ax.imshow(kernels[i][j], cmap="jet", aspect="auto")
                ax.axis("off")
                ax.set_title(f"Channel:{i+1}-{col_labels[j]}", fontsize=7)

    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    plt.savefig(save_name, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"保存可视化到 {save_name}")


# def visualize_conv3d_kernels_grid(weights, mode="ht", save_name="conv3d_kernels_grid.png"):
#     """
#     可视化 Conv3D 权重
#     :param weights: torch.Tensor, 形状 (out_channels, in_channels, H, W, T)
#     :param mode: 
#         "ht" -> 每行画 W 个 kernel (H×T)
#         "wt" -> 每行画 H 个 kernel (W×T)
#         "hw" -> 每行画 T 个 kernel (H×W)
#     """
#     out_channels, in_channels, H, W, T = weights.shape
#     assert in_channels == 1, "当前仅支持 in_channel=1 的情况"

#     w_np = weights.squeeze(1).detach().cpu().numpy()  # (out, H, W, T)

#     if mode == "ht":
#         ncols = W
#         kernels = [[w_np[oc, :, wi, :] for wi in range(W)] for oc in range(out_channels)]
#         col_labels = [f"W:{j}" for j in range(W)]
#     elif mode == "wt":
#         ncols = H
#         kernels = [[w_np[oc, hi, :, :] for hi in range(H)] for oc in range(out_channels)]
#         col_labels = [f"H:{j}" for j in range(H)]
#     elif mode == "hw":
#         ncols = T
#         kernels = [[w_np[oc, :, :, ti] for ti in range(T)] for oc in range(out_channels)]
#         col_labels = [f"T:{j}" for j in range(T)]
#     else:
#         raise ValueError("mode 必须是 ht, wt 或 hw")

#     nrows = out_channels
#     fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*1.5, nrows*1.5))

#     if nrows == 1:
#         axes = np.expand_dims(axes, 0)
#     if ncols == 1:
#         axes = np.expand_dims(axes, 1)

#     for i in range(nrows):
#         for j in range(ncols):
#             ax = axes[i, j]
#             ax.imshow(kernels[i][j], cmap="jet", aspect="auto")
#             ax.axis("off")
#             # 每个色块上方标注 Channel 和索引
#             ax.set_title(f"Channel:{i+1}-{col_labels[j]}", fontsize=7)

#     plt.subplots_adjust(wspace=0.05, hspace=0.3)
#     plt.savefig(save_name, dpi=200, bbox_inches="tight")
#     plt.close()
#     print(f"保存可视化到 {save_name}")


# # ===== 示例 =====
# weights = torch.randn(64, 1, 5, 5, 15)

# # 每行 5 个 kernel，每个 kernel 大小 H×T
# visualize_conv3d_kernels_grid(weights, mode="ht", save_name="conv3d_ht_grid.png")

# # 每行 5 个 kernel，每个 kernel 大小 W×T
# visualize_conv3d_kernels_grid(weights, mode="wt", save_name="conv3d_wt_grid.png")

# # 每行 64 个 kernel，每个 kernel 大小 H×W
# visualize_conv3d_kernels_grid(weights, mode="hw", save_name="conv3d_hw_grid.png")


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
    
    save_root = './viz_ckpt_3dkernel_results'
    model = Spatial3dconvMultiscale_v2_sweep(
        scale1_small=[5,5,15],
        scale1_mid=[8,8,15],
        scale1_large=[10,10,25],
        out_channels=128
        )
    # ckpt_path = 'daily_resnet_results/ResNet_1_3/20250917_134029/checkpoints/20250325_20250326_20250327/20250401/20250325_20250326_20250327-75.pt'
    # model.load_state_dict(torch.load(ckpt_path))
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")
    # HT 可视化
    w1 = model.branches[0][0].weight #([64, 1, 5, 5, 15])
    w2 = model.branches[1][0].weight #([64, 1, 8, 8, 15])
    w3 = model.branches[2][0].weight #([64, 1, 10, 10, 25])
    for w in tqdm([w1,w2,w3]):
        os.makedirs(f"{save_root}/{w.shape[2:]}", exist_ok=True)
        # visualize_conv3d_kernels_grid(w, mode='ht', save_name=f"{save_root}/{w.shape[2:]}/ht_demo.png")
        # visualize_conv3d_kernels_grid(w, mode='wt', save_name=f"{save_root}/{w.shape[2:]}/wt_demo.png")
        # visualize_conv3d_kernels_grid(w, mode='hw', save_name=f"{save_root}/{w.shape[2:]}/hw_demo.png")
        visualize_conv3d_kernels_grid(w, mode='hwt', save_name=f"{save_root}/{w.shape[2:]}/wo_cluster_hwt_demo.png")

    # WT 可视化
    # visualize_conv3d_kernels(demo_weight, mode='wt', out_channels_to_show=16, save_name="wt_demo.png")

    # 如果要可视化自己模型的卷积层:
    # import torch.nn as nn
    # conv_layer = your_model.conv1   # 替换为你的 conv3d 层
    # visualize_conv3d_kernels(conv_layer.weight, mode='ht')
    # visualize_conv3d_kernels(conv_layer.weight, mode='wt')
