import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import torch
import torch.nn as nn   
from model.ResNet import *
from tqdm import tqdm

def visualize_conv1d_filters_by_sort(layer_name, weight_tensor, save_dir="conv_filters"):
    """
    可视化1D卷积核并分层保存
    :param layer_name: 卷积层名称(用于文件名)
    :param weight_tensor: 权重张量 [out_channels, in_channels, kernel_length]
    :param save_dir: 保存目录
    """
    import os
    import matplotlib
    matplotlib.use('Agg')  # 无头模式，避免GUI
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    kernels = weight_tensor.detach().cpu().numpy() if hasattr(weight_tensor, 'detach') else weight_tensor
    out_channels, in_channels, kernel_length = kernels.shape
    out_channels = min(out_channels, draw_filter_num)  # 限制绘制数量
    kernels = kernels[kernel_rank[:out_channels], :, :]  # 按rank排序
    
    # 布局参数（接近原图比例）
    cols = 16
    rows = int(np.ceil(out_channels / cols))
    figsize = (cols * 2.5, rows * 2.3)  # 宽度>高度
    
    # 创建带间距控制的网格
    plt.figure(figsize=figsize, facecolor='white')
    spec = gridspec.GridSpec(rows, cols, wspace=0.1, hspace=0.2)
    
    for i in tqdm(range(out_channels)):
        ax = plt.subplot(spec[i])
        
        # 组合输入通道 (H, W) = (in_channels, kernel_length)
        img = np.zeros((in_channels, kernel_length))
        for j in range(in_channels):
            # 归一化到[0,1]并保留原始值分布
            channel_data = kernels[i, j]
            norm_data = (channel_data - np.min(channel_data)) / (np.max(channel_data) - np.min(channel_data) + 1e-8)
            img[j, :] = norm_data
            # img[j, :] = channel_data
        
        # 使用原图的彩色映射
        ax.imshow(img, cmap='jet', aspect='auto', interpolation='nearest')
        
        # 装饰设置
        ax.set_title(f'Filter {kernel_rank[i]}', fontsize=9, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 保存并关闭
    plt.subplots_adjust(top=0.96, left=0.03, right=0.97, bottom=0.03)
    plt.suptitle(f'{layer_name} Filters', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 紧凑布局，保留标题
    plt.savefig(f'{save_dir}/{layer_name}_filters.png', 
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_conv1d_filters(layer_name, weight_tensor, save_dir="conv_filters"):
    """
    可视化1D卷积核并分层保存
    :param layer_name: 卷积层名称(用于文件名)
    :param weight_tensor: 权重张量 [out_channels, in_channels, kernel_length]
    :param save_dir: 保存目录
    """
    import os
    import matplotlib
    matplotlib.use('Agg')  # 无头模式，避免GUI
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    kernels = weight_tensor.detach().cpu().numpy() if hasattr(weight_tensor, 'detach') else weight_tensor
    out_channels, in_channels, kernel_length = kernels.shape
    
    # 布局参数（接近原图比例）
    cols = 32
    rows = int(np.ceil(out_channels / cols))
    figsize = (cols * 2.5, rows * 2.3)  # 宽度>高度
    
    # 创建带间距控制的网格
    plt.figure(figsize=figsize, facecolor='white')
    spec = gridspec.GridSpec(rows, cols, wspace=0.1, hspace=0.2)
    
    for i in tqdm(range(out_channels)):
        ax = plt.subplot(spec[i])
        
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
        ax.set_title(f'Filter {i}', fontsize=9, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 保存并关闭
    plt.subplots_adjust(top=0.96, left=0.03, right=0.97, bottom=0.03)
    plt.suptitle(f'{layer_name} Filters', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 紧凑布局，保留标题
    plt.savefig(f'{save_dir}/{layer_name}_filters.png', 
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

kernel_rank_multi_day = [485, 136, 313, 287, 203, 400, 310, 209, 510, 116,  85, 229, 375, 506,
 29, 463, 230, 169, 204, 271, 371, 277, 425, 241, 464, 291,  93, 367,
264,   7, 147,   4, 129, 222, 331, 494, 127,  62, 432, 227, 122,  86,
186, 321, 306, 303, 373, 216,  60, 399, 113, 467, 185, 452,  82, 106,
431, 132, 295,  50, 309, 493,  83, 372, 140, 477, 404, 479, 126,  52,
162, 340, 472, 108,  47,  49, 428, 237,  98, 275, 427, 322, 149, 511,
419, 498, 247, 487,  44,  66, 434, 109, 274, 210, 102, 457, 305, 141,
273, 492,  27, 468, 325, 383, 158, 197,   9, 348, 448, 103, 252, 195,
 11, 201, 182, 415, 131, 471, 267, 188, 395, 342, 302,   3, 402,   0,
 40, 111, 451, 221, 407, 123, 179,  19, 443, 362, 334, 379, 505, 369,
398, 405,  79, 166, 335, 330, 187, 168, 420,  54, 391,  12, 385, 315,
450,  35, 193,   2,  75, 503, 157, 351, 211,  59, 242, 301, 365, 249,
161, 353, 466, 159, 417, 445, 460, 298, 366, 115, 423, 483, 300, 150,
381, 316, 290, 473, 281, 160, 408,  39, 171, 293, 341,  97, 454, 144,
350, 280, 235, 212, 497, 164,  10, 156, 245, 297,  33,  53, 480, 268,
232, 509,  41, 374, 449,  80, 112, 441,  36,  74, 478, 181, 174, 118,
225, 416, 361, 134, 255, 263,  95,  34, 243, 240, 364,  13,   5, 312,
 43, 202, 215, 289,  48, 110, 307, 121, 377, 435, 324, 474, 191, 455,
 96, 368, 148, 481, 508, 336, 352, 386, 392, 155,  67, 254, 442, 180,
142,  68, 456, 439, 220, 380,   8, 228, 489, 401, 272, 260, 504, 226,
 16,  89, 279,  94, 354, 484, 382, 270, 347, 421, 412, 337, 224, 314,
120, 208, 339, 198, 344, 333,  57, 296, 388, 411, 214, 250, 387, 424,
262, 133, 207, 447, 486, 223, 496,  78,  14, 176,  72,  37, 192, 194,
 65, 299, 288, 177, 178,  63, 358, 507,  55, 246, 258, 327, 384, 502,
284, 436, 304,  71, 429, 319, 356, 213, 397, 469, 183, 499,  42, 100,
117, 190, 359, 217, 153,  21, 396, 196, 406, 135,  51, 251,  73, 346,
311, 114, 433, 490, 501, 282, 338, 394, 172,  18, 349, 276, 152, 189,
326, 462, 446, 167,  77,  84, 151, 318, 206,  24, 461, 163, 248, 393,
294, 199, 360, 286, 234, 145, 323, 257, 437, 130,  26, 154, 363, 200,
413,  70, 244, 238, 124,   6,  23, 403, 376, 105, 259, 104, 219, 422,
119,  20,  56, 430, 409, 128, 438, 482, 231,  45, 476, 308, 236,  91,
 38, 184, 218,  88, 101, 440,  69, 253,  46, 266, 488,  25, 355, 265,
  1, 357, 107, 426, 332, 165, 320, 317, 138, 329, 205, 261, 139, 292,
328, 370,  15,  22, 491, 465, 470,  17, 500,  30, 173, 137, 170, 256,
 76,  90, 418,  31,  64,  92, 475, 278, 146,  28,  58,  61,  32,  87,
269,  81, 125, 239, 444, 453,  99, 458, 414, 143, 459, 283, 390, 378,
233, 175, 495, 343, 345, 410, 389, 285]
kernel_rank_0325 = [506, 147, 175, 313, 306,  29, 229, 410, 494,  38, 277, 264, 343, 367,
170, 345,  62, 106, 132, 256, 278,  93, 249, 222, 124, 463, 241, 399,
320,  58, 434, 266, 162,  28, 438,  15, 261, 376, 500, 141, 436, 452,
295,  90, 303, 302,  92,  30,  85, 139, 193,   4,  60, 488, 414, 357,
    7, 173, 197, 216, 210,  46, 331,  25,  45, 231, 356,  91, 122, 453,
372, 239, 149, 275, 291, 196,   2, 467, 205, 493,  57,  49, 482, 200,
12, 184, 153, 104, 171,  76,  19, 317, 487,  27, 240, 433, 286, 144,
75, 437,  26,  24, 301,  34, 102,   6, 276, 274, 201, 110, 163, 483,
189, 297, 384, 167,   0, 308, 101,  83, 329, 178, 358,  23, 448, 140,
383, 481, 447, 326,  44, 362, 346, 177,  51, 403, 477, 338, 471, 342,
186, 282, 252, 360, 501, 137, 387, 430, 429, 304, 460, 461, 108,  21,
123,  14, 379, 480, 115,  37, 420, 451, 373, 294, 105, 350, 225, 164,
238, 366, 237, 212, 284,  59, 192, 263, 351, 339, 312, 396, 112,  84,
42, 363, 497,  40, 328,  94, 468, 154, 503,  77, 397, 421,  73, 217,
121, 450, 183,  89, 243, 391, 359, 380, 361, 161, 457, 486, 174, 455,
393, 148, 405, 172, 191, 412, 330, 257, 206, 232, 401, 474, 478, 250,
100, 228, 215,  68, 441, 316, 224, 151, 509, 508, 254, 273, 152,   5,
489, 469, 296, 262, 208,  48, 336,  43, 392,  95, 439, 386, 270,  67,
435, 202, 220, 134, 504, 368, 354, 289, 142, 307,   8, 443,  13, 182,
248, 377, 344, 353, 109, 456,  96, 382, 341, 290, 337, 255,  53, 155,
150, 118, 402, 223, 211, 245, 454, 314, 234, 188, 511,  74, 499, 187,
280, 180, 103, 466, 352, 213,  78,  10, 181, 235, 496, 364, 507,  97,
411, 449, 114, 130,  54, 311,  18, 159, 165, 135, 156, 258, 445, 298,
195, 120,  80, 133, 268, 198, 335, 473, 502, 348, 413, 408,  16, 492,
138,  72, 246, 385, 426, 299, 423, 319, 479, 333, 281, 199, 242, 126,
334, 406, 214, 128, 226, 374, 143,  63, 446, 428, 194, 324, 131, 251,
381,  36, 113, 207, 490, 157, 279, 117, 145, 388, 484, 398, 221, 253,
267, 424, 472, 168, 236, 394,  20,  98, 417,  79, 365, 505, 323, 293,
190, 369,  66, 318, 325, 416, 219,  47, 431, 347, 442, 415, 322,  35,
    1, 315, 425, 119, 409, 462, 166, 498, 288, 158,   9,  39, 465,  33,
259, 260, 204, 247,  86, 427, 218, 179,  71, 244, 185,  32,  70,  55,
    3, 300, 125, 327,  11, 419, 233, 422,  56, 107, 491, 375, 129,  69,
321, 404, 111, 160,  50,  81,  31, 203, 355, 340, 309,  65,  61, 349,
82,  52, 332, 444, 292, 272, 371, 176, 265, 271, 285,  64, 475, 209,
395, 370, 440, 169, 390, 418, 269,  22,  87, 432, 127, 146, 230, 400,
88,  41, 459, 389, 283, 116,  17, 378, 476, 470,  99, 458, 227, 287,
510, 305, 464, 495, 407, 136, 310, 485]
if __name__ == '__main__':
    # model_path = 'daily_resnet_results/ResNet_1_3/20250725_130630/checkpoints/20250325/20250402/20250325-81.pt'
    # model = ResNetv2(
    #         74,256)
    # model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model_path = '/mnt/c/gaochao/CODE/BCI/XZD4class/daily_resnet_results/ResNet_1_3/20250722_104158/checkpoints/20250325/20250402/20250325-58.pt'
    model_path = '/mnt/c/gaochao/CODE/BCI/XZD4class/daily_resnet_results/ResNet_1_3/20250725_165423/checkpoints/20250325/20250402/20250325-80.pt' # xzd repeat
    # model_path = '/mnt/c/gaochao/CODE/BCI/XZD4class/conv_filter_xzd/20250325-40.pt'
    model_path = '/mnt/c/gaochao/CODE/BCI/XZD4class/daily_resnet_results/ResNet_1_3/20250728_110218/checkpoints/20250325/20250402/20250325-60.pt'#xzd m1
    model_paths = [
        # 'daily_resnet_results/ResNet_1_3/20250729_170707/checkpoints/20250409/20250320/20250409-40.pt',
        # 'daily_resnet_results/ResNet_1_3/20250729_170707/checkpoints/20250320/20250320/20250320-81.pt',
        # 'daily_resnet_results/ResNet_1_3/20250729_170707/checkpoints/20250331/20250320/20250331-4.pt',
        # 'daily_resnet_results/ResNet_1_3/20250729_170707/checkpoints/20250327/20250320/20250327-9.pt',
        # 'daily_resnet_results/ResNet_1_3/20250729_170707/checkpoints/20250326/20250320/20250326-53.pt',
        # 'daily_resnet_results/ResNet_1_3/20250729_170707/checkpoints/20250320/20250320/20250320-4.pt',
        # 'daily_resnet_results/ResNet_1_3/20250729_154936/checkpoints/20250319/20250402/20250319-24.pt',
        # 'daily_resnet_results/ResNet_1_3/20250729_154936/checkpoints/20250402/20250401/20250402-80.pt',
        # 'daily_resnet_results/ResNet_1_3/20250729_150743/checkpoints/20250401/20250402/20250401-81.pt',
        # 'daily_resnet_results/ResNet_1_3/20250731_164037/checkpoints/20250325_20250326_20250327_20250329_20250331/20250402/20250325_20250326_20250327_20250329_20250331-54.pt'
        # 'daily_resnet_results/ResNet_1_3/20250804_142913/checkpoints/20250320/20250320/20250320-94.pt',
        # 'daily_resnet_results/ResNet_1_3/20250804_142913/checkpoints/20250409/20250320/20250409-41.pt'
        # --------32kernel--------
        # 'daily_resnet_results/ResNet_1_3/20250801_105119/checkpoints/20250325/20250326/20250325-24.pt',
        # 'daily_resnet_results/ResNet_1_3/20250801_105119/checkpoints/20250401/20250402/20250401-24.pt',
        # 'daily_resnet_results/ResNet_1_3/20250801_105119/checkpoints/20250402/20250401/20250402-24.pt',
        # '/mnt/c/gaochao/CODE/BCI/XZD4class/daily_resnet_results/ResNet_1_3/20250804_111907/checkpoints/20250320/20250320/20250320-94.pt',
        # '/mnt/c/gaochao/CODE/BCI/XZD4class/daily_resnet_results/ResNet_1_3/20250804_111907/checkpoints/20250409/20250320/20250409-94.pt',
        # 'daily_resnet_results/ResNet_1_3/20250805_144249/checkpoints/20250320/20250325/20250320-80.pt',
        # 'daily_resnet_results/ResNet_1_3/20250805_144249/checkpoints/20250409/20250325/20250409-80.pt',
        # 'daily_resnet_results/ResNet_1_3/20250805_144249/checkpoints/20250409_20250320/20250325/20250409_20250320-80.pt'

        # 'daily_resnet_results/ResNet_1_3/20250808_101422/checkpoints/20250325_20250326_20250327/20250401/20250325_20250326_20250327-90.pt',
        # 'daily_resnet_results/ResNet_1_3/20250807_170125/checkpoints/20250325_20250326_20250327/20250401/20250325_20250326_20250327-55.pt'
        # 'daily_resnet_results/ResNet_1_3/20250807_170125/checkpoints/20250325/20250401/20250325-45.pt',
        'daily_resnet_results/ResNet_1_3/20250808_101422/checkpoints/20250325/20250401/20250325-55.pt',
    ]
    model_names = [
        # 409,
        # 320,
        # 331,
        # 327,
        # 326,
        # 320,
        # 319,
        # 402,
        # 401,
        # '20250325_20250326_20250327_20250329_20250331'
        # 325,
        # 401,
        # 402,
        # 320,
        # 409,
        # 320,
        # 409,
        # '20250409_20250320',
        # '20250325_20250326_20250327_xzd',
        # '20250325_20250326_20250327_our',
        # '20250325_our',
        '20250325_xzd',
    ]
    global kernel_rank
    kernel_rank = kernel_rank_multi_day
    save_root = 'viz_conv_filter_sortedbygradcam_multi_day_rank'
    os.makedirs(save_root, exist_ok=True)
    global draw_filter_num
    draw_filter_num = 100
    
    for model_path, model_name in zip(model_paths,model_names):
        model = ResNetv2(
                in_channels=128,
                conv1_channels=512,
                kernel_size=3,
                n_classes=4,
                n_layers=1,
                first_kernel_size=25,
                drop_out=0.0,
                )
        model = ResNet(
                in_channels=128,
                out_channels=512,
                kernel_size=3,
                n_classes=4,
                n_layers=1,
                first_kernel_size=25,
                drop_out=0.0,
                )
        model.load_state_dict(torch.load(model_path, map_location='cpu'),strict=True)

        # ===== 使用示例 =====
        # 遍历模型中的所有1D卷积层
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, torch.nn.Conv1d)):  # PyTorch
                # visualize_conv1d_filters_by_sort(name, module.weight.data, save_dir=f"{save_root}/conv_filter_{model_name}_m1_512check_top100_no-norm")
                print(module.weight.data.mean().item(), module.weight.data.std().item())
                visualize_conv1d_filters(name, module.weight.data, save_dir=f"viz_conv_filter_32/conv_filter_{model_name}_m1s1_512")

