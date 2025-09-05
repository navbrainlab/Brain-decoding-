import torch
from pi_vae_pytorch import PiVAE
import argparse
import os
from dataset import read_pkl, slide_window
import matplotlib.pyplot as plt
import json
import numpy as np

def date_to_int(date_str):
    """
    将日期字符串转换为基于'20250319'为起始点的整数值
    
    参数:
        date_str (str): 日期字符串，格式为'YYYYMMDD'
        
    返回:
        int: 基于'20250319'为0的整数值
        
    异常:
        ValueError: 如果输入的日期格式不正确
    """
    from datetime import datetime
    
    try:
        # 基准日期
        base_date = datetime(2025, 3, 19)
        
        # 将输入字符串转换为datetime对象
        input_date = datetime.strptime(date_str, '%Y%m%d')
        
        # 计算天数差
        delta = input_date - base_date
        
        # 返回天数差作为整数
        return delta.days
    except ValueError as e:
        raise ValueError(f"日期格式不正确，应为'YYYYMMDD'格式: {date_str}") from e


def int_to_date(date_int):
    """
    将基于'20250319'为0的整数值转换回日期字符串
    
    参数:
        date_int (int): 基于'20250319'为0的整数值
        
    返回:
        str: 格式为'YYYYMMDD'的日期字符串
        
    示例:
        >>> int_to_date(0)
        '20250319'
        >>> int_to_date(12)
        '20250331'
        >>> int_to_date(13)
        '20250401'
    """
    from datetime import datetime, timedelta
    
    # 基准日期
    base_date = datetime(2025, 3, 19)
    
    # 计算目标日期
    target_date = base_date + timedelta(days=date_int)
    
    # 格式化为字符串
    return target_date.strftime('%Y%m%d')



def main(ii):
    action_id = json.load(open(os.path.join(args.root, 'action_id.json')))[args.action_id]
    all_data = []
    all_data_labels = []
    all_data_dates = []
    color = ['red', 'green', 'blue', 'black', 'purple', 'orange', 'cyan', 'magenta', 'lime', 'olive', 'navy', 'teal', 'maroon', 'indigo', 'gold', 'pink', 'gray', 'silver']

    for date in args.dates:
        # os.makedirs(os.path.join(args.save_path, date), exist_ok=True)
        date_path = os.path.join(args.data_path, date)
        data, data_labels, data_session = read_pkl(date_path, session=16)
        data, data_labels = slide_window(data, list(data_labels), windows_size=args.window_size, step=args.window_stride, start_from=args.start_from)
        all_data.append(torch.tensor(data, dtype=torch.float32))
        all_data_labels.append(torch.tensor(data_labels, dtype=torch.int))
        all_data_dates.append(torch.tensor([date_to_int(date)] * len(data_labels)))

    all_data = torch.cat(all_data, dim=0)
    all_data_labels = torch.cat(all_data_labels, dim=0)
    all_data_dates = torch.cat(all_data_dates, dim=0)
    all_data = all_data.cuda()
    all_data_labels = all_data_labels.cuda()
    all_data_dates = all_data_dates.cuda()

    print(all_data.device)
    print(all_data_labels.shape)
    print(all_data_dates.shape)
    n_sample, n_channel, n_time_point = all_data.shape

    model = PiVAE(
        x_dim = n_time_point*n_channel,
        u_dim = 15,
        z_dim = 2,
        discrete_labels=True,
        encoder_n_hidden_layers = args.encoder_n_hidden_layers,
        encoder_hidden_layer_dim = args.encoder_hidden_layer_dim,
        decoder_n_gin_blocks = args.decoder_n_gin_blocks,
        decoder_gin_block_depth = args.decoder_gin_block_depth
    )
    model.cuda()
    print(model)
    # outputs = model(all_data.reshape(n_sample, -1), all_data_dates) # dict
    outputs = model(all_data.reshape(n_sample, -1), all_data_labels) # dict
    z_sample = outputs['posterior_z_sample'].cpu().detach().numpy()
    plt.figure(figsize=(16, 10))
    ax1 = plt.subplot(121)
    # ax = plt.subplot(111, projection='3d')
    all_data_dates = all_data_dates.cpu().detach().numpy()
    for i, label in enumerate(np.unique(all_data_dates)):
        idx = all_data_dates == label
        ax1.scatter(z_sample[idx, 0], z_sample[idx, 1], c=color[i], label=int_to_date(int(label)), alpha=0.5)
    ax1.legend()
    
    ax2 = plt.subplot(122)
    all_data_labels = all_data_labels.cpu().detach().numpy()
    for i, label in enumerate(np.unique(all_data_labels)):
        idx = all_data_labels == label
        ax2.scatter(z_sample[idx, 0], z_sample[idx, 1], c=color[i], label=action_id[str(label)], alpha=0.5)
        # ax2.scatter(z_sample[:, 0], z_sample[:, 1], z_sample[:, 2], c=all_data_labels)        
    ax2.legend()
    # plt.show()
    figpath = f'{args.save_path}/separating_movements/enlay{args.encoder_n_hidden_layers}_endim{args.encoder_hidden_layer_dim}_deblk{args.decoder_n_gin_blocks}_dedp{args.decoder_gin_block_depth}'
    os.makedirs(figpath, exist_ok=True)
    plt.savefig(os.path.join(figpath, f'posterior_z_sample_multiday{ii}.png'))
    plt.close()
    del model, all_data_dates, all_data_labels, z_sample, outputs


def parse_args():
    """参数配置"""
    parser = argparse.ArgumentParser(description='ResNet1D训练系统')

    # 数据参数
    parser.add_argument('--data_path', type=str,
                        default='/media/ubuntu/Storage/ecog_data/preprocessed_removed_wrong_trials',
                        help='脑电数据根目录')
    parser.add_argument('--root', type=str, default='/media/ubuntu/Storage/ecog_data',
                        help='脑电数据根目录')    
    parser.add_argument('--save_path', type=str, default='/home/ubuntu/ecog_proj/piVAE_result')
    parser.add_argument('--dates', nargs='+',
                        default=[
                                 '20250319',
                                #  '20250320',
                                #  '20250321',
                                #  '20250323',
                                #  '20250324',
                                #  '20250325',
                                #  '20250326',
                                #  '20250327',
                                #  '20250329',
                                #  '20250331',
                                #  '20250401',
                                 '20250402',
                                #  '20250409'
                                 ],                   
                        help='实验日期列表，按时间顺序排列')

    parser.add_argument('--window_size', type=int, default=256)
    parser.add_argument('--window_stride', type=int, default=32)
    parser.add_argument('--start_from', type=int, default=0)
    parser.add_argument('--action_id', type=str, default='tiantan', help='Use today\'s date for validation')

    parser.add_argument('--encoder_n_hidden_layers', type=int, default=2)
    parser.add_argument('--encoder_hidden_layer_dim', type=int, default=128)

    parser.add_argument('--decoder_n_gin_blocks', type=int, default=2)
    parser.add_argument('--decoder_gin_block_depth', type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    for ii in range(1,6):
        main(ii)