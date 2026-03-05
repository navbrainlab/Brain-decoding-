import argparse
import os
from dataset import read_pkl, slide_window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def SVD():

    # Conduct SVD
    for date_compositions in args.base_dates:
        all_data = None
        for date in date_compositions:
            date_path = os.path.join(args.data_path, date)
            data, data_labels, data_session = read_pkl(date_path, session=16)
            if all_data is None:
                all_data = np.hstack(data)
            else:
                all_data = np.concatenate((all_data, np.hstack(data)), axis=1)
        print(all_data.shape)
        all_data -= np.mean(all_data, axis=1, keepdims=True)
        U, S, Vt = np.linalg.svd(all_data, full_matrices=False)
        plt.scatter(range(len(S)), S)
        dates = '_'.join(date_compositions)
        plt.title(f'Singular values {dates}')
        # plt.show()
        os.makedirs(args.save_path, exist_ok=True)
        plt.savefig(os.path.join(args.save_path, f'svd_{dates}.png'))
        plt.close()
        np.save(os.path.join(args.save_path, f'U_{dates}.npy'), U)                      



def visulization():

    electrode_mapping_df = pd.read_excel(os.path.join(args.data_path, '..', 'electrode_mapping.xlsx'), header=None)
    electrode_mapping = electrode_mapping_df.values
    electrode_mapping[0,0] = 129
    electrode_mapping[0,-1] = 130
    electrode_mapping[-1,0] = 131
    electrode_mapping[-1,-1] = 132
    electrode_mapping = electrode_mapping.astype(int)
    electrode_mapping -= 1

    for date_compositions in args.base_dates:
        dates = '_'.join(date_compositions)
        U = np.load(os.path.join(args.save_path, f'U_{dates}.npy'))
        os.makedirs(os.path.join(args.save_path, f'U_{dates}'), exist_ok=True)
        for i in range(args.svd_rank):
            U_reshaped = np.concatenate((U[:, i], np.zeros(4)))
            U_reshaped = U_reshaped[electrode_mapping.reshape(-1)].reshape(11,12)
            plt.figure(figsize=(10, 10))
            plt.imshow(U_reshaped, cmap='gray')
            plt.title(f'U_rank{i+1}_{dates}')
            # Text at the bottom of the image
            # plt.text(0, 20, 'Each grid is colored by the weight of the rank i singular vector for the channel at the grid', ha='left', va='top')
            cbar = plt.colorbar(aspect=30)
            cbar.set_label(f'Weight of the rank {i+1} singular vector for the channel at each grid')
            plt.clim(-0.2, 0.2)
            plt.yticks(range(11), range(1,12))
            plt.xticks(range(12), range(1,13))            
            plt.savefig(os.path.join(args.save_path, f'U_{dates}', f'U_rank{i+1}_{dates}.jpg'))
            print(f'Figure saved')



        # # Visulize matrix U.T
        # # test[i] = svd['U'][:, :svd['rank']].T @ test[i]
        # # U.T[m,n] is the weight of the m-th singular vector for the n-th channel
        # plt.figure(figsize=(20, 10))
        # plt.imshow(U[:, :args.svd_rank].T, cmap='gray')
        # plt.title(f'U.T_{dates}')
        # # Text at the bottom of the image
        # plt.text(0, 50, 'U.T[m,n] is the weight of the m-th singular vector for the n-th channel', ha='left', va='top')
        # plt.colorbar()
        # plt.show()



def parse_args():
    """参数配置"""
    parser = argparse.ArgumentParser(description='ResNet1D训练系统')

    # 数据参数
    parser.add_argument('--data_path', type=str,
                        default='/media/ubuntu/Storage/ecog_data/preprocessed',
                        help='脑电数据根目录')
    parser.add_argument('--save_path', type=str, default='/home/ubuntu/ecog_proj/svd')
    parser.add_argument('--base_dates', nargs='+',
                        default=[
                                #  ['20250319'],
                                #  ['20250320'],
                                #  ['20250321'],
                                #  ['20250323'],
                                #  ['20250324'],
                                 ['20250325'],
                                #  ['20250326'],
                                #  ['20250327'],
                                #  ['20250329'],
                                #  ['20250331'],
                                 ['20250401'],
                                #  ['20250402'],
                                #  ['20250409'],
                                 ['20250325', '20250326', '20250327'],
                                #  ['20250325', '20250326', '20250327', '20250329', '20250331'],
                                 ],                   
                        help='Put the data from each sublist together to conduct SVD, and conduct SVD once for each sublist')
    parser.add_argument('--svd_rank', type=int, default=64, help='The rank of the SVD matrix')
    

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # SVD()
    visulization()