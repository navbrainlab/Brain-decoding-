import argparse
import os
from dataset import read_pkl, slide_window
from model.plot import plot_embedding
import matplotlib.pyplot as plt
import json
import umap
import numpy as np
from sklearn.decomposition import PCA
from dPCA import dPCA
import torch

def fit_and_plot_feature_scmd(feature_vector, test_dates, y_pred, path, title=None):
    # Project the features to a common space 
    if args.projection == 'umap':
        embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    elif args.projection == 'pca':
        embedding = PCA(n_components=2)
    embedding.fit(np.vstack(feature_vector))

    action_id = json.load(open(os.path.join(args.root, 'action_id.json')))[args.action_id]
    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'black', 'yellow', 'lime', 'maroon', 'navy', 'teal', 'fuchsia', 'aqua']
    marker = ['s', 'o', '^', 'x', '>', 'p', 'h', 'H', '+', '*', 'D', 'd', '1', '2', '3', '4']
    fontsize = 10

    # for j in range(len(test_dates)):
    #     embedding_point = embedding.transform(feature_vector[j])
    #     for i, label in enumerate(np.unique(y_pred[j])):
    #         if j == 0:
    #             plt.scatter(embedding_point[y_pred[j] == label, 0], embedding_point[y_pred[j] == label, 1],
    #                         c=color[i], marker=marker[j], s=3, label=f'{action_id[str(label)]}')
    #         else:
    #             if i == 0:
    #                 plt.scatter(embedding_point[y_pred[j] == label, 0], embedding_point[y_pred[j] == label, 1],
    #                             c=color[i], marker=marker[j], s=3, label=f'{test_dates[j]}')
    #             else:
    #                 plt.scatter(embedding_point[y_pred[j] == label, 0], embedding_point[y_pred[j] == label, 1],
    #                             c=color[i], marker=marker[j], s=3)

    # for j in range(len(test_dates)):
    #     embedding_point = embedding.transform(feature_vector[j])
    #     plt.scatter(embedding_point[:, 0], embedding_point[:, 1], c=color[j], s=3,
    #             label=test_dates[j])
    
    embedding_points = []
    for j in range(len(feature_vector)):
        embedding_points.append(embedding.transform(feature_vector[j]))

    for i, label in enumerate(np.unique(y_pred[j])):    
        plt.figure(figsize=(10, 10))
        for j in range(len(feature_vector)):
            embedding_point = embedding_points[j]
            
            plt.scatter(embedding_point[y_pred[j] == label, 0], embedding_point[y_pred[j] == label, 1],
                    c=color[j], marker=marker[i], s=5, label=f'{action_id[str(label)]} {test_dates[j]}')
        
        plt.xlabel('Latent 1', fontsize=fontsize)
        plt.ylabel('Latent 2', fontsize=fontsize)
        plt.legend(loc='lower right', fontsize=fontsize)
        # if not title:
        #     title = 'common_feature_space_%s_show_2_classes_only' % args.projection
        class_title = title + f'_{action_id[str(label)]}_{args.projection}'
        plt.title(class_title, fontsize=fontsize)
        plt.savefig(os.path.join(path, f'{class_title}.png'), bbox_inches='tight', dpi=300)
        plt.close()

def fit_and_plot_feature_mcsd(feature_vector, test_dates, y_pred, path, title=None):
    # Project the features to a common space 
    if args.projection == 'umap':
        embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    elif args.projection == 'pca':
        embedding = PCA(n_components=2)
    embedding.fit(np.vstack(feature_vector))

    action_id = json.load(open(os.path.join(args.root, 'action_id.json')))[args.action_id]
    # color = ['gold', 'indigo', 'crimson', 'coral', 'salmon', 'khaki', 'plum', 'orchid', 'tan', 'silver', 'tomato', 'chocolate', 'peru', 'darkgreen', 'midnightblue', 'dodgerblue', 'turquoise', 'hotpink', 'lightgreen', 'wheat']
    cmap = plt.get_cmap('tab10')

    marker = ['s', 'o', '^', 'x', '>', 'p', 'h', 'H', '+', '*', 'D', 'd', '1', '2', '3', '4']
    fontsize = 10
    
    for j in range(len(feature_vector)):
        embedding_point = embedding.transform(feature_vector[j])
        plt.figure(figsize=(10, 10))
        for i, label in enumerate(np.unique(y_pred[j])):
            plt.scatter(embedding_point[y_pred[j] == label, 0], embedding_point[y_pred[j] == label, 1],
                    c=[cmap(i)], s=5, label=f'{action_id[str(label)]} {test_dates[j]}')
        
        plt.xlabel('Latent 1', fontsize=fontsize)
        plt.ylabel('Latent 2', fontsize=fontsize)
        plt.legend(loc='lower right', fontsize=fontsize)
        class_title = title + f'_{test_dates[j]}_{args.projection}'
        plt.title(class_title, fontsize=fontsize)
        plt.savefig(os.path.join(path, f'{class_title}.png'), bbox_inches='tight', dpi=300)
        plt.close()

def fit_and_plot(data, data_labels, title, path):
    # print(data.shape)
    # print(data_labels.shape)
    action_id = json.load(open(os.path.join(args.root, 'action_id.json')))[args.action_id]
    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'black', 'yellow', 'lime', 'maroon', 'navy', 'teal', 'fuchsia', 'aqua']
    fontsize = 20    
    if args.projection == 'umap':
        embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    elif args.projection == 'pca':
        embedding = PCA(n_components=2)

    fig = plt.figure(figsize=(10, 10))
    # fig.suptitle(f'UMAP embedding val acc {acc} mahalanobis distance {distance}')
    embedding.fit(data)
    embedding_point = embedding.transform(data)    
    for i, label in enumerate(np.unique(data_labels)):
        plt.scatter(embedding_point[data_labels == label, 0], embedding_point[data_labels == label, 1], c=color[i], s=5,
                        label=action_id[str(label)])
    plt.title(title, fontsize=fontsize)
    plt.xlabel('Latent 1', fontsize=fontsize)
    plt.ylabel('Latent 2', fontsize=fontsize)
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(path, f'{title}.png'), bbox_inches='tight', dpi=300)
    np.save(os.path.join(path, f'{title}_embedding.npy'), embedding_point)  # Save as .npy
    plt.close(fig)

def fit_and_plot_dpca(trial_averaged_data, labels, title, path):

    S = len(np.unique(labels))
    dpca = dPCA.dPCA(labels='ts', n_components=2)
    dpca_data = dpca.fit_transform(trial_averaged_data)
    time = np.arange(args.window_size)
    plt.figure(figsize=(16,9))

    if args.simple_plot:
        for s in [0, S-1]:
            plt.plot(time,dpca_data['s'][0,:,s], label=labels[s])
        plt.title('1st condition component')
        plt.legend()
        plt.savefig(os.path.join(path, f'{title}_simple.png'), bbox_inches='tight', dpi=300)
        plt.close()
        return

    plt.subplot(331)
    for s in range(S):
        plt.plot(time,dpca_data['t'][0,:,s])
    plt.title('1st time component')
    plt.subplot(334)
    for s in range(S):
       plt.plot(time,dpca_data['t'][1,:,s]) 
    plt.title('2nd time component')
    plt.subplot(337)
    for s in range(S):
        plt.plot(dpca_data['t'][0,:,s], dpca_data['t'][1,:,s])
    plt.title('2nd vs 1st time components')

    plt.subplot(332)
    for s in range(S):
        plt.plot(time,dpca_data['s'][0,:,s])
    plt.title('1st condition component')
    plt.subplot(335)
    for s in range(S):
        plt.plot(time,dpca_data['s'][1,:,s])
    plt.title('2nd condition component')
    plt.subplot(338)
    for s in range(S):
        plt.plot(dpca_data['s'][0,:,s], dpca_data['s'][1,:,s])
    plt.title('2nd vs 1st condition components')

    plt.subplot(333)
    for s in range(S):
        plt.plot(time,dpca_data['ts'][0,:,s])
    plt.title('1st mixing component')
    plt.subplot(336)
    for s in range(S):
        plt.plot(time,dpca_data['ts'][1,:,s])
    plt.title('2nd mixing component')
    plt.subplot(339)
    for s in range(S):
        plt.plot(dpca_data['ts'][0,:,s], dpca_data['ts'][1,:,s], label=labels[s])
    plt.title('2nd vs 1st mixing components')
    plt.legend()

    plt.suptitle(title)
    plt.savefig(os.path.join(path, f'{title}.png'), bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()

def fit_and_plot_multiday_dpca(trial_averaged_data, labels, date_tag, title, path):
    dpca = dPCA.dPCA(labels='tcd', n_components=2)
    dpca_data = dpca.fit_transform(trial_averaged_data)
    time = np.arange(args.window_size)
    label_number = len(np.unique(labels))
    date_number = len(np.unique(date_tag))

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'black', 'yellow', 'lime', 'maroon', 'navy', 'teal', 'fuchsia', 'aqua']
    linestyles = ['-', '--', ':', '-.']

    plt.figure(figsize=(16,9))
    if args.simple_plot:
        for l in [0, label_number-1]:
            for d in [0, date_number-1]:
                plt.plot(time, dpca_data['c'][0, :, l, d], c=colors[l], linestyle=linestyles[d])
        plt.title('1st condition component')
        plt.savefig(os.path.join(path, f'{title}_simple.png'), bbox_inches='tight', dpi=300)
        plt.close()
        return
        
    plt.subplot(2, 3, 1)
    for l in range(label_number):
        for d in range(date_number):
            plt.plot(time, dpca_data['t'][0, :, l, d], c=colors[l], linestyle=linestyles[d])
    plt.title('1st time component')
    plt.subplot(2, 3, 4)
    for l in range(label_number):
        for d in range(date_number):
            plt.plot(time, dpca_data['t'][1, :, l, d], c=colors[l], linestyle=linestyles[d])
    plt.title('2nd time component')

    plt.subplot(2, 3, 2)
    for l in range(label_number):
        for d in range(date_number):
            plt.plot(time, dpca_data['c'][0, :, l, d], c=colors[l], linestyle=linestyles[d])
    plt.title('1st condition component')
    plt.subplot(2, 3, 5)
    for l in range(label_number):
        for d in range(date_number):
            plt.plot(time, dpca_data['c'][1, :, l, d], c=colors[l], linestyle=linestyles[d])
    plt.title('2nd condition component')

    plt.subplot(2, 3, 3)
    for l in range(label_number):
        for d in range(date_number):
            plt.plot(time, dpca_data['d'][0, :, l, d], c=colors[l], linestyle=linestyles[d])
    plt.title('1st day component')
    plt.subplot(2, 3, 6)
    for l in range(label_number):
        for d in range(date_number):
            plt.plot(time, dpca_data['d'][1, :, l, d], c=colors[l], linestyle=linestyles[d])
    plt.title('2nd day component')

    plt.suptitle(title)
    plt.savefig(os.path.join(path, f'{title}.png'), bbox_inches='tight', dpi=300)
    plt.close()



def original_data_analysis(args):

    root_path = os.path.join(args.save_root_path, 'win_%d_stride_%d_start_from_%d' % (args.window_size, args.window_stride, args.start_from))
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    
    if args.projection == 'dpca':
        dpca_multiday_data = []
        dpca_date_tag = []

    else:
        allocate_length = 1000000
        multiday_labels = np.zeros(allocate_length, dtype=np.int8)
        multiday_data = np.zeros((allocate_length, args.window_size*args.input_dim), dtype=np.float32)
        date_tag = np.zeros(allocate_length, dtype='<U8')
        data_number = 0

    for date in args.dates:
        date_path = os.path.join(args.data_path, date)
        data, data_labels, data_session = read_pkl(date_path, session=16)
        data, data_labels = slide_window(data, list(data_labels), windows_size=args.window_size, step=args.window_stride, start_from=args.start_from)
        print(data.shape)

        if args.projection == 'dpca':
            unique_labels = np.unique(data_labels)
            S = len(unique_labels)
            results = []
            labels = []

            for label in unique_labels:
                data_for_label = data[data_labels == label]
                mean_data_for_label = np.mean(data_for_label, axis=0)
                results.append(mean_data_for_label)
                labels.append(label)

            trial_averaged_data = np.array(results).transpose(1,2,0) # Channels x Time x Conditions
            fit_and_plot_dpca(trial_averaged_data, labels, title='%s_%s' % (date, args.projection), path=root_path)
            if date == '20250319' or date == '20250409':
                dpca_multiday_data.append(trial_averaged_data)
                dpca_date_tag.append(date)
        else:
            data = data.reshape(data.shape[0], -1)
            multiday_data[data_number:data_number+data.shape[0],:] = data.copy()
            multiday_labels[data_number:data_number+data.shape[0]] = data_labels.astype(np.int8).copy()
            date_tag[data_number:data_number+data.shape[0]] = np.array([date]*data.shape[0])
            data_number += data.shape[0]
            fit_and_plot(data, data_labels, title='%s_%s' % (date, args.projection), path=root_path)
        # break

    if args.projection == 'dpca':
        dpca_multiday_data = np.array(dpca_multiday_data).transpose(1,2,3,0) # Channels x Time x Conditions x Days
        dpca_date_tag = np.array(dpca_date_tag)
        fit_and_plot_multiday_dpca(dpca_multiday_data, labels, dpca_date_tag, title='all_date_%s' % args.projection, path=root_path)

    else:
        multiday_data = multiday_data[:data_number,:]
        multiday_labels = multiday_labels[:data_number]
        date_tag = date_tag[:data_number]
        # fit_and_plot(multiday_data, date_tag, title='all_date_%s' % args.projection, path=root_path)
        fit_and_plot(multiday_data, multiday_labels, title='all_label_%s' % args.projection, path=root_path)



def feature_analysis(args):
    train_checkpoint_path = os.path.join(args.checkpoint_path, args.train_date)
    test_dates = os.listdir(train_checkpoint_path)
    feature_vector = []
    y_pred = []
    y_test = [] # Label


    # Get feature vectors
    for test_date in test_dates:
        test_checkpoint_path = os.path.join(train_checkpoint_path, test_date)
        metrics_paths = os.listdir(test_checkpoint_path)
        metrics_paths = [metrics_paths[i] for i in range(len(metrics_paths)) if 'metrics' in metrics_paths[i]]
        max_accuracy = float('-inf')
        best_metrics = None        
        # Choose the feature at the epoch reaching the highest accuracy
        for metrics_path in metrics_paths:
            metrics = torch.load(os.path.join(test_checkpoint_path, metrics_path))
            if metrics['accuracy'] > max_accuracy:
                max_accuracy = metrics['accuracy']
                best_metrics = metrics
                
        feature_vector.append(best_metrics['feature_vector'])
        y_pred.append(best_metrics['y_pred'])
        y_test.append(best_metrics['y_test'])
        if test_date == args.train_date:
            train_feature_vector = best_metrics['feature_vector']

    path = os.path.join(args.save_root_path, 'feature_analysis')
    os.makedirs(path, exist_ok=True)
    fit_and_plot_feature_scmd(feature_vector, test_dates, y_pred, path)

    # Project the features to the space of the training feature

def feature_analysis2(args):

    # Get features
    test_dates = args.dates
    save_path = os.path.join(args.save_root_path, 'feature_analysis2_1', '%strain' % args.train_date)
    os.makedirs(save_path, exist_ok=True)
    feature1 = []
    feature2 = []
    feature3 = []
    y_pred = []
    y_test = []

    for test_date in test_dates:
        print('loading metrics for %s' % test_date)
        metrics_path = os.path.join(args.checkpoint_path, 'metrics_%s→%s-50.pth' % (args.train_date, test_date))
        metrics = torch.load(os.path.join(args.checkpoint_path, metrics_path))
        feature1.append(metrics['feature1'])
        feature2.append(metrics['feature2'])
        feature3.append(metrics['feature3'])
        y_pred.append(metrics['y_pred'])
        y_test.append(metrics['y_test'])
        # fit_and_plot(metrics['feature1'], metrics['y_test'], '%s_feature1_%s' % (test_date,args.projection), save_path)
        # fit_and_plot(metrics['feature2'], metrics['y_test'], '%s_feature2_%s' % (test_date,args.projection), save_path)
        # fit_and_plot(metrics['feature3'], metrics['y_test'], '%s_feature3_%s' % (test_date,args.projection), save_path)
    del metrics
    # print('fitting and plotting feature 1')
    # fit_and_plot_feature_scmd(feature1, test_dates, y_test, save_path, title='feature1_gt')
    # fit_and_plot_feature_mcsd(feature1, test_dates, y_test, save_path, title='feature1_gt')
    # del feature1
    print('fitting and plotting feature 2')
    fit_and_plot_feature_scmd(feature2, test_dates, y_test, save_path, title='feature2_gt')
    fit_and_plot_feature_mcsd(feature2, test_dates, y_test, save_path, title='feature2_gt')
    del feature2
    print('fitting and plotting feature 3')
    fit_and_plot_feature_scmd(feature3, test_dates, y_test, save_path, title='feature3_gt')
    fit_and_plot_feature_mcsd(feature3, test_dates, y_test, save_path, title='feature3_gt')



def parse_args():
    """参数配置"""
    parser = argparse.ArgumentParser(description='ResNet1D训练系统')

    # 数据参数
    parser.add_argument('--root', type=str, default='/media/ubuntu/Storage/ecog_data',
                        help='脑电数据根目录')
    parser.add_argument('--data_path', type=str,
                        default='/media/ubuntu/Storage/ecog_data/preprocessed',
                        help='脑电数据根目录')
    parser.add_argument('--save_root_path', type=str, default='/home/ubuntu/ecog_proj/data_analysis')
    parser.add_argument('--dates', nargs='+',
                        default=[
                                #  '20250319',
                                 '20250320',
                                #  '20250321',
                                #  '20250323',
                                #  '20250324',
                                #  '20250325',
                                #  '20250326',
                                #  '20250327',
                                #  '20250329',
                                #  '20250331',
                                #  '20250401',
                                #  '20250402',
                                 '20250409'
                                 ],
                        # default=[
                                #  '20250325',
                                #  '20250326',
                                #  '20250327',
                                #  '20250329',
                                #  '20250331',
                                #  '20250401',
                                #  '20250402'
                                #  ],                        
                        help='实验日期列表，按时间顺序排列')
    
    parser.add_argument('--checkpoint_path', type=str, help='Used when projection is resnet_feature',
                        default='/media/ubuntu/Storage/ecog_data/feature2_1')
                        # default='/media/ubuntu/Storage/ecog_data/daily_resnet_results/ResNet_1_3/20250724_151911/checkpoints')
    parser.add_argument('--train_date', type=str, default='20250409')
    # parser.add_argument('--train_date', type=str, default='20250325_20250326_20250327_20250329_20250331')



    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=5)


    parser.add_argument('--window_size', type=int, default=768)
    parser.add_argument('--window_stride', type=int, default=1024)
    parser.add_argument('--start_from', type=int, default=0)

    # 系统参数
    parser.add_argument('--action_id', type=str, default='tiantan', help='Use today\'s date for validation')
    parser.add_argument('--today', action='store_true', help='Use today\'s date for validation')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_root', type=str,
                        # default='contra_resnet_results_compare'
                        default='daily_resnet_results'
                        # default='tiantan_s01'
                        )

    # 其他参数
    parser.add_argument('--seed', type=int, default=1024, help='Random seed for reproducibility')
    parser.add_argument('--projection', type=str, default='umap', choices=['umap', 'pca', 'dpca'])
    parser.add_argument('--data_type', type=str, default='feature', choices=['feature', 'original'])
    parser.add_argument('--simple_plot', type=bool, default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.data_type == 'feature':
        feature_analysis2(args)
        # feature_analysis(args)
    else:
        original_data_analysis(args)
