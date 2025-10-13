
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import calinski_harabasz_score
import math
import re
import os
from statistics import mean

def calc_psth(dat,tid,speed,tid0):
    act0 = dat[:,:] # 下标1对应400个时间点，其中282-60到282+60对应手臂运动的time window (282 corresponds to GO signal)
    psth = []
    for jj in range(8):
        idx = np.where([(tid0==jj)&(tid==speed)])#
        psth.append(np.nanmean(act0[idx[1]],axis=0))#沿0轴计算算数平均值，且跳过其中的nan值
    psth = np.array(psth)
    return psth

def detect_MO(data):
    window_size = 10
    MO_list = []
    for trail in range(len(data)):
        R = []
        for t in range(len(data[trail])):
            R.append(math.sqrt(np.sum(np.square(data[trail][t]))))
        series = pd.Series(R)
        sma = series.rolling(window=window_size).mean()
        diff = sma.diff()
        second_diff = diff.diff()
        potential_inflexion_points = second_diff[(second_diff > 0) & (second_diff.shift() < 0)].index #一阶导数
        if potential_inflexion_points.empty:
            MO_list.append(None)
        else:
            MO_list.append(potential_inflexion_points[-1])
    return round(mean(MO_list)), MO_list

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

def ChooseMethod(w,link,aff,vmin,vmax,cmap):
    plt.figure(figsize=[12,11])
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None,linkage=link, affinity=aff)
    model = model.fit(w)
    R=plot_dendrogram(model, truncate_mode="level", p=1,no_plot=True)
    boundaries_y = []
    for item in R['ivl']:
        if '(' in item:
            boundaries_y.append(int(re.findall(r'\d+', item)[0]))
        else:
            boundaries_y.append(1)
    idx=cluster(w,link,aff)
    X_sort=w[idx][:,idx]
    ax0=plt.subplot(1,1,1)
    plt.title(f'{link}+{aff}')
    sns.heatmap(X_sort, ax=ax0, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'location': 'left', 'orientation': 'vertical'})
    boundary = np.zeros(len(boundaries_y)+1)
    for j in range(len(boundaries_y)+1):
        boundary[j] = np.sum(boundaries_y[:j])
        if j==0 or j== len(boundaries_y):
            continue
        ax0.axhline(y=boundary[j], color='white', linestyle='--')
        ax0.axvline(x=boundary[j], color='white', linestyle='--')
    score=EvaluateCluster(w,boundary.astype(int),idx)
    print(link,aff,score)

def PlotActivityHeatmap(matrix,area,speed,dir,bound,idx):
    ax_list=np.linspace(0,399,21,dtype=int)
    ax_list=list(ax_list)
    speed_num=len(speed)
    dir_num=len(dir)
    fig=plt.figure(figsize=(15, 24))
    for i in range(speed_num):#speed cond
        axs=plt.subplot(1,speed_num,i+1)
        axs.xaxis.set_ticks_position('none')  # 不显示x轴的刻度
        axs.yaxis.set_ticks_position('none')  # 不显示y轴的刻度
        axs.set_xticks([])  # 移除x轴的刻度标记
        axs.set_yticks([])
        axs.spines['bottom'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.spines['left'].set_visible(False)
        axs.text(0.5, 1.03, f'speed={speed[i]}', fontsize=12,horizontalalignment='center',\
                  verticalalignment='center', transform=plt.gca().transAxes)
        # axs.set_title(f'speed={speed[i]}',position=(0.5,1.02), verticalalignment='bottom')
        gs00 = GridSpecFromSubplotSpec(dir_num,1, subplot_spec=axs, wspace=0.1, hspace=0.1)
        for j in range(dir_num):#dir cond
            ax1 = fig.add_subplot(gs00[j,0])
            ax1=sns.heatmap(matrix[i,j].T,ax=ax1,cmap='jet',cbar=False)
            ax1.text(0.5, 1.01,f'dir={dir[j]*45}°', fontsize=12,horizontalalignment='center',\
                  verticalalignment='bottom', transform=plt.gca().transAxes)
            ax1.set_yticks(ax_list,idx[ax_list])
            ax1.set_xticks([0,58],['TO','MO'],rotation=0)
            ax1.set_ylabel(f'{area}_neuron')
            for k in range(len(bound)):
                ax1.axhline(bound[k],linestyle='--',color='white')
            ax1.invert_yaxis()
            if (i==speed_num-1) and (j==dir_num-1):
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
                fig.colorbar(ax1.collections[0], cax=cbar_ax)
        ax1.set_xlabel('time')
    return fig

def cluster(matrix,link,aff):
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None,linkage=link,metric=aff)
    model = model.fit(matrix)
    idx=[]
    R=plot_dendrogram(model, no_plot=True)
    idx=np.array(R['leaves'])
    return idx

def GetBound(matrix,link,aff,threshold):
    Z = linkage(matrix, link,aff)
    clusters = fcluster(Z, t=threshold, criterion='distance')
    boundaries_y=np.zeros(len(set(clusters)))
    for i in range(len(set(clusters))):
        boundaries_y[i]=np.sum(clusters==i+1)
    return boundaries_y

def EvaluateCluster(matrix,bound,idx):
    kde=np.zeros(len(bound)-1)
    for i in range(len(bound)-1):
        if i==0:
            cluster=matrix[idx[bound[i]:bound[i+1]],:]
            kind=[f'cluster{i+1}']*cluster.shape[0]
            continue
        cluster_i=matrix[idx[bound[i]:bound[i+1]],:]
        cluster=np.concatenate([cluster,cluster_i])
        
        kind_i=[f'cluster{i+1}']*cluster_i.shape[0]
        kind=kind+kind_i
        
    score=calinski_harabasz_score(cluster, kind)
    return score

def PlotM1Cluster(w_out,idx_m1,bound_m1,ylabel):
    plt.rcParams.update({'font.size': 16})  
    ax_list=list(np.linspace(0,399,21,dtype=int))
    boundary = np.zeros(len(bound_m1)+1)
    fig = plt.figure(figsize=(2, 6))
    X_sort=w_out[idx_m1,:]
    ax=sns.heatmap(X_sort, cmap='brg',cbar_kws={'location': 'right', 'orientation': 'vertical'})
    plt.yticks(ax_list, idx_m1[ax_list],fontsize=16)
    plt.xticks([0.5,1.5], ['x','y'],rotation=0,fontsize=16)
    # ax3.invert_yaxis()
    for i in range(len(bound_m1)+1):
        boundary[i] = np.sum(bound_m1[:i])
        if i==0 or i== len(bound_m1):
            continue
        plt.axhline(y=boundary[i], color='white', linestyle='--')
    plt.ylabel(ylabel,fontsize=16)

def PlotInput2PMdCluster(w,idx,bound,ylabel,vmin, vmax):
    plt.rcParams.update({'font.size': 16})
    ax_list=list(np.linspace(0,399,21,dtype=int))
    boundary = np.zeros(len(bound)+1)
    fig = plt.figure(figsize=(3, 6))
    X_sort=w[idx, :]
    ax=sns.heatmap(X_sort, cmap='brg', cbar_kws={'location': 'right', 'orientation': 'vertical'}, vmin=vmin, vmax=vmax)
    plt.yticks(ax_list, idx[ax_list],fontsize=16)
    plt.xticks([0.5,1.5,2.5], ['x','y','MO Signal'],rotation=0,fontsize=12)
    for i in range(len(bound)+1):
        boundary[i] = np.sum(bound[:i])
        if i==0 or i== len(bound):
            continue
        plt.axhline(y=boundary[i], color='white', linestyle='--')
    plt.ylabel(ylabel,fontsize=16)
    
def heatmap( X_sort, bound,idx_x, idx_y, vmin, vmax, xlabel, ylabel, title, cmap='jet'):
    # plt.title(title)
    plt.figure(figsize=[6,5])
    sns.heatmap(X_sort[idx_y][:,idx_x],  cmap=cmap, vmin=vmin, vmax=vmax)
    ax_list=list(np.linspace(0,399,21,dtype=int))
    plt.xticks(ax_list, idx_x[ax_list],fontsize=16)
    plt.yticks(ax_list, idx_y[ax_list],fontsize=16)
    plt.xlabel(xlabel,fontsize=16)
    plt.ylabel(ylabel,fontsize=16)
    boundary = 0
    idx_bound=np.zeros(len(bound)+1)
    # print(boundaries_y)
    for i in range(len(bound)-1):
        boundary += bound[i]
        idx_bound[i+1]=boundary
        # print(boundary)
        plt.axhline(y=boundary, color='white', linestyle='--')
        plt.axvline(x=boundary, color='white', linestyle='--')

    idx_bound[-1]=400

def PlotPmdCluster(w_p2m,idx_pmd,idx_m1,bound_pmd,bound_m1,link,aff,vmin,vmax,threshold,xlabel,ylabel,cmap,level):
    fig = plt.figure(figsize=(7, 6))
    ax = plt.subplot(1,1,1)
    ax.xaxis.set_ticks_position('none')  # 不显示x轴的刻度
    ax.yaxis.set_ticks_position('none')  # 不显示y轴的刻度
    ax.set_xticks([])  # 移除x轴的刻度标记
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # plt.title(title)
    gs00 = GridSpecFromSubplotSpec(1, 2, subplot_spec=ax.get_subplotspec(), wspace=0.02, hspace=0.02, width_ratios=[3., 0.5])
    ax0 = fig.add_subplot(gs00[0, 0])
    ax1 = fig.add_subplot(gs00[0, 1])
    # ax2 = fig.add_subplot(gs00[0, 0])
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None,linkage=link, metric=aff)
    model = model.fit(w_p2m)
    plot_dendrogram(model, truncate_mode="level", p=level, \
                    no_labels=True, orientation='right', ax=ax1,color_threshold=threshold)
    # ax1.set_xlim([0.5,2])
    ax1.xaxis.set_ticks_position('none')  # 不显示x轴的刻度
    ax1.yaxis.set_ticks_position('none')  # 不显示y轴的刻度
    ax1.set_xticks([])  # 移除x轴的刻度标记
    ax1.set_yticks([])
    ax1.invert_yaxis()
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax_list=list(np.linspace(0,399,21,dtype=int))
    X_sort = w_p2m[idx_pmd][:,idx_m1]
    ax0=sns.heatmap(X_sort, ax=ax0, cmap=cmap, vmin=vmin, vmax=vmax,cbar=False)
    cbar_ax = fig.add_axes([-0.02, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar=fig.colorbar(ax0.collections[0], cax=cbar_ax)
    cbar.ax.yaxis.set_ticks_position('left')  # 将刻度移动到左边
    cbar.ax.yaxis.set_label_position('left')
    ax0.set_xticks(ax_list, idx_m1[ax_list],fontsize=16)
    ax0.set_yticks(ax_list, idx_pmd[ax_list],fontsize=16)
    # ax0.invert_yaxis()
    ax0.set_xlabel(xlabel,fontsize=16)
    ax0.set_ylabel(ylabel,fontsize=16)
    boundary = 0
    idx_bound=np.zeros(len(bound_pmd)+1)
    # print(boundaries_y)
    for i in range(len(bound_pmd)-1):
        boundary += bound_pmd[i]
        idx_bound[i+1]=boundary
        # print(boundary)
        ax0.axhline(y=boundary, color='white', linestyle='--')
    idx_bound[-1]=400
    boundary = 0
    for i in range(len(bound_m1)-1):
        boundary += bound_m1[i]  
        # print(boundary)
        ax0.axvline(x=boundary, color='white', linestyle='--')
    return idx_bound.astype(int),fig

def PlotOutput(matrix,idx,bound,colors,markers,activity=np.array([])):#activity!=None means we use their activity as their color
    fig=plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 18})  
    cmap = plt.get_cmap('jet') 
    ax=plt.subplot(1,1,1)
    if activity.size==0:
        for i in range(len(bound)-1):
            class_data = matrix[idx[bound[i]:bound[i+1]]]
            plt.scatter(class_data[:, 0], class_data[:, 1], c=colors[i], marker=markers[i], label=f'Cluster {i+1}')
    else:
        for i in range(len(bound)-1):
            n=np.nanmean(activity[:,idx[bound[i]:bound[i+1]]],axis=0)
            colors=cmap((n+abs(np.min(n)))*(1/np.max(n+abs(np.min(n)))))
            class_data = matrix[idx[bound[i]:bound[i+1]]]
            sc=ax.scatter(class_data[:, 0], class_data[:, 1], c=colors, marker=markers[i], label=f'Cluster {i+1}')
        
    plt.legend(bbox_to_anchor=(1.05, 0.25), loc=3, borderaxespad=0)
    plt.grid(False)
    plt.axhline(y=0,color='gray',linestyle=':')
    plt.axvline(x=0,color='gray',linestyle=':')
    plt.xlabel('w_x')
    plt.ylabel('w_y')
    return fig

def ConnectDistributionBetween(matrix,bound,idx,plot_idx):#由plot_idx给定绘图顺序
    mean=np.ones([(len(bound)-1),(len(bound)-1)])
    var=np.ones([(len(bound)-1),(len(bound)-1)])
    cluster=[]
    for i in range(len(bound)-1):
        for j in range(len(bound)-1):
            if len(cluster)==0:
                cluster=matrix[idx[bound[i]:bound[i+1]]][:,idx[bound[j]:bound[j+1]]].reshape(-1)
                mean[i,j]=np.nanmean(cluster)
                var[i,j]=np.var(cluster) 
                kind=[f'{i+1}←{j+1}']
                continue
            # print(i*(len(bound)-2)+k)
            cluster_ij=matrix[idx[bound[i]:bound[i+1]]][:,idx[bound[j]:bound[j+1]]].reshape(-1) 
            cluster=np.concatenate([cluster,cluster_ij])
            kind_ij=[f'{i+1}←{j+1}']
            kind=kind+kind_ij
            mean[i,j]=np.nanmean(cluster_ij)
            var[i,j]=np.var(cluster_ij)
            
    df_cluster={}
    df_cluster['connect_weights']=cluster
    df_cluster['cluster']=kind
    # ax=sns.violinplot(x='cluster', y='connect_weights', data=df_cluster)
    # plt.axhline(0,linestyle='--',color='red')
    # plt.ylabel('connect_weights')
    # plt.xlabel('cluster relationship')
    fig=plt.figure(figsize=[8,5])
    plt.text(0.13, 0.2, f'within cluster', fontsize=16,horizontalalignment='center',\
                  verticalalignment='center', transform=plt.gca().transAxes,color='blue')
    plt.text(0.6, 0.8, f'between cluster', fontsize=16,horizontalalignment='center',\
                  verticalalignment='center', transform=plt.gca().transAxes,color='blue')
    for i in range(plot_idx.shape[0]):
        x=plot_idx[i,:,0]
        y=plot_idx[i,:,1]
        if i ==0:
            tikes=[kind[j] for j in x*4+y]
            colors='y'
        elif i==1:
            tikes=tikes+[kind[j] for j in x*4+y]
            colors='r'
        else:
            tikes=tikes+[kind[j] for j in x*4+y]
            colors='b'

        plt.errorbar(range((len(x))*(i+1))[-len(x):],y=mean[x,y],yerr=var[x,y],fmt='o',capsize=4,\
                     markersize=10,color=colors)
    # print(kind)
    plt.xticks(range((len(bound)-1)*(len(bound)-1)),tikes,rotation=45)
    plt.axhline(0,linestyle='--',color='red')
    plt.axvline(3.5,linestyle='--',color='blue')
    plt.xlabel('cluster relationship')
    plt.ylabel('mean_connect_weights')
    plt.ylim([-0.01,0.01])
    plt.yticks([-0.01,-0.005,0,0.005,0.01])

    return fig

def ConnectDistributionA1toA2(matrix,bound_1,bound_2,idx_1,idx_2,colors,markers,pmd_idx,m1_idx,A1,A2):#area1->area2 eg:PMd->M1
    mean=np.zeros([(len(bound_1)-1),(len(bound_2)-1)])
    var=np.zeros([(len(bound_1)-1),(len(bound_2)-1)])
    cluster=[]
    fig=plt.figure(figsize=[7,5])
    for i in range(len(bound_1)-1):#area1
        if i ==0:
            kind_i=[f'{A1}-C{i+1}']
        else:
            kind_i=kind_i+[f'{A1}-C{i+1}']
        for j in range(len(bound_2)-1):#area2
            if j == 0:
                kind_j=[f'{A2}-C{j+1}']
            else:
                kind_j=kind_j+[f'{A2}-C{j+1}']
            if len(cluster)==0:
                cluster=matrix[idx_1[bound_1[i]:bound_1[i+1]]][:,idx_2[bound_2[j]:bound_2[j+1]]].reshape(-1)
                mean[i,j]=np.nanmean(cluster)
                var[i,j]=np.var(cluster)
                continue
            cluster_ij=matrix[idx_1[bound_1[i]:bound_1[i+1]]][:,idx_2[bound_2[j]:bound_2[j+1]]].reshape(-1) 
            cluster=np.concatenate([cluster,cluster_ij])
            mean[i,j]=np.nanmean(cluster_ij)
            var[i,j]=np.var(cluster_ij)
    # idx=np.array([0,3,1,2]).astype(int)
    for i in range(pmd_idx.shape[0]):
        ax = plt.subplot(1,1,i+1)
        for j in range(pmd_idx.shape[1]):
            k=pmd_idx[i,j]
            plt.errorbar(range((len(bound_2)-1)),y=mean[k,m1_idx],yerr=var[k,:],\
                         fmt=markers[k],capsize=4,markersize=10,color=colors[len(bound_1)-2-k],\
                            label=kind_i[k])
    # print(kind)
        ax.set_xticks(range((len(bound_2)-1)))
        ax.set_xticklabels((kind_j[i] for i in m1_idx))
        colorsx=['orange', 'purple', 'red', 'green']
        for label, color in zip(ax.get_xticklabels(), colorsx):
            label.set_color(color)
        ax.axhline(0,linestyle=':',color='black')
        # dir=['x','y']
        # ax.set_title(f'PMd cluster→M1_{dir[i]}')
        ax.set_xlabel(f'{A2} cluster')
        ax.set_ylabel('mean_connect_weights')
        ax.set_ylim([-0.01,0.01])
        ax.set_yticks([-0.01,-0.005,0,0.005,0.01])
        xticks = list(range(len(kind_j)))
        xticklabels = [kind_j[i] for i in m1_idx]

        # 在 cluster 1 和 4 之间画虚线
        if f'{A2}-C{m1_idx[0]+1}' in xticklabels and f'{A2}-C{m1_idx[1]+1}' in xticklabels:
            x1, x4 = xticks[xticklabels.index(f'{A2}-C{m1_idx[0]+1}')], xticks[xticklabels.index(f'{A2}-C{m1_idx[1]+1}')]
            for j in range(pmd_idx.shape[1]):
                k=pmd_idx[i,j]
                ax.plot([x1, x4], [mean[k,m1_idx[0]], mean[k,m1_idx[1]]], linestyle='--', color=colors[len(bound_1)-2-k])

        # 在 cluster 2 和 3 之间画虚线
        if f'{A2}-C{m1_idx[2]+1}' in xticklabels and f'{A2}-C{m1_idx[3]+1}' in xticklabels:
            x2, x3 = xticks[xticklabels.index(f'{A2}-C{m1_idx[2]+1}')], xticks[xticklabels.index(f'{A2}-C{m1_idx[3]+1}')]
            for j in range(pmd_idx.shape[1]):
                k=pmd_idx[i,j]
                ax.plot([x2, x3], [mean[k,m1_idx[2]], mean[k,m1_idx[3]]], linestyle='--', color=colors[len(bound_1)-2-k])

        ax.legend(bbox_to_anchor=(1.05, 0.25), loc=3, borderaxespad=0)
    return fig

def psthplot(matrix,bound,idx,title,figsize,color,savname,MO):
    x = np.linspace(0,1,100)
    for i in range(len(bound)-1):
        fig=plt.figure(figsize=figsize)
        ax=plt.subplot(1,1,1)
        for j in range(8):
            mean=np.nanmean(matrix[j,:,idx[bound[i]:bound[i+1]]],axis=0)
            var=np.nanstd(matrix[j,:,idx[bound[i]:bound[i+1]]],axis=0)/np.sqrt(bound[i+1]-bound[i])
            plt.plot(x, mean, label=f'{j*45}°', color=color[j])
            plt.fill_between(x, mean-var, mean+var, color=color[j], alpha=0.2)
        ax.text(0.5, 1.05, f'psth of cluster{i+1}', fontsize=14,horizontalalignment='center',\
                  verticalalignment='center', transform=plt.gca().transAxes)
        ax.set_xticks([0,MO])
        ax.set_ylim([-0.15,0.15])
        ax.axvline(x=MO, linestyle='--', color='k', linewidth=1.5)
        ax.set_xticklabels(['TO','MO'],rotation=0)
        if (not os.path.exists(savname+title)):
            os.makedirs(savname+title)
        plt.savefig(savname+title+f'_cluster{i+1}.png')
    return fig

def plotClusterPsthEnd(matrix,bound,idx,title,velo,figsize,speed,colors,savname):
    x=np.linspace(0,7,8).astype(int)
    dir=[f'{i*45}°' for i in x]
    label='final hand position'
    if len(matrix.shape)==4:
        label='final hand position'
        matrix=np.nanmean(matrix,axis=2)
    for i in range(len(bound)-1):
        fig=plt.figure(figsize=figsize)
        ax=plt.subplot(1,1,1)
        mean=np.zeros([5,8])
        var=np.zeros([5,8])
        for s in range(5):
            for j in range(8):
                mean[s,j]=np.nanmean(matrix[s,j,idx[bound[i]:bound[i+1]]])
                var[s,j]=np.nanstd(matrix[s,j,idx[bound[i]:bound[i+1]]])/np.sqrt(bound[i+1]-bound[i])
                
            plt.plot(x, mean[s], label=f'speed={speed[s]}°/s', color=colors[s])
            plt.fill_between(x, mean[s]-var[s], mean[s]+var[s], color=colors[s], alpha=0.2)
                
        ax.set_xticks(x)
        ax.set_xlabel(label)
        ax.set_ylabel('mean activity')
        ax.set_title(f'cluster={i+1}')
        ax.set_xticklabels(dir,rotation=45)
        if (not os.path.exists(savname+title)):
            os.makedirs(savname+title)
        plt.savefig(savname+title+f'{i}.png')
        plt.show()

def plotClusterPsthPrep(matrix,MO,bound,idx,title,velo,figsize,speed,object_traj,color,savname):
    pi=np.pi
    for i in range(len(bound)-1):
        fig=plt.figure(figsize=figsize)
        for s in range(5):
            ax=plt.subplot(2,3,s+1)
            for j in range(8):
                x = object_traj[s,j,:MO,0]
                y = object_traj[s,j,:MO,1]
                r=np.sqrt(x**2+y**2)
                theta = np.arcsin(y/r)/pi*180       
                theta[x<0] = 180-theta[x<0]
                theta[theta<0]+=360
                if speed[s]>0:
                    for t in range(MO-1):
                        if theta[t+1]>theta[t]:
                            theta[t+1]-=360
                if speed[s]<0:
                    for t in range(MO-1):
                        if theta[t+1]<theta[t]:
                            theta[t+1]+=360
                mean=np.nanmean(matrix[s,j,:MO,idx[bound[i]:bound[i+1]]],axis=0)
                var=np.nanstd(matrix[s,j,:MO,idx[bound[i]:bound[i+1]]],axis=0)/np.sqrt(bound[i+1]-bound[i])
                plt.plot(theta, mean, label=f'{j*45}°', color=color[j])
                plt.fill_between(theta, mean-var, mean+var, color=color[j], alpha=0.2)        
            x_ticks=np.linspace(-180,540,9).astype(int)
            dir=[f'{round(i%360.1)}°' for i in x_ticks]
            ax.text(0.5, 1.05, f'speed={speed[s]}°/s', fontsize=14,horizontalalignment='center',\
                    verticalalignment='center', transform=plt.gca().transAxes)
            ax.set_xticks(x_ticks)
            ax.set_ylabel('mean activity')
        # ax.axvline(x=0.68, linestyle='--', color='k', linewidth=1.5)
            ax.set_xticklabels(dir,rotation=0)
            # ax.set_ylim([-0.14,0.14])
            
        ax.text(velo, 2.35, title+f'cluster{i+1}', fontsize=17,horizontalalignment='center',\
                  verticalalignment='center', transform=plt.gca().transAxes)
        plt.legend(bbox_to_anchor=(1.05, 0.25), loc=3, borderaxespad=0)  
        if (not os.path.exists(savname+title)):
            os.makedirs(savname+title)
        plt.savefig(savname+title+f'cluster{i+1}'+'.png')
        plt.show()
    
def plotClusterPsthPrepConclude(matrix,MO,bound,idx,title,figsize,speed,object_traj,colors,savname):
    pi=np.pi
    for i in range(len(bound)-1):
        fig=plt.figure(figsize=figsize)
        ax=plt.subplot(1,1,1)
        for s in range(matrix.shape[0]):
            a_l=np.linspace(0,360,13).astype(int)
            n=360/(12*2)
            if speed[s] == 0:
                a_l=np.linspace(0,360,9).astype(int)
                n=360/(8*2)
            activity=np.nanmean(matrix[s][:,15:MO][:,:,bound[i]:bound[i+1]],axis=2).reshape(-1)
            # print(activity)
            x = object_traj[s,:,15:MO,0]
            y = object_traj[s,:,15:MO,1]
            r=np.sqrt(x**2+y**2)
            theta = np.arcsin(y/r)/pi*180       
            theta[x<0] = 180-theta[x<0]
            theta[theta<0]+=360
            theta=theta.reshape(-1)
            # if speed[s] == 0:
            #     print(set(theta))
            for a in range(a_l.shape[0]-1):
                theta[(a_l[a-1]+n<=theta)&(theta<a_l[a+1]-n)]=a_l[a]
                if a_l[a]==0:
                    # print(a_l[a-2]+n,np.sort(theta+360))
                    theta[(a_l[a-2]+n<=theta)&(theta<=a_l[a-1])]=a_l[a]
                    theta[(a_l[a]<theta)&(theta<a_l[a]+n)]=a_l[a]

            ac_mean = []  
            ac_var=[]
            for a in range(a_l.shape[0]-1):
                idx = np.where([theta==a_l[a]])#
                # if speed[s] == 0:
                # print(a_l[a],':',idx[1])
                ac_mean.append(np.nanmean(activity[idx[1]]))
                # print(a_l[a],':',ac_mean)
                ac_var.append(np.nanstd(activity[idx[1]])/np.sqrt(bound[i+1]-bound[i]))
            ac_mean = np.array(ac_mean) 
            # if speed[s] == 0:
            #     print(ac_mean)
            ac_var = np.array(ac_var) 
            plt.plot(a_l[:-1], ac_mean, label=f'{speed[s]}°/s', color=colors[s])
            plt.fill_between(a_l[:-1], ac_mean-ac_var, ac_mean+ac_var, color=colors[s], alpha=0.2)        
        x_ticks=np.linspace(0,315,8).astype(int)
        dir=[f'{k}°' for k in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_ylabel('mean activity')
        ax.set_xlabel('traget theta')
        ax.set_title(f'cluster{i+1}')
        # ax.axvline(x=0.68, linestyle='--', color='k', linewidth=1.5)
        ax.set_xticklabels(dir,rotation=45)
            # ax.set_ylim([-0.1,0.1])
        plt.legend(bbox_to_anchor=(1.05, 0.25), loc=3, borderaxespad=0)  
        if (not os.path.exists(savname+title)):
            os.makedirs(savname+title)
        plt.savefig(savname+title+f'cluster{i+1}'+'.png')
        plt.show()

def plotClusterPsthExe(matrix,MO,bound,idx,title,velo,figsize,speed,object_traj,color,savname):
    pi=np.pi
    for i in range(len(bound)-1):
        fig=plt.figure(figsize=figsize)
        for s in range(5):
            ax=plt.subplot(2,3,s+1)
            for j in range(8):
                x = object_traj[s,j,MO:,0]
                y = object_traj[s,j,MO:,1]
                r=np.sqrt(x**2+y**2)
                theta = np.arcsin(y/r)/pi*180       
                theta[x<0] = 180-theta[x<0]
                theta[theta<0]+=360
                if speed[s]>0:
                    for t in range(100-MO-1):
                        if theta[t+1]>theta[t]:
                            theta[t+1]-=360
                if speed[s]<0:
                    for t in range(100-MO-1):
                        if theta[t+1]<theta[t]:
                            theta[t+1]+=360
                mean=np.nanmean(matrix[s,j,MO:,idx[bound[i]:bound[i+1]]],axis=0)
                var=np.nanstd(matrix[s,j,MO:,idx[bound[i]:bound[i+1]]],axis=0)/np.sqrt(bound[i+1]-bound[i])
                plt.plot(theta, mean, label=f'{j*45}°', color=color[j])
                plt.fill_between(theta, mean-var, mean+var, color=color[j], alpha=0.2)        
            x_ticks=np.linspace(-180,540,9).astype(int)
            dir=[f'{round(i%360.1)}°' for i in x_ticks]
            ax.text(0.5, 1.05, f'speed={speed[s]}°/s', fontsize=14,horizontalalignment='center',\
                    verticalalignment='center', transform=plt.gca().transAxes)
            ax.set_xticks(x_ticks)
            ax.set_ylabel('mean activity')
        # ax.axvline(x=0.68, linestyle='--', color='k', linewidth=1.5)
            ax.set_xticklabels(dir,rotation=0)
            # ax.set_ylim([-0.14,0.14])
            
        ax.text(velo, 2.35, f'cluster{i+1}', fontsize=17,horizontalalignment='center',\
                  verticalalignment='center', transform=plt.gca().transAxes)
        plt.legend(bbox_to_anchor=(1.05, 0.25), loc=3, borderaxespad=0)  
        if (not os.path.exists(savname+title)):
            os.makedirs(savname+title)
        plt.savefig(savname+title+f'cluster{i+1}'+'.png')
        plt.show()