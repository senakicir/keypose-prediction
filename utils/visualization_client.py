import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm, colors
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from scipy.stats import mode
import seaborn as sn
import pandas as pd
from itertools import product

import numpy as np
import torch 
from PIL import Image
import imageio
import os

from .data_utils import set_equal_joints, all_bone_connections, joint_num


def get_contrast_colors():
    """ Returns 67 contrast colors.

    Returns:
        dict (str -> tuple): Colors, name -> (R, G, B), values in [0, 1].
    """
    clr_names = \
        ['black', 'crimson', 'blue', 'brown', 'chartreuse', 'chocolate', 'coral',
         'cornflowerblue', 'darkblue', 'darkcyan', 'darkgoldenrod',
         'darkgreen', 'darkmagenta', 'darkolivegreen', 'darkorange',
         'darkorchid', 'darkred', 'darkslateblue', 'darkturquoise',
         'darkviolet', 'deeppink', 'deepskyblue', 'firebrick', 'forestgreen',
         'gold', 'goldenrod', 'green', 'greenyellow', 'hotpink', 'indianred',
         'indigo', 'lawngreen', 'lightsalmon', 'lightseagreen', 'lime',
         'limegreen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue',
         'mediumorchid', 'mediumseagreen', 'mediumslateblue',
         'mediumspringgreen', 'mediumturquoise', 'mediumvioletred',
         'midnightblue', 'navy', 'olive', 'orange', 'orangered', 'peru',
         'purple', 'rebeccapurple', 'red', 'royalblue', 'saddlebrown', 'salmon',
         'seagreen', 'sienna', 'slateblue', 'springgreen', 'steelblue', 'teal',
         'tomato', 'yellow', 'yellowgreen', 'aqua']
    return {k: mcolors.to_rgb(mcolors.CSS4_COLORS[k]) for k in clr_names}

colors = get_contrast_colors()
color_list = []
for key, value in colors.items():
    color_list.append(value)


def save_duration_res(pred_durations, true_durations, epoch, writer=None, plot_loc=None):
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    
    labels = [""]*true_durations.shape[0]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(x - width/2, true_durations, width, label='GT')
    rects2 = ax.bar(x + width/2, pred_durations, width, label='pred')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Durations')
    ax.set_title('Predicted vs. GT durations')
    ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(np.round(height,2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    if writer is not None:
        writer.add_figure('fig/duration_img', fig, epoch+1)
    if plot_loc is not None:
        plt.savefig(plot_loc + 'duration_img.png', dpi=200)

    plt.close(fig)

def save_distribution_res(res_matrix, true_labels, clustering_method, cluster_n, epoch, writer=None, plot_loc=None):
    fig = plt.figure(figsize=(10, 3))

    ax = fig.add_subplot(111)
    im_ = ax.imshow(res_matrix, interpolation='nearest', cmap='viridis', aspect="auto")
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    text_ = np.empty_like(res_matrix, dtype=object)
    #res_matrix = res_matrix.reshape([-1,res_matrix.shape[1]//2])

    # print text with appropriate color depending on background
    thresh = (res_matrix.max() + res_matrix.min()) / 2.0

    for i in range(res_matrix.shape[0]):

        for j in range(res_matrix.shape[1]):
            color = cmap_max if res_matrix[i, j] < thresh else cmap_min

            text_cm = 'X'
            fontsize_=6
            mark_text = False
            if clustering_method == "naive" and j == true_labels[i]:
                mark_text = True
            
            elif clustering_method == "armslegs" and (j == true_labels[i,0] or j == cluster_n[0]+true_labels[i,1]):
                mark_text = True
            
            if not mark_text:
                fontsize_=0
                text_cm = ""

            text_[i, j] = ax.text(
                j, i, text_cm,
                ha="center", va="center", 
                fontsize=fontsize_,
                color=color)

        # Turn spines off and create white grid.
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    fig.tight_layout()
    fig.colorbar(im_, ax=ax)

    if writer is not None:
        writer.add_figure('fig/overfitting_img', fig, epoch+1)
#        plt.savefig(plot_loc + '_training_pred.png', dpi=200)
    if plot_loc is not None:
        plt.savefig(plot_loc + '_training_pred.png', dpi=200)


    ax.set_ylim((res_matrix.shape[0] - 5, -5))
    # plot_loc.add_figure('fig/overfitting_img', fig, epoch+1)
    plt.close(fig)



def display_pose_point_cloud(pose, bone_connections):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121,  projection='3d')
    X = pose[0,:]
    Y = pose[2,:]
    Z = pose[1,:]
    for joint_ind in range(32):
        ax.scatter(X[joint_ind], Y[joint_ind], Z[joint_ind], c='xkcd:black')
        ax.text(X[joint_ind], Y[joint_ind], Z[joint_ind], s=str(joint_ind), c='xkcd:red')

    ax2 = fig.add_subplot(122,  projection='3d')
    for _, bone in enumerate(bone_connections):
        bone = list(bone)
        ax2.plot(X[bone], Y[bone], Z[bone], c='xkcd:black')
        ax2.text(X[bone[1]], Y[bone[1]], Z[bone[1]], s=str(bone[1]), c='xkcd:red')


    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
    plt.close(fig)
    return fig

def display_poses(poses_list, bone_connections, save_loc=None, custom_name=None, time=0, custom_title=None, legend_=None):
    fig = plt.figure(figsize=(4,4))

    ax = fig.add_subplot(111,  projection='3d')
    plots = []
    for ind, poses in enumerate(poses_list):
        X = poses[0,:]
        Y = poses[2,:]
        Z = poses[1,:]
        for _, bone in enumerate(bone_connections):
            bone = list(bone)
            p, = ax.plot(X[bone], Y[bone], Z[bone], c=color_list[ind])
            if len(plots) <= ind:
                plots.append(p)

        ax.set_xlim(-600,600)
        ax.set_ylim(-600,600)
        ax.set_zlim(-600,600)

    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Get rid of the panes                          
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 

    # Get rid of the spines                         
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    if custom_title is not None:
        ax.set_title(custom_title)

    if legend_ is not None:
        ax.legend(plots, legend_)

    if save_loc is not None:
        plt.savefig(save_loc + '/' + custom_name + str(time) + '.png', dpi=100)

    plt.close(fig)
    return fig

def display_pose(pose, bone_connections, color, custom_name, time,save_loc=None):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111,  projection='3d')
    X = pose[0,:]
    Y = pose[2,:]
    Z = pose[1,:]
    for _, bone in enumerate(bone_connections):
        bone = list(bone)
        ax.plot(X[bone], Y[bone], Z[bone], c=color, linewidth=3)

        ax.set_xlim(-600,600)
        ax.set_ylim(-600,600)
        ax.set_zlim(-600,600)

    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Get rid of the panes                          
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 

    # Get rid of the spines                         
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))


    if save_loc is not None:
        plt.savefig(save_loc + '/' + custom_name + str(time) + '.png', dpi=100)

    plt.close(fig)
    return fig


def display_poses_multi(poses_list, bone_connections, rows=1, cols=1, save_loc=None, custom_name=None):
    fig = plt.figure(figsize=(cols*5, rows*5))
    count = 0 
    while True:
        if (count >= rows*cols or count >= poses_list[0].shape[0]):
            break

        ax = fig.add_subplot(rows, cols, int(count+1),  projection='3d')
        for ind, poses in enumerate(poses_list):
            X = poses[count, 0,:]
            Y = poses[count, 2,:]
            Z = poses[count, 1,:]
            for _, bone in enumerate(bone_connections):
                bone = list(bone)
                ax.plot(X[bone], Y[bone], Z[bone], c=color_list[ind])

        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        count += 1

    if save_loc is not None:
        plt.savefig(save_loc + '/poses_' + custom_name + '.png')
    plt.close(fig)


def display_TSNE_clusters(pose_points, labels, save_loc, max_cluster_num, epoch, custom_name = ""):
    assert (pose_points.shape[0]==len(labels))

    #load data
    color_dict = get_contrast_colors()
    num_of_colors = len(color_dict)
    num_clusters = min(num_of_colors, max_cluster_num)

    arr = np.arange(0, max_cluster_num)
    np.random.shuffle(arr)
    random_labels = arr[:num_clusters]
    mask = np.zeros(labels.shape)
    for random_label in random_labels:
         mask = np.logical_or(mask, np.array(labels)==random_label)
    mask_ind = np.nonzero(mask)[0]
    sampled_pose_points = pose_points[mask_ind, :]
    sampled_label_points = labels[mask_ind]
    assert len(sampled_label_points) == sampled_pose_points.shape[0]

    #if too many points:
    max_data_points = 10000
    max_dims = 50

    if sampled_pose_points.shape[0] > max_data_points:
        arr = np.arange(0, sampled_pose_points.shape[0])
        np.random.shuffle(arr)
        random_data_indices = arr[:max_data_points]
        x = sampled_pose_points[random_data_indices]
        y = sampled_label_points[random_data_indices]
    else:
        x = sampled_pose_points
        y = sampled_label_points
    assert x.shape[0] == y.shape[0]
    assert x.shape[0] <= max_data_points

    # PCA
    if sampled_pose_points.shape[1] > max_dims:
        pca = PCA(n_components=max_dims)
        x = pca.fit_transform(x)

    # Settings.
    num_embed_dims = 2
    perplexity = 30.0
    early_exaggeration = 12.0
    lr = 25
    iters = 1000
    iters_without_progress = 600
    metric = 'euclidean'  # 'euclidean', 'cosine', 'l1'
    init = 'random'  # 'random', 'pca'
    angle = 0.5

    # t-SNE
    tsne = TSNE(n_components=num_embed_dims, perplexity=perplexity,
                early_exaggeration=early_exaggeration, learning_rate=lr,
                n_iter=iters, n_iter_without_progress=iters_without_progress,
                metric=metric, init=init, verbose=False, angle=angle)
    x_emb = tsne.fit_transform(x)

    # Plot.
    fig = plt.figure()
    label_ind = 0
    for key, color in color_dict.items():
        x_emb_label = x_emb[y == random_labels[label_ind]] 
        plt.plot(*x_emb_label.T, linestyle='', marker='o', color=color)
        label_ind += 1
        if label_ind == num_clusters:
            break
    plt.savefig(save_loc + '/TSNE_epoch_' + str(epoch) + custom_name +'.png')
    plt.close(fig)




def display_pose_clusters_centers(pose_clusters, bone_connections, save_loc, cluster_num, cluster_weights, epoch, custom_name = ""):
    if cluster_num < 9:
        fig = plt.figure(figsize=(4*cluster_num, 4))
    elif cluster_num < 32: 
        fig = plt.figure(figsize=(int(4*cluster_num/2), 8))
    else:
        #pick first 144
        cluster_num = 144
        fig = plt.figure(figsize=(48, 50))

    for cluster in range(cluster_num):
        if cluster_num < 9:
            ax = fig.add_subplot(1,cluster_num,cluster+1,  projection='3d')
        elif cluster_num < 32: 
            ax = fig.add_subplot(2,np.ceil(cluster_num/2),cluster+1,  projection='3d')
        else:
            ax = fig.add_subplot(12,12,cluster+1, projection='3d')
        
        if cluster_weights is not None:
            plot_title = "weight:{0:.2f}".format(cluster_weights[cluster])
            ax.title.set_text(plot_title)
        
        pose = pose_clusters[cluster,:,:]
        X = pose[0,:]
        Y = pose[1,:]
        Z = pose[2,:]
        for _, bone in enumerate(bone_connections):
            ax.plot(X[bone], Y[bone], Z[bone], c='xkcd:black')

        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    #plt.show()
    plt.suptitle("Epoch {}".format(epoch))
    plt.savefig(save_loc + '/clusters_epoch_' + str(epoch) + custom_name +'.png')
    plt.close(fig)


def display_poses_reconst(reconst_mu, reconst_full, predicted_labels_just_mu, bone_connections, epoch, save_loc):
    fig = plt.figure(figsize=(60, 80))
    clusters = list(range(0,30,6)) 

    color_dict = get_contrast_colors()

    for time_step in range(reconst_mu.shape[1]):
        for cluster_ind, cluster_label in enumerate(clusters):
        
            mask = (predicted_labels_just_mu==cluster_label)
            reconst_list = (reconst_mu[mask], reconst_full[mask])

            if mask.sum()>=5:
                ind = np.arange(mask.sum())
                np.random.shuffle(ind) ##ALERT, YOU SHOULD USE RNG_OBJECT HERE.
                subsample_ind = ind[:5]

                for i in range(2):
                    ax = fig.add_subplot(5, 2, 2*cluster_ind+i+1, projection='3d')
                    reconst_pose = reconst_list[i]
                    pose = np.zeros([5, reconst_pose.shape[1], reconst_pose.shape[2], reconst_pose.shape[3]])
                    pose[:] = reconst_pose[subsample_ind,:,:,:].cpu().data

                    color_ind = 0
                    for key, color in color_dict.items():
                        for _, bone in enumerate(bone_connections):
                            X = pose[color_ind,time_step,0,:]
                            Y = pose[color_ind,time_step,1,:]
                            Z = pose[color_ind,time_step,2,:]
                            ax.plot(X[bone], Y[bone], Z[bone], c=color)
                        color_ind +=1
                        if color_ind==5:
                            break
                    my_var = np.mean(np.std(pose[:,time_step,:,:], axis=0))
                    ax.title.set_text("std: "+str(my_var))

                    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() *0.4
                    mid_x = (X.max()+X.min()) * 0.5
                    mid_y = (Y.max()+Y.min()) * 0.5
                    mid_z = (Z.max()+Z.min()) * 0.5
                    ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax.set_zlim(mid_z - max_range, mid_z + max_range)
                

        #plt.show()
        # plt.suptitle("Epoch {}".format(epoch))
        plt.savefig(save_loc + '/poses_reconst_epoch_' +str(epoch) + "_time_step_" + str(time_step) +'.png')
    plt.close(fig)


def visualize_clusters_both_embed(embed_dict, predicted_labels_just_mu, predicted_labels_with_noise, epoch, save_loc):
    fig = plt.figure(figsize=(40,40))

    mu_embed = embed_dict["mu"]
    complete_embed = torch.cat((embed_dict["mu"], embed_dict["noise"]), dim=1)

    #color of data
    color_dict = get_contrast_colors()

    mask = np.zeros(predicted_labels_just_mu.shape)
    mask_complete = np.zeros(predicted_labels_just_mu.shape)
    mask_4 =  np.zeros(predicted_labels_just_mu.shape)
    mask_complete_4 =  np.zeros(predicted_labels_just_mu.shape)

    labels_list = list(range(0,30,3))

    for label_ind, label in enumerate(labels_list):
        label_mask =  np.array(predicted_labels_just_mu==label)
        mask = np.logical_or(mask, label_mask)

        #check what label "label" corresponds to in second label results.
        complete_label = mode(predicted_labels_with_noise[np.nonzero(label_mask)[0]])
        mask_complete = np.logical_or(mask, np.array(predicted_labels_with_noise==complete_label))

        if label_ind < 4:
            mask_4 = mask.copy()
            mask_complete_4 = mask_complete.copy()

    masks = [mask, mask_complete, mask_4, mask_complete_4]
    num_clusters_list = [10,10,4,4]

    # Settings.
    num_embed_dims = 2
    perplexity = 30.0
    early_exaggeration = 12.0
    lr = 25
    iters = 1000
    iters_without_progress = 600
    metric = 'euclidean'  # 'euclidean', 'cosine', 'l1'
    init = 'random'  # 'random', 'pca'
    angle = 0.5



    for mask_ind in range(0,4):
        ax = fig.add_subplot(2,2,mask_ind+1)
    
        if mask_ind == 0 or mask_ind == 2:
            sampled_embed = mu_embed[np.nonzero(masks[mask_ind])[0], :]
            sampled_labels = predicted_labels_just_mu[np.nonzero(masks[mask_ind])[0]]
        if mask_ind == 1 or mask_ind == 2:
            sampled_embed = complete_embed[np.nonzero(masks[mask_ind])[0], :]
            sampled_labels = predicted_labels_with_noise[np.nonzero(masks[mask_ind])[0]]

        # t-SNE
        tsne = TSNE(n_components=num_embed_dims, perplexity=perplexity,
                    early_exaggeration=early_exaggeration, learning_rate=lr,
                    n_iter=iters, n_iter_without_progress=iters_without_progress,
                    metric=metric, init=init, verbose=False, angle=angle)
        sampled_embed_tsne = tsne.fit_transform(sampled_embed.detach().cpu().numpy())

        # Plot.
        label_ind = 0
        for key, color in color_dict.items():
            x_emb_label = sampled_embed_tsne[sampled_labels == labels_list[label_ind]] 
            ax.plot(*x_emb_label.T, linestyle='', marker='o', color=color)
            label_ind += 1
            if label_ind == num_clusters_list[mask_ind]:
                break
    plt.savefig(save_loc + '/clusters_TSNE_epoch_' + str(epoch)  +'.png')
    plt.close(fig)


def visualize_cluster_sequence(all_poses_clustered, num_clusters, save_loc):
    #import pdb; pdb.set_trace()
    clusters = np.zeros([50, all_poses_clustered.shape[0]])
    #iterate over clusters
    for i in range(num_clusters):
        clusters[:, all_poses_clustered==i] = i

    fig = plt.figure()
    plt.imshow(clusters)
    plt.savefig(save_loc + '/sequence_clustered.png')
    plt.close(fig)

def display_transition_matrix(transition_matrix, sparsity, save_loc, epoch, custom_name =""):
    df_tm = pd.DataFrame(transition_matrix, range(transition_matrix.shape[0]), range(transition_matrix.shape[0]))
  #  df_tm = pd.DataFrame(transition_matrix[:20,:20], range(20), range(20))

    fig = plt.figure(figsize=(40,40))

    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_tm, annot=False, annot_kws={"size": 12}) # font size
    #plt.title("At epoch {}, sp:{0:.2f}".format(epoch, sparsity))
    plt.title(str(sparsity))
    plt.savefig(save_loc + '/transition_matrix_epoch' + str(epoch) + custom_name + '.png')
    plt.close(fig)

def display_confusion_matrix(conf_mat, writer, epoch, actions, plot_loc, ar_type):
    fig, ax = plt.subplots(figsize=(15,15))

    n_classes = conf_mat.shape[0]
    im = ax.imshow(conf_mat, interpolation='nearest', cmap='viridis')
    cmap_min, cmap_max = im.cmap(0), im.cmap(256)

    text_ = np.empty_like(conf_mat, dtype=object)

    # print text with appropriate color depending on background
    thresh = (conf_mat.max() + conf_mat.min()) / 2.0

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if conf_mat[i, j] < thresh else cmap_min

        text_cm = format(conf_mat[i, j], '.2g')
        if conf_mat.dtype.kind != 'f':
            text_d = format(conf_mat[i, j], 'd')
            if len(text_d) < len(text_cm):
                text_cm = text_d

        text_[i, j] = ax.text(j, i, text_cm,ha="center", va="center",color=color)

    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=actions,
            yticklabels=actions,
            ylabel="True label",
            xlabel="Predicted label")

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation="vertical")

    fig.tight_layout()
    if writer is not None:
        writer.add_figure('fig/confusion_matrix', fig, epoch+1)

    if plot_loc is not None:
        fig.savefig(plot_loc + ar_type+'_confusion_matrix.png', dpi=200)

    plt.close(fig)

def sena_display_cluster_centers(dummy_pose, cc_initial, cc_current, dim_used, dataset, writer, save_loc, epoch):
    num_clusters = cc_initial.shape[0]
    cc_list = [cc_initial.clone(), cc_current.detach().cpu().clone()]
    bone_connections = all_bone_connections(dataset)
    n_joints = joint_num(dataset)

    for ind in range(2):
        cc_tmp = cc_list[ind]
        cc = torch.zeros([num_clusters, n_joints*3])
        cc[:,:] = torch.from_numpy(dummy_pose)
        cc[:, dim_used] = cc_tmp
        cc = set_equal_joints(cc.unsqueeze(0), dataset)[0,:,:]
        cc_list[ind] = cc.reshape(cc.shape[0], -1, 3).transpose(2,1).numpy()

    fig = plt.figure(figsize=(20, 20))
    count = 1
    colors = ['xkcd:black', 'xkcd:red']

    for cc_ind in range(int(int(cc.shape[0]//10)*10)):            
        ax = fig.add_subplot(int(cc.shape[0]//10),10,count,  projection='3d')
        count += 1
        
        for plot_ind in range(2):
            plot_cc = cc_list[plot_ind]
            X = plot_cc[cc_ind,0,:]
            Y = plot_cc[cc_ind,2,:]
            Z = plot_cc[cc_ind,1,:]
            for _, bone in enumerate(bone_connections):
                bone = list(bone)
                ax.plot(X[bone], Y[bone], Z[bone], c=colors[plot_ind], linewidth=1)

        ax.set_xlim(-600,600)
        ax.set_ylim(-600,600)
        ax.set_zlim(-600,600)

        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Get rid of the panes                          
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 

        # Get rid of the spines                         
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # if writer is not None:
    #     writer.add_figure('fig/cluster_centers', fig, epoch+1)

    plt.tight_layout()
    plt.savefig(save_loc+'cc_'+str(epoch)+'.png')
    plt.close(fig)
