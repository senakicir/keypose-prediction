import numpy as np
import matplotlib.pyplot as plt
import os, io
import pickle
import time
import torch

from .linefit import LinearFit

def run_keypose_finder(target, keypose_loc, kp_threshold, save_kp=True):

    """
    nc: 'Number of curves per nuplet'
    nk: 'Number of keypoints per curve (default 10)'
    ns: 'Number of samples per curve (default 100) (for plotting and interp)'
    """

    nc = target.shape[1]
    ns = target.shape[0]//8
    #target: dim of N,66
    
    #%% Find keyposes.
    eps = kp_threshold
    lin = LinearFit(num_t=ns,target=target,eps=eps)
    x0 = lin.getState()
    print('Found {} different poses'.format(lin.nk))

    data = {}
    data["loc"] = x0[0:lin.nk].transpose()
    data["keyposes"] = x0[lin.nk:].reshape(nc, lin.nk).transpose()

    if save_kp:
        with open(keypose_loc, 'wb') as keypose_pkl_file:
            pickle.dump(data, keypose_pkl_file)
            
    return np.array(data["keyposes"]), np.array(data["loc"])


def load_keyposes(keypose_loc, cluster_n):
    '''
        reads pkl file, saves to dict
        dict has keys
        "value"
        "loc"
        "naive_labels"
        
    '''
    keypose_list = []
    if (os.path.isfile(keypose_loc)):
        with open(keypose_loc, 'rb') as pkl_file:
            data_loaded =  pickle.load(pkl_file)
            values = np.array(data_loaded["keyposes"],  dtype=np.float64)
            locs = np.array(data_loaded["loc"])
            labels = np.array(data_loaded["naive_labels"][cluster_n])
            distances = np.array(data_loaded["naive_distances"][cluster_n])
            keep_indices = np.array(data_loaded["naive_inds"][cluster_n])
            for idx in range(len(keep_indices)):  
                keypose = {}
                keypose["keyposes"] = values[keep_indices[idx]]
                keypose["loc"] = int(locs[keep_indices[idx]])
                keypose["naive_labels"] = labels[keep_indices[idx]]
                keypose["naive_distances"] = distances[keep_indices[idx],:]
                if idx != 0:
                    keypose["duration"] = int(locs[keep_indices[idx]] - locs[keep_indices[idx-1]])
                else:
                    keypose["duration"] = 10.0
                keypose_list.append(keypose)         

    assert len(keypose_list)!= 0 and len(keypose_list)> 1
    return keypose_list

def load_cluster_centers(path_to_keyposes, num_clusters):
    cc_filename = '{0}/cluster_centers_{1}.pkl'.format(path_to_keyposes, num_clusters)
    with open(cc_filename, 'rb') as pkl_file:
        cluster_center_dict = pickle.load(pkl_file)

    dim = len(cluster_center_dict["naive"][0])
    cluster_center_array = torch.zeros([num_clusters, dim])
    for cluster_ind in range(num_clusters):
        cluster_center_array[cluster_ind, :] = torch.tensor(cluster_center_dict["naive"][cluster_ind])    
    return cluster_center_array, num_clusters
