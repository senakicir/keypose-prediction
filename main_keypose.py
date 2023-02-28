import os
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch 
import pickle

from utils import loss_funcs, utils as utils
from utils.opt import Options
import utils.data_utils as data_utils
import utils.h36motion3d as h36m_dataset
import utils.cmu_motion_3d as cmu_dataset
import utils.data_utils as data_utils

import utils.keypose_extract as keypose_module
from utils.keypose_extract.keypose_utils import keypose_directory, set_data

torch.set_num_threads(8)

def main(opt):

    dataset = opt.dataset
    _, _, dimensions_to_use  = data_utils.dimension_info(dataset=dataset)

    #define the subjects
    training_test_subjects = np.array([1,5,6,7,8,9])
    test_subj = np.array([opt.test_subj])
    training_subjs = np.setdiff1d(training_test_subjects, test_subj)
    all_subjects = [1, 5, 6, 7, 8, 9, 11, 5]

    acts = data_utils.define_actions(opt.actions, dataset=dataset) 

    opt.keypose_dir += keypose_directory(dataset, opt.kp_threshold, opt.kp_suffix)

    #create the directories if they do not exist
    if dataset=="h36m":
        for subj in all_subjects:
            if not os.path.exists('{0}/S{1}'.format(opt.keypose_dir, subj)):
                os.makedirs('{0}/S{1}'.format(opt.keypose_dir, subj))
                opt.reevaluate_keyposes = True
    elif dataset=="cmu":
        subdirs = ["train", "test", "val"]
        for action in acts:
            for subdir in subdirs:
                if not os.path.exists('{0}/{1}'.format(opt.keypose_dir+"/"+subdir, action)):
                    os.makedirs('{0}/{1}'.format(opt.keypose_dir+"/"+subdir, action))
                    opt.reevaluate_keyposes = True
            
    def call_keypose_module(sequence, keypose_filename, save_kp=True):
        kp,loc=keypose_module.run_keypose_finder(sequence, keypose_filename, opt.kp_threshold, save_kp)
        return kp,loc

    def load_keypose_data(keypose_filename):
        data_loaded=None
        if (os.path.isfile(keypose_filename)):
            data = {}
            data_loaded=None
            with open(keypose_filename, 'rb') as pkl_file:
                data_loaded =  pickle.load(pkl_file)
        if data_loaded is None:
            print(keypose_filename)
        return np.array(data_loaded["keyposes"]), np.array(data_loaded["loc"])

    def select_relevant_dimensions(sequence):
        input_sequence = sequence[:, dimensions_to_use]
        return input_sequence 

    if opt.reevaluate_keyposes:
        if dataset == "h36m":
            for subj in all_subjects:
                if subj not in test_subj:
                    for act in acts:
                        for subact in [1,2]:
                            #load the sequence
                            sequence, num_frames = h36m_dataset.load_human36m_full_data(opt.data_dir, subj, act, subact, sample_rate=2, is_cuda=torch.cuda.is_available())                    
                            input_sequence = select_relevant_dimensions(sequence)

                            #find the keyposes
                            start = time.time()
                            keypose_filename = '{0}/S{1}/{2}_{3}.pkl'.format(opt.keypose_dir, subj, act, subact)
                            
                            call_keypose_module(input_sequence, keypose_filename)
                            print("Reevaluating keyposes took",  time.time()-start)

            #for test subject, we should only look at the past when extracting keyposes, instead of the entire sequence.
            for subj in test_subj:
                for act in acts:
                    for subact in [1,2]:
                        #load the sequence
                        sequence, num_frames = h36m_dataset.load_human36m_full_data(opt.data_dir, subj, act, subact, sample_rate=2, is_cuda=torch.cuda.is_available())                    
                        ind_list = data_utils.find_indices(num_frames, num_ind=32)
                        input_sequence = select_relevant_dimensions(sequence)
                        
                        #find the keyposes
                        for ind in range(len(ind_list)):
                            start = time.time()
                            keypose_filename = '{0}/S{1}/{2}_{3}_{4}.pkl'.format(opt.keypose_dir, subj, act, subact, ind)
                            #past kp
                            past_kp, past_loc= call_keypose_module(input_sequence[:ind_list[ind]], keypose_filename, save_kp=False)
                            #future kp
                            seq_len = input_sequence.shape[0]
                            kp, loc=call_keypose_module(input_sequence[ind_list[ind]:seq_len], keypose_filename, save_kp=False)
                            loc = loc + ind_list[ind]
                            
                            appended_loc = np.concatenate([past_loc, loc], axis=0)
                            appended_kp = np.concatenate([past_kp, kp], axis=0)
                            #write
                            data ={}
                            data["loc"] = appended_loc
                            data["keyposes"] = appended_kp
                            with open(keypose_filename, 'wb') as keypose_pkl_file:
                                pickle.dump(data, keypose_pkl_file)
                            print("Reevaluating keyposes took", time.time()-start)

        elif dataset=="cmu":
            for action in acts:
                for split in []:#["/train/", "/val/"]:
                    path = '{}/{}'.format(opt.cmu_data_dir+split, action)
                    count = 0
                    for _ in os.listdir(path):
                        count = count + 1
                    for examp_index in np.arange(count):
                        sequence, num_frames = cmu_dataset.load_cmu_full_data(opt.cmu_data_dir+split, action, examp_index, is_cuda=False)
                        input_sequence = select_relevant_dimensions(sequence)

                        #find the keyposes
                        start = time.time()
                        print("Sequence length is", num_frames)
                        keypose_filename =  '{0}{3}{1}/{1}_{2}.pkl'.format(opt.keypose_dir, action, examp_index+1, split)
                        call_keypose_module(input_sequence, keypose_filename)
                        print("Reevaluating keyposes took",  time.time()-start)

            for action in acts:
                path = '{}/{}'.format(opt.cmu_data_dir+"/test/", action)
                count = 0
                for _ in os.listdir(path):
                    count = count + 1
                for examp_index in np.arange(count):
                    sequence, num_frames = cmu_dataset.load_cmu_full_data(opt.cmu_data_dir+"/test/", action, examp_index, is_cuda=False)
                    input_sequence = select_relevant_dimensions(sequence)
                    ind_list = data_utils.find_indices(num_frames, num_ind=32, short_seq=True)

                    #find the keyposes
                    for ind in range(len(ind_list)):
                        start = time.time()
                        keypose_filename =  '{0}/test/{1}/{1}_{2}_{3}.pkl'.format(opt.keypose_dir, action, examp_index+1, ind)
                        print("Sequence length is",ind_list[ind])
                        past_kp, past_loc= call_keypose_module(input_sequence[:ind_list[ind]], keypose_filename, save_kp=False)

                        seq_len = input_sequence.shape[0]
                        kp, loc=call_keypose_module(input_sequence[ind_list[ind]:seq_len], keypose_filename, save_kp=False)
                        loc = loc + ind_list[ind]

                        #append both together
                        appended_loc = np.concatenate([past_loc, loc], axis=0)
                        appended_kp = np.concatenate([past_kp, kp], axis=0)

                        #write
                        data ={}
                        data["loc"] = appended_loc
                        data["keyposes"] = appended_kp
                        with open(keypose_filename, 'wb') as keypose_pkl_file:
                            pickle.dump(data, keypose_pkl_file)
                        
                        print("Reevaluating keyposes took", time.time()-start)
                

    ## CLUSTERING:
    n_clusters = opt.cluster_n

    training_data = np.zeros((0, len(dimensions_to_use)))
    # load training data
    if dataset == "h36m":
        for subj in training_subjs:
            for act in acts:
                for subact in [1,2]:
                    keypose_filename = '{0}/S{1}/{2}_{3}.pkl'.format(opt.keypose_dir, subj, act, subact)
                    loaded_kp_data,_ = load_keypose_data(keypose_filename)
                    training_data = np.concatenate([training_data, loaded_kp_data], axis=0)     
    
    elif dataset == "cmu":
        for action in acts:
            for split in ["/train/"]:
                path = '{}/{}'.format(opt.cmu_data_dir+split, action)
                count = 0
                for _ in os.listdir(path):
                    count = count + 1
                for examp_index in np.arange(count):
                    keypose_filename =  '{0}{3}{1}/{1}_{2}.pkl'.format(opt.keypose_dir, action, examp_index+1, split)
                    loaded_kp_data,_ = load_keypose_data(keypose_filename)
                    training_data = np.concatenate([training_data, loaded_kp_data], axis=0)    


    dims = len(dimensions_to_use)
    print("Loading data done")
    print("Clustering began")

    #prepare cluster center file
    cc_filename = '{0}/cluster_centers_{1}.pkl'.format(opt.keypose_dir, opt.cluster_n)
    if (not os.path.isfile(cc_filename)):
        with open(cc_filename, 'wb') as pkl_file:
            centers={"naive":{}}
            pickle.dump(centers, pkl_file)

    centers = {}
    with open(cc_filename, 'rb') as pkl_file:
        centers_loaded =  pickle.load(pkl_file)
        for key, value in centers_loaded.items():
            centers[key] = value
        
    # clustering using k-means
    clf = KMeans(n_clusters=n_clusters)
    t = time.time()

    if not opt.load_clusters:
        print("Clustering {} keyposes".format(training_data.shape[0]))
        clf.fit(training_data)
        cluster_centers_np = clf.cluster_centers_
    else:
        with open(cc_filename, 'rb') as pkl_file:
            centers_loaded = pickle.load(pkl_file)
            cc_dict = centers_loaded["naive"]
            cluster_centers_np = np.zeros([opt.cluster_n, dims])
            for cluster in range(n_clusters):
                cluster_centers_np[cluster, :] =  cc_dict[cluster]

    print("Clustering took", time.time()-t)

    centers["naive"] = {}
    for cluster in range(n_clusters):
        centers["naive"][cluster] = (cluster_centers_np[cluster])

    if not opt.load_clusters:
        cc_filename = '{0}/cluster_centers_{1}.pkl'.format(opt.keypose_dir, n_clusters)
        with open(cc_filename, 'wb') as pkl_file:
            pickle.dump(centers, pkl_file)

    def cluster_keyposes(keypose_filename, cluster_n):
        data = {}
        if (os.path.isfile(keypose_filename)):
            with open(keypose_filename, 'rb') as pkl_file:
                # load the keypose file that already exists
                data_loaded =  pickle.load(pkl_file)
                # keyposes and loc do not need to be overwritten, so keep them
                for key, value in data_loaded.items():
                    data[key] = value

                keyposes = np.array(data_loaded["keyposes"])
                locs = np.array(data_loaded["loc"])

                #calculate labels
                distances = np.linalg.norm(cluster_centers_np[np.newaxis, :, :]-keyposes[:, np.newaxis, :], axis=2)
                labels = np.argmin(distances, axis=1)

                # remove some unnecessary labels
                keep_list = [0]
                label_before = labels[0]
                for label_ind in range(1,len(labels)-1):
                    keep=False
                    label_before = labels[label_ind-1]
                    label_after = labels[label_ind+1]
                    if not(labels[label_ind] == label_before and labels[label_ind] == label_after):
                        keep=True
                    if locs[label_ind-1] >= locs[label_ind] or locs[label_ind] >= locs[label_ind+1]:
                        keep=False
                    if keep:
                        keep_list.append(label_ind)
                keep_list.append(len(labels)-1)

                
                #resets potentially existing data for naive and hirechical clustering
                data = set_data(data, data_loaded)
                
                data["naive_labels"][cluster_n] = labels
                data["naive_distances"][cluster_n] = distances
                data["naive_inds"][cluster_n] = keep_list

            # print("Writing", len(labels), "keyposes to", keypose_filename)
            with open(keypose_filename, 'wb') as keypose_pkl_file:
                pickle.dump(data, keypose_pkl_file)
            
            
        return data

    t = time.time()    
    if dataset == "h36m":
        for subj in all_subjects:
            for act in acts:
                for subact in [1,2]:
                    if subj not in test_subj:
                        keypose_filename = '{0}/S{1}/{2}_{3}.pkl'.format(opt.keypose_dir, subj, act, subact)
                        cluster_keyposes(keypose_filename, opt.cluster_n)
                    else:
                        for ind in range(32):
                            keypose_filename = '{0}/S{1}/{2}_{3}_{4}.pkl'.format(opt.keypose_dir, subj, act, subact, ind)
                            cluster_keyposes(keypose_filename, opt.cluster_n)
                        
    elif dataset == "cmu":
        for action in acts:
            for split in ["/train/", "/val/", "/test/"]:
                path = '{}/{}'.format(opt.cmu_data_dir+split, action)
                count = 0
                for _ in os.listdir(path): 
                    count = count + 1
                for examp_index in np.arange(count):
                    keypose_filename =  '{0}{3}{1}/{1}_{2}.pkl'.format(opt.keypose_dir, action, examp_index+1, split)
                    cluster_keyposes(keypose_filename, opt.cluster_n)

        for action in acts:
            path = '{}/{}'.format(opt.cmu_data_dir+"/test/", action)
            count = 0
            for _ in os.listdir(path):
                count = count + 1
            for examp_index in np.arange(count):
                #find the keyposes
                for ind in range(32):
                    keypose_filename =  '{0}/test/{1}/{1}_{2}_{3}.pkl'.format(opt.keypose_dir, action, examp_index+1, ind)
                    cluster_keyposes(keypose_filename, opt.cluster_n)                

    print("Predicting and rewriting took", time.time()-t)

if __name__ == "__main__":
    option = Options().parse()
    main(option)