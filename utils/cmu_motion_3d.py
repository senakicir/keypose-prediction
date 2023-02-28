import numpy as np
import os
import torch


from torch.utils.data import Dataset
from torch.autograd.variable import Variable

from utils import data_utils
from . import keypose_extract as keypose_module


def load_cmu_full_data(path_to_dataset, action, examp_index, sample_rate=2, is_cuda=True):
    print("Reading action {0}, index {1}".format(action, examp_index+1))
    filename = '{}/{}/{}_{}.txt'.format(path_to_dataset, action, action, examp_index + 1)
    action_sequence = data_utils.readCSVasFloat(filename)
    n, d = action_sequence.shape

    exptmps = Variable(torch.from_numpy(action_sequence)).float()
    if is_cuda:
        exptmps = exptmps.cuda()

    xyz = data_utils.expmap2xyz_torch_cmu(exptmps, is_cuda=is_cuda)
    xyz = xyz.view(-1, 38 * 3)
    xyz = xyz.cpu().data.numpy()
    action_sequence = xyz

    even_list = range(0, n, sample_rate)
    num_frames = len(even_list)
    the_sequence = np.array(action_sequence[even_list, :])
    the_sequence = torch.from_numpy(the_sequence).float()
    if is_cuda:
        the_sequence = the_sequence.cuda()

    num_frames = len(the_sequence)
    return the_sequence, num_frames



class CMU_Motion3D(Dataset):

    def __init__(self, opt, cluster_n, cluster_centers, actions, split="train", nosplits=False):

        self.path_to_data = opt.cmu_data_dir
        self.overfitting_exp = opt.overfitting_exp

        self.split = data_utils.splits[split]
        self.in_n = opt.input_seq_n
        self.out_n = opt.output_seq_n
        seq_len = self.in_n + self.out_n

        self.sample_rate = opt.sample_rate
        self.p3d = {}
        self.kp = {}
        self.kp_durs = {}
        self.kp_val = {}
        self.kp_labels = {}
        self.action_labels = {}
        self.data_idx = []
        self.kp_idx = []
        self.offset_list = []

        kp_dim = 75
        self.nosplits = nosplits

        joint_to_ignore, dimensions_to_ignore, self.dim_used  = data_utils.dimension_info(dataset="cmu")
        
        self.cluster_centers = cluster_centers
        self.mean_poses = data_utils.load_mean_poses(self.path_to_data)
        if self.split == 0: 
            path_to_data = self.path_to_data + '/train'
        elif self.split==1:
            path_to_data = self.path_to_data + '/val'
        elif self.split == 2:
            path_to_data = self.path_to_data + '/test'

        key = 0 #increments over subjects, actions, subactions
        discarded_count = 0

        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]
            action_label = data_utils.actions_cmu.index(action)
            path = '{}/{}'.format(path_to_data, action)
            count = 0
            for _ in os.listdir(path):
                count = count + 1
            for examp_index in np.arange(count):
                # the_sequence, num_frames = load_cmu_full_data(path_to_dataset, action, examp_index)
                self.p3d[key], num_frames = load_cmu_full_data(path_to_data, action, examp_index, is_cuda=False)
                self.action_labels[key] = torch.ones(1)*action_label

                if self.split == 0 or self.split==1 or self.nosplits:
                    #keypose stuff
                    keypose_filename = '{0}/{3}/{1}/{1}_{2}.pkl'.format(opt.keypose_dir, action, examp_index+1, split)
                    kp_seq = keypose_module.load_keyposes(keypose_filename, cluster_n)
                    self.kp_val[key], self.kp[key], kp_locs, self.kp_durs[key], self.kp_labels[key] = data_utils.kp_dict_to_input(kp_seq, cluster_n, kp_dim)

                    middle_indice_list = []
                    for ind in range(opt.input_kp-1, len(kp_locs)-opt.output_kp):
                        kp_loc = kp_locs[ind]
                        #check if kp is in valid frames
                        if (kp_loc.item() >= self.in_n) and (kp_loc.item() <= num_frames - self.out_n):
                            middle_indice_list.append(kp_loc)
                    
                    tmp_data_idx_1 = [key] * len(middle_indice_list) #which key
                    tmp_data_idx_2 = list(middle_indice_list) #the middle frames
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                    for test_index_point in middle_indice_list:
                        kp_list, offset = data_utils.find_keyposes_in_sequence(kp_locs, test_index_point, opt.input_kp, opt.output_kp)
                        if kp_list is not None:
                            self.kp_idx.append(kp_list)
                            self.offset_list.append(offset)
                        else:
                            discarded_count += 1
                    key += 1
                else:
                    fs_sel1 = data_utils.find_indices(num_frames, 32, short_seq=True)
                    valid_frames = np.array(fs_sel1)
                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))                        

                    #keypose stuff
                    self.kp_val[key] = {}
                    self.kp[key] = {}
                    self.kp_labels[key] = {}
                    self.kp_durs[key] = {}

                    if self.split == 1: #validation
                        keypose_filename =  '{0}/val/{1}/{1}_{2}.pkl'.format(opt.keypose_dir, action, examp_index+1)
                        kp_seq = keypose_module.load_keyposes(keypose_filename, cluster_n)
                        
                    for ind in range(32):
                        if self.split == 2: #test
                            keypose_filename = '{0}/test/{1}/{1}_{2}_{3}.pkl'.format(opt.keypose_dir, action, examp_index+1, ind)
                            kp_seq = keypose_module.load_keyposes(keypose_filename, cluster_n)
                        test_index_point = valid_frames[ind]

                        self.kp_val[key][ind], self.kp[key][ind], kp_locs1, self.kp_durs[key][ind], self.kp_labels[key][ind] = data_utils.kp_dict_to_input(kp_seq, cluster_n, kp_dim)
                        if opt.final_frame_kp:
                            if test_index_point not in kp_locs1:
                                new_ind = torch.sum(kp_locs1<test_index_point)
                                final_kp_val = self.p3d[key][test_index_point:test_index_point+1,self.dim_used]
                                final_kp_label = torch.argmin(torch.norm(self.cluster_centers-final_kp_val, dim=1), dim=0)
                                prev_kp_index=0
                                if final_kp_label == self.kp_labels[key][ind][new_ind-1]:
                                    prev_kp_index=1

                                kp_duration = test_index_point - kp_locs1[new_ind-prev_kp_index-1]
                                kp = torch.zeros(self.kp[key][ind][0:1,:].shape)
                                kp[:,-1] = float(kp_duration)
                                kp_locs1 = torch.cat([kp_locs1[:new_ind-prev_kp_index], torch.IntTensor([test_index_point]), kp_locs1[new_ind:]])
                                self.kp_val[key][ind] = torch.cat([self.kp_val[key][ind][:new_ind-prev_kp_index], final_kp_val, self.kp_val[key][ind][new_ind:]], dim=0)
                                self.kp[key][ind] = torch.cat([self.kp[key][ind][:new_ind-prev_kp_index], kp, self.kp[key][ind][new_ind:]], dim=0)
                                self.kp_durs[key][ind] = torch.cat([self.kp_durs[key][ind][:new_ind-prev_kp_index], torch.IntTensor([kp_duration]), self.kp_durs[key][ind][new_ind:]])
                                self.kp_labels[key][ind] = torch.cat([self.kp_labels[key][ind][:new_ind-prev_kp_index], torch.LongTensor([final_kp_label]), self.kp_labels[key][ind][new_ind:]])
                        
                        kp_list, offset = data_utils.find_keyposes_in_sequence(kp_locs1, test_index_point, opt.input_kp, opt.output_kp)
                        if kp_list is not None:
                            self.kp_idx.append((ind, kp_list))
                            self.offset_list.append(offset)
                        else:
                            discarded_count += 1

                    key += 1


    def __len__(self):
        if self.overfitting_exp:
            return 1        
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, middle_frame = self.data_idx[item]
        
        seq_len = self.p3d[key].shape[0]
        if seq_len >= middle_frame + self.out_n:
            fs = np.arange(middle_frame-self.in_n, middle_frame + self.out_n)
            all_seq = self.p3d[key][fs]
        else:
            fs = np.arange(middle_frame-self.in_n, seq_len)
            all_seq = torch.zeros(self.in_n+self.out_n, self.p3d[key].shape[1])
            all_seq[0:len(fs)] = self.p3d[key][fs]
            flipped_seq = self.p3d[key].clone().flip(dims=[0])
            all_seq[len(fs):] = flipped_seq[:all_seq.shape[0]-len(fs)]

        if self.split < 2 or self.nosplits:
            ks =  self.kp_idx[item]
            return all_seq, self.kp_val[key][ks,:],  self.kp_durs[key][ks], self.offset_list[item], self.action_labels[key]
        else:
            test_ind, ks =  self.kp_idx[item]
            return all_seq, self.kp_val[key][test_ind][ks,:], self.kp_durs[key][test_ind][ks],self.offset_list[item], self.action_labels[key]

     

def load_cmu_fixed_sec(path_to_dataset, actions, sample_rate, seq_len=125):
    """
    adapted from
    https://github.com/una-dinosauria/human-motion-prediction/src/data_utils.py#L216
    :param path_to_dataset:
    :param subjects:
    :param actions:
    :param sample_rate:
    :param seq_len:
    :return:
    """

    # ignore constant joints and joints at same position with other joints    
    joint_to_ignore, dimensions_to_ignore, dimensions_to_use  = data_utils.dimension_info(dataset="cmu")
    
    all_sampled_sequences = []
    labels = []
    action_folder_num = {}
    ave_num_frames = {}
    for action_idx in np.arange(len(actions)):
        ave_num_frames[action_idx] =0
        action = actions[action_idx]
        path = '{}/{}'.format(path_to_dataset, action)
        count = 0
        for _ in os.listdir(path):
            count = count + 1
        action_folder_num[action_idx] = count
        for examp_index in np.arange(action_folder_num[action_idx]):
            the_sequence, num_frames =  load_cmu_full_data(path_to_dataset, action, examp_index, is_cuda=False)
            ave_num_frames[action_idx] += num_frames
        ave_num_frames[action_idx] = ave_num_frames[action_idx]/action_folder_num[action_idx]

    min_num_frames = np.inf
    for action_idx in np.arange(len(actions)):
        action = actions[action_idx]
        print("Ave num of frames for action", action, "is", ave_num_frames[action_idx])
        if ave_num_frames[action_idx] < min_num_frames:
            min_num_frames =  int(ave_num_frames[action_idx])

    for action_idx in np.arange(len(actions)):
        action = actions[action_idx]
        action_label = data_utils.actions_cmu.index(action)
        for examp_index in np.arange(action_folder_num[action_idx]):
            the_sequence, num_frames =  load_cmu_full_data(path_to_dataset, action, examp_index, is_cuda=False)
            # if num_frames > min_num_frames:
            #     the_sequence = the_sequence[:min_num_frames]
            #     num_frames = min_num_frames

            inds = np.tile(np.arange(seq_len), (num_frames-seq_len, 1)) + np.arange(num_frames-seq_len)[:, np.newaxis]
            sampled_seq = the_sequence[inds, :]
            seq_list = np.split(sampled_seq, sampled_seq.shape[0], axis=0)
            new_seq_list = [seq.squeeze() for seq in seq_list]
            all_sampled_sequences.extend(new_seq_list) 
            sampled_labels = [action_label]*sampled_seq.shape[0]
            labels.extend(sampled_labels)

    return all_sampled_sequences, labels, dimensions_to_ignore, dimensions_to_use



class CMU_Motion3D_SeqOnly(Dataset):
    def __init__(self, path_to_data, actions, seq_len, split="train", sample_rate=2, overfitting_exp=False):
        self.path_to_data = path_to_data
        self.overfitting_exp = overfitting_exp
        self.split = data_utils.splits[split]


        if self.split==0:
            path_to_data = path_to_data + '/train'
        elif self.split==1:
            path_to_data = path_to_data + '/val'
        elif self.split==2:
            path_to_data = path_to_data + '/test'

        all_seqs, labels, dim_ignore, dim_used = load_cmu_fixed_sec(path_to_data, actions, sample_rate, seq_len)
        self.dim_used = dim_used
        self.all_seqs = all_seqs
        self.labels = labels

    def __len__(self):
        if self.overfitting_exp:
            return 1
        return len(self.all_seqs)

    def __getitem__(self, item):
        if self.overfitting_exp:
            return (self.all_seqs[0], self.labels[0])
        return self.all_seqs[item], self.labels[item]