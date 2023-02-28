import numpy as np
import torch
from torch.utils.data import Dataset

from . import keypose_extract as keypose_module
from utils import data_utils as data_utils


h36m_joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site", "LeftUpLeg", "LeftLeg",
                "LeftFoot","LeftToeBase", "Site", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
                "LeftForeArm","LeftHand", "LeftHandThumb", "Site", "L_Wrist_End", "Site", "RightShoulder", "RightArm",
                "RightForeArm","RightHand", "RightHandThumb", "Site", "R_Wrist_End", "Site"]


def load_human36m_full_data(path_to_dataset, subj, action, subact, sample_rate, is_cuda=True):
    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
    filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, subact)
    action_sequence = data_utils.readCSVasFloat(filename)
    n, d = action_sequence.shape
    even_list = range(0, n, sample_rate)
    num_frames = len(even_list)
    the_sequence = np.array(action_sequence[even_list, :])
    the_sequence = torch.from_numpy(the_sequence).float()
    if is_cuda:
        the_sequence = the_sequence.cuda()

    # remove global rotation and translation
    the_sequence[:, 0:6] = 0
    p3d = data_utils.expmap2xyz_torch(the_sequence, is_cuda=is_cuda)
    the_sequence = p3d.view(num_frames, -1).cpu().data.numpy()

    the_sequence = the_sequence.reshape([num_frames, -1, 3])
    flipped_sequences = the_sequence[:,14,1] < the_sequence[:,3,1]
    if np.sum(flipped_sequences) > 0:
        the_sequence[flipped_sequences, :, 1] = -1*the_sequence[flipped_sequences, :, 1]
    the_sequence = the_sequence.reshape([num_frames, -1])
    return the_sequence, num_frames


class H36motion3D(Dataset):

    def __init__(self, opt, cluster_n, cluster_centers, actions, split="train", nosplits=False):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 validation, 2 testing
        :param sample_rate:
        """
        self.path_to_data = opt.data_dir
        self.overfitting_exp = opt.overfitting_exp
        self.cluster_centers = cluster_centers

        self.split = data_utils.splits[split]
        self.in_n = opt.input_seq_n
        self.out_n = opt.output_seq_n
        seq_len = self.in_n + self.out_n

        self.sample_rate = opt.sample_rate
        self.p3d = {}
        self.kp_val = {}
        self.kp_durs = {}
        self.action_labels = {}
        self.data_idx = []
        self.kp_idx = []
        self.offset_list = []

        self.mean_poses = data_utils.load_mean_poses(self.path_to_data)

        kp_dim = 66
        
        training_test_subjects = np.array([1,5,6,7,8,9])
        test_subject = np.array([opt.test_subj])
        training_subjects = np.setdiff1d(training_test_subjects, test_subject)
        subs = np.array([training_subjects.tolist(), [11], test_subject.tolist()])
    
        # ignore constant joints and joints at same position with other joints
        _, _, self.dim_used  = data_utils.dimension_info(dataset="h36m")

        subs = subs[self.split]
        self.nosplits = nosplits
        key = 0 #increments over subjects, actions, subactions

        for subj in subs:
            for action_idx in np.arange(len(actions)):
                action = actions[action_idx]
                action_label = data_utils.actions_h36m.index(action)
                if self.split == 0 or nosplits:
                    for subact in [1, 2]:  # subactions
                        self.action_labels[key] = torch.ones(1)*action_label
                        self.p3d[key], num_frames = load_human36m_full_data(self.path_to_data, subj, action, subact, self.sample_rate)

                        valid_frames = np.arange(self.in_n, num_frames - self.out_n + 1, opt.sample_rate)

                        #keypose stuff
                        keypose_filename = '{0}/S{1}/{2}_{3}.pkl'.format(opt.keypose_dir, subj, action, subact)
                        kp_seq = keypose_module.load_keyposes(keypose_filename, cluster_n)
                        self.kp_val[key], _, kp_locs, self.kp_durs[key], _ = data_utils.kp_dict_to_input(kp_seq, cluster_n, kp_dim)
       
                        middle_indice_list = []
                        for ind in range(opt.input_kp-1, len(kp_locs)-opt.output_kp):
                            kp_loc = kp_locs[ind]
                            #check if kp is in valid frames
                            if (kp_loc.item() >= self.in_n) and (kp_loc.item() <= num_frames - self.out_n):
                                middle_indice_list.append(kp_loc.item())
                        
                        tmp_data_idx_1 = [key] * len(middle_indice_list) #which key
                        tmp_data_idx_2 = list(middle_indice_list) #the middle frames
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        for test_index_point in middle_indice_list:
                            kp_list, offset = data_utils.find_keyposes_in_sequence(kp_locs, test_index_point, opt.input_kp, opt.output_kp)
                            
                            if kp_list is not None:
                                self.kp_idx.append(kp_list)
                                self.offset_list.append(offset)
                        key += 1
                else:
                    for subact in [1, 2]: 
                        self.action_labels[key] = torch.ones(1)*action_label

                        the_sequence1, num_frames1 = load_human36m_full_data(self.path_to_data, subj, action, subact, self.sample_rate, is_cuda=False)
                        self.p3d[key] = the_sequence1

                        fs_sel1 = data_utils.find_indices(num_frames1, 32)
                        valid_frames = np.array(fs_sel1)
                        tmp_data_idx_1 = [key] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))                        
                        
                        #keypose stuff
                        self.kp_val[key] = {}
                        self.kp_durs[key] = {}

                        if self.split == 1: #validation
                            keypose_filename = '{0}/S{1}/{2}_{3}.pkl'.format(opt.keypose_dir, subj, action, subact)
                            kp_seq = keypose_module.load_keyposes(keypose_filename, cluster_n)
                        
                        for ind in range(32):
                            test_index_point = valid_frames[ind]

                            if self.split == 2: #test #fix this to 2 again!
                                keypose_filename = '{0}/S{1}/{2}_{3}_{4}.pkl'.format(opt.keypose_dir, subj, action, subact, ind)
                                kp_seq = keypose_module.load_keyposes(keypose_filename, cluster_n)
                                
                           
                            self.kp_val[key][ind], _, kp_locs1, self.kp_durs[key][ind], kp_labels = data_utils.kp_dict_to_input(kp_seq, cluster_n, kp_dim)
                            assert self.kp_val[key][ind].shape[0] == self.kp_durs[key][ind].shape[0]

                            if opt.final_frame_kp:
                                if test_index_point not in kp_locs1:
                                    new_ind = torch.sum(kp_locs1<test_index_point)
                                    final_kp_val = torch.from_numpy(the_sequence1[test_index_point:test_index_point+1,self.dim_used])
                                    final_kp_label = torch.argmin(torch.norm(self.cluster_centers-final_kp_val, dim=1), dim=0)
                                    
                                    prev_kp_index=0
                                    if final_kp_label == kp_labels[new_ind-1]:
                                        prev_kp_index=1


                                    kp_duration = test_index_point - kp_locs1[new_ind-prev_kp_index-1]
                                    kp_locs1 = torch.cat([kp_locs1[:new_ind-prev_kp_index], torch.IntTensor([test_index_point]), kp_locs1[new_ind:]])
                                    
                                    self.kp_val[key][ind] = torch.cat([self.kp_val[key][ind][:new_ind-prev_kp_index], final_kp_val, self.kp_val[key][ind][new_ind:]], dim=0)
                                    self.kp_durs[key][ind] = torch.cat([self.kp_durs[key][ind][:new_ind-prev_kp_index], torch.IntTensor([kp_duration]), self.kp_durs[key][ind][new_ind:]])
                                
                            kp_list, offset = data_utils.find_keyposes_in_sequence(kp_locs1, test_index_point, opt.input_kp, opt.output_kp)
                    
                            if kp_list is not None:
                                self.kp_idx.append((ind, kp_list))
                                self.offset_list.append(offset)

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

        if self.split == 0 or self.nosplits:
            ks =  self.kp_idx[item]
            return all_seq, self.kp_val[key][ks,:], self.kp_durs[key][ks],self.offset_list[item], self.action_labels[key]
        else:
            test_ind, ks =  self.kp_idx[item]
            return all_seq, self.kp_val[key][test_ind][ks,:], self.kp_durs[key][test_ind][ks],self.offset_list[item], self.action_labels[key]



class H36motion3D_KP_Sampler(Dataset):

    def __init__(self, opt, cluster_n, cluster_centers, actions, split="train", nosplits=False):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 validation, 2 testing
        :param sample_rate:
        """
        self.path_to_data = opt.data_dir
        self.overfitting_exp = opt.overfitting_exp
        self.cluster_centers = cluster_centers

        self.split = data_utils.splits[split]
        self.in_n = opt.input_seq_n
        self.out_n = opt.output_seq_n
        self.input_kp = opt.input_kp
        seq_len = self.in_n + self.out_n

        self.sample_rate = opt.sample_rate
        self.p3d = {}
        self.kp_val = {}
        self.kp_durs = {}
        self.kp_locs = {}
        self.action_labels = {}
        self.data_idx = []
        self.kp_idx = []
        self.offset_list = []

        self.mean_poses = data_utils.load_mean_poses(self.path_to_data)

        kp_dim = 66
        
        training_test_subjects = np.array([1,5,6,7,8,9])
        test_subject = np.array([opt.test_subj])
        training_subjects = np.setdiff1d(training_test_subjects, test_subject)
        subs = np.array([training_subjects.tolist(), [11], test_subject.tolist()])
    
        # ignore constant joints and joints at same position with other joints
        _, _, self.dim_used  = data_utils.dimension_info(dataset="h36m")

        subs = subs[self.split]
        self.nosplits = nosplits
        key = 0 #increments over subjects, actions, subactions

        for subj in subs:
            for action_idx in np.arange(len(actions)):
                action = actions[action_idx]
                action_label = data_utils.actions_h36m.index(action)
                if self.split == 0 or nosplits:
                    for subact in [1, 2]:  # subactions
                        self.action_labels[key] = torch.ones(1)*action_label
                        self.p3d[key], num_frames = load_human36m_full_data(self.path_to_data, subj, action, subact, self.sample_rate)

                        valid_frames = np.arange(self.in_n, num_frames - self.out_n + 1, opt.sample_rate)

                        #keypose stuff
                        keypose_filename = '{0}/S{1}/{2}_{3}.pkl'.format(opt.keypose_dir, subj, action, subact)
                        kp_seq = keypose_module.load_keyposes(keypose_filename, cluster_n)
                        self.kp_val[key], _, self.kp_locs[key], self.kp_durs[key], _ = data_utils.kp_dict_to_input(kp_seq, cluster_n, kp_dim)
       
                        middle_indice_list = range(self.in_n, num_frames - self.out_n)
                        
                        tmp_data_idx_1 = [key] * len(middle_indice_list) #which key
                        tmp_data_idx_2 = list(middle_indice_list) #the middle frames
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        
                        for test_index_point in middle_indice_list:
                            future_kp_list, offset = data_utils.find_future_keyposes_in_sequence(self.kp_locs[key], test_index_point, opt.output_kp)
                            if future_kp_list is not None:
                                self.kp_idx.append(future_kp_list)
                                self.offset_list.append(offset)
                        key += 1
                else:
                    for subact in [1, 2]: 
                        self.action_labels[key] = torch.ones(1)*action_label

                        the_sequence1, num_frames1 = load_human36m_full_data(self.path_to_data, subj, action, subact, self.sample_rate, is_cuda=False)
                        self.p3d[key] = the_sequence1

                        fs_sel1 = data_utils.find_indices(num_frames1, 32)
                        valid_frames = np.array(fs_sel1)
                        tmp_data_idx_1 = [key] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))                        
                        
                        #keypose stuff
                        self.kp_val[key] = {}
                        self.kp_durs[key] = {}

                        if self.split == 1: #validation
                            keypose_filename = '{0}/S{1}/{2}_{3}.pkl'.format(opt.keypose_dir, subj, action, subact)
                            kp_seq = keypose_module.load_keyposes(keypose_filename, cluster_n)
                        
                        for ind in range(32):
                            test_index_point = valid_frames[ind]

                            if self.split == 2: #test #fix this to 2 again!
                                keypose_filename = '{0}/S{1}/{2}_{3}_{4}.pkl'.format(opt.keypose_dir, subj, action, subact, ind)
                                kp_seq = keypose_module.load_keyposes(keypose_filename, cluster_n)
                                
                                # keypose_filename = '{0}/S{1}/lol_{2}_{3}_{4}.pkl'.format(opt.keypose_dir, 5, action, subact, ind)
                                # joint_to_ignore, dimensions_to_ignore, dimensions_to_use  = dimension_info(dataset="h36m")

                                # kp,loc=keypose_module.run_keypose_finder(sequence[test_index_point-self.in_n:test_index_point,dimensions_to_use], keypose_filename, opt.kp_threshold, False)
                                # vals_gt = torch.from_numpy(kp).float()
                                # loc = torch.FloatTensor(loc)

                                # keypose_durations = loc[1:] - loc[0:-1]
                                # keypose_durations = torch.cat([torch.FloatTensor([10]).cuda(), keypose_durations], dim=0)

                                # self.kp_val[key][ind] = 


                            self.kp_val[key][ind], _, kp_locs1, self.kp_durs[key][ind], kp_labels = data_utils.kp_dict_to_input(kp_seq, cluster_n, kp_dim)
                            assert self.kp_val[key][ind].shape[0] == self.kp_durs[key][ind].shape[0]

                            if opt.final_frame_kp:
                                if test_index_point not in kp_locs1:
                                    new_ind = torch.sum(kp_locs1<test_index_point)
                                    final_kp_val = torch.from_numpy(the_sequence1[test_index_point:test_index_point+1,self.dim_used])
                                    final_kp_label = torch.argmin(torch.norm(self.cluster_centers-final_kp_val, dim=1), dim=0)
                                    
                                    prev_kp_index=0
                                    if final_kp_label == kp_labels[new_ind-1]:
                                        prev_kp_index=1

                                    kp_duration = test_index_point - kp_locs1[new_ind-prev_kp_index-1]
                                    kp_locs1 = torch.cat([kp_locs1[:new_ind-prev_kp_index], torch.IntTensor([test_index_point]), kp_locs1[new_ind:]])
                                    
                                    self.kp_val[key][ind] = torch.cat([self.kp_val[key][ind][:new_ind-prev_kp_index], final_kp_val, self.kp_val[key][ind][new_ind:]], dim=0)
                                    self.kp_durs[key][ind] = torch.cat([self.kp_durs[key][ind][:new_ind-prev_kp_index], torch.IntTensor([kp_duration]), self.kp_durs[key][ind][new_ind:]])
                                
                            kp_list, offset = data_utils.find_keyposes_in_sequence(kp_locs1, test_index_point, opt.input_kp, opt.output_kp)
                            if offset > 0:
                                import pdb; pdb.set_trace()
                            if kp_list is not None:
                                self.kp_idx.append((ind, kp_list))
                                self.offset_list.append(offset)

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

        if self.split == 0 or self.nosplits:
            future_ks =  self.kp_idx[item]

            #extract input kp randomly
            random_input_kp_ind = np.sort(np.random.choice(self.in_n, size=self.input_kp, replace=False))
            kp_val_past = all_seq[random_input_kp_ind, :]
            kp_val_past = kp_val_past[:,  self.dim_used]
            kp_val_future = self.kp_val[key][future_ks,:]
            kp_val = np.concatenate([kp_val_past, kp_val_future], axis=0)

            kp_locs_past_absolute = random_input_kp_ind+middle_frame-self.in_n
            #sanity_check
            kp_loc_future = self.kp_locs[key][future_ks]
            kp_loc = np.concatenate([kp_locs_past_absolute, kp_loc_future])
            kp_durs =  np.concatenate([np.array([10]), kp_loc[1:]-kp_loc[:-1]])
            kp_durs[kp_durs==0]=10
            return all_seq, kp_val, kp_durs, self.offset_list[item], self.action_labels[key]
        else:
            test_ind, ks =  self.kp_idx[item]
            return all_seq, self.kp_val[key][test_ind][ks,:], self.kp_durs[key][test_ind][ks],self.offset_list[item], self.action_labels[key]



class H36motion3D_Sampler(Dataset):

    def __init__(self, opt, cluster_n, cluster_centers, actions, split="train", nosplits=False):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 validation, 2 testing
        :param sample_rate:
        """
        self.path_to_data = opt.data_dir
        self.overfitting_exp = opt.overfitting_exp
        self.cluster_centers = cluster_centers

        self.split = data_utils.splits[split]
        self.in_n = opt.input_seq_n
        self.out_n = opt.output_seq_n
        self.input_kp = opt.input_kp
        self.output_kp = opt.output_kp
        self.interpolator_sample_seq_len = opt.interpolator_sample_seq_len
        seq_len = self.in_n + self.out_n

        self.sample_rate = opt.sample_rate
        self.p3d = {}
        self.action_labels = {}
        self.data_idx = []
        self.kp_val, self.kp_durs = {}, {}
        self.kp_idx = []
        self.offset_list = []
        kp_dim = 66

        self.mean_poses = data_utils.load_mean_poses(self.path_to_data)

        training_test_subjects = np.array([1,5,6,7,8,9])
        test_subject = np.array([opt.test_subj])
        training_subjects = np.setdiff1d(training_test_subjects, test_subject)
        subs = np.array([training_subjects.tolist(), [11], test_subject.tolist()])
    
        # ignore constant joints and joints at same position with other joints
        _, _, self.dim_used  = data_utils.dimension_info(dataset="h36m")

        subs = subs[self.split]
        key = 0 #increments over subjects, actions, subactions

        #for debugging
        self.split = 0 

        for subj in subs:
            for action_idx in np.arange(len(actions)):
                action = actions[action_idx]
                action_label = data_utils.actions_h36m.index(action)
                if self.split == 0:
                    for subact in [1, 2]:  # subactions
                        self.action_labels[key] = torch.ones(1)*action_label
                        self.p3d[key], num_frames = load_human36m_full_data(self.path_to_data, subj, action, subact, self.sample_rate)
        
                        middle_indice_list = range(self.in_n, num_frames - self.out_n)
                        tmp_data_idx_1 = [key] * len(middle_indice_list) #which key
                        tmp_data_idx_2 = list(middle_indice_list) #the middle frames
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        
                        key += 1
                else:
                    for subact in [1, 2]:  # subactions
                        self.action_labels[key] = torch.ones(1)*action_label
                        self.p3d[key], num_frames = load_human36m_full_data(self.path_to_data, subj, action, subact, self.sample_rate)

                        valid_frames = np.arange(self.in_n, num_frames - self.out_n + 1, opt.sample_rate)

                        #keypose stuff
                        keypose_filename = '{0}/S{1}/{2}_{3}.pkl'.format(opt.keypose_dir, subj, action, subact)
                        kp_seq = keypose_module.load_keyposes(keypose_filename, cluster_n)
                        self.kp_val[key], _, kp_locs, self.kp_durs[key], _ = data_utils.kp_dict_to_input(kp_seq, cluster_n, kp_dim)
       
                        middle_indice_list = []
                        for ind in range(opt.input_kp-1, len(kp_locs)-opt.output_kp):
                            kp_loc = kp_locs[ind]
                            #check if kp is in valid frames
                            if (kp_loc.item() >= self.in_n) and (kp_loc.item() <= num_frames - self.out_n):
                                middle_indice_list.append(kp_loc.item())
                        
                        tmp_data_idx_1 = [key] * len(middle_indice_list) #which key
                        tmp_data_idx_2 = list(middle_indice_list) #the middle frames
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        for test_index_point in middle_indice_list:
                            kp_list, offset = data_utils.find_keyposes_in_sequence(kp_locs, test_index_point, opt.input_kp, opt.output_kp)
                            
                            if kp_list is not None:
                                self.kp_idx.append(kp_list)
                                self.offset_list.append(offset)
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

        if self.split == 0:
            full_seq_len = int(self.in_n+self.out_n-2)
            if self.input_kp+self.output_kp > 2:
                random_input_kp_ind = np.sort(np.random.choice(full_seq_len, size=self.input_kp+self.output_kp-2, replace=False))+1        
                random_input_kp_ind = np.concatenate([np.array([0]), random_input_kp_ind, np.array([self.in_n+self.out_n-1])])
            else:
                random_input_kp_ind = np.sort(np.random.choice(full_seq_len, size=1, replace=False))+1        
                random_input_kp_ind = np.concatenate([np.array([0]), random_input_kp_ind])
            
            kp_val = all_seq[random_input_kp_ind, :]
            kp_val = kp_val[:,  self.dim_used]
        
            kp_loc_absolute = random_input_kp_ind+middle_frame-self.in_n
            kp_durs =  np.concatenate([np.array([10]), kp_loc_absolute[1:]-kp_loc_absolute[:-1]])
            kp_durs[kp_durs==0]=10
            
            return all_seq, kp_val, kp_durs, np.zeros([0]), self.action_labels[key]
        else:
            ks =  self.kp_idx[item]
            return all_seq, self.kp_val[key][ks,:], self.kp_durs[key][ks], self.offset_list[item], self.action_labels[key]


def load_h36m_fixed_sec(path_to_dataset, subjects, actions, sample_rate, seq_len=125):
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
    joint_to_ignore, dimensions_to_ignore, dimensions_to_use = data_utils.dimension_info(dataset="h36m")
    all_sampled_sequences = []
    labels = []
    for subj in subjects:
        for action in actions:
            action_label = data_utils.actions_h36m.index(action)
            for subact in [1, 2]:  # subactions
                the_sequence, num_frames = load_human36m_full_data(path_to_dataset, subj, action, subact, sample_rate)
                inds = np.tile(np.arange(seq_len), (num_frames-seq_len, 1)) + np.arange(num_frames-seq_len)[:, np.newaxis]
                sampled_seq = the_sequence[inds, :]
                seq_list = np.split(sampled_seq, sampled_seq.shape[0], axis=0)
                new_seq_list = [seq.squeeze() for seq in seq_list]
                all_sampled_sequences.extend(new_seq_list) 
                sampled_labels = [action_label]*sampled_seq.shape[0]
                labels.extend(sampled_labels)

    return all_sampled_sequences, labels, dimensions_to_ignore, dimensions_to_use


class H36motion3D_SeqOnly(Dataset):
    def __init__(self, path_to_data, actions, seq_len, split="train", sample_rate=2, overfitting_exp=False):
        self.path_to_data = path_to_data
        self.split = split
        self.overfitting_exp = overfitting_exp

        #splits according to training validation and test
        subs = np.array([[1, 6, 7, 8, 9], [11], [5]])
        if self.overfitting_exp:
            subs = np.array([[1], [11], [5]])
        

        subjs = subs[data_utils.splits[split]]
        all_seqs, labels, dim_ignore, dim_used = load_h36m_fixed_sec(path_to_data, subjs, actions, sample_rate, seq_len)
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

class H36M_Test(Dataset):

    def __init__(self, path_to_data, actions, seq_len, split="test", sample_rate=2, overfitting_exp=False):

        self.path_to_data = opt.data_dir
        self.overfitting_exp = opt.overfitting_exp
        self.cluster_centers = cluster_centers

        self.split = data_utils.splits[split]
        self.in_n = opt.input_seq_n
        self.out_n = opt.output_seq_n
        seq_len = self.in_n + self.out_n

        self.split = split
        self.sample_rate = 2
        self.p3d = {}
        self.data_idx = []

        
        if actions is None:
            acts = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
        else:
            acts = actions

        subs = [5]
        key = 0
        for subj in subs:
            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]
                if self.split <= 1:
                    pass
                else:
                    for subact in [1, 2]:
                        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                        filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)
                        the_sequence1 = data_utils.readCSVasFloat(filename)
                        n, d = the_sequence1.shape
                        even_list = range(0, n, self.sample_rate)

                        num_frames1 = len(even_list)
                        the_sequence1 = np.array(the_sequence1[even_list, :])
                        the_seq1 = torch.from_numpy(the_sequence1).float().cuda()
                        the_seq1[:, 0:6] = 0
                        p3d1 = data_utils.expmap2xyz_torch(the_seq1)
                        self.p3d[key] = p3d1.view(num_frames1, -1).cpu().data.numpy()

                        fs_sel1 = data_utils.find_indices(num_frames1, 32, short_seq=False)

                        valid_frames = np.array(fs_sel1)
                        tmp_data_idx_1 = [key] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                        key += 1

        # ignore constant joints and joints at same position with other joints
        joint_to_ignore, dimensions_to_ignore, self.dim_used  = data_utils.dimension_info(dataset="h36m")

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, middle_frame = self.data_idx[item]
        fs = np.arange(middle_frame-self.in_n, middle_frame + self.out_n)
        return self.p3d[key][fs]



class H36motion3D_EntireSeq(Dataset):

    def __init__(self, opt, actions, split="train"):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 validation, 2 testing
        :param sample_rate:
        """
        self.path_to_data = opt.data_dir
        self.overfitting_exp = opt.overfitting_exp

        self.split = data_utils.splits[split]
        kp_threshold = opt.kp_threshold

        self.sample_rate = opt.sample_rate
        self.p3d = {}
        self.seq_len = {}
        self.kp = {}
        self.kp_num = {}
        self.action_labels={}
        self.max_len = 0
        self.max_kp_num = 0

        subs = np.array([[1, 6, 7, 8, 9], [11], [5]])
        if self.overfitting_exp:
            subs = np.array([[1], [11], [5]])
    
        # ignore constant joints and joints at same position with other joints
        _, _, self.dim_used  = data_utils.dimension_info(dataset="h36m")

        subs = subs[self.split]
        key = 0 #increments over subjects, actions, subactions

        for subj in subs:
            for action_idx in np.arange(len(actions)):
                action = actions[action_idx]
                action_label = data_utils.actions_h36m.index(action)
                for subact in [1, 2]:  # subactions
                    self.action_labels[key] = torch.ones(1)*action_label
                    self.p3d[key], self.seq_len[key] = load_human36m_full_data(self.path_to_data, subj, action, subact, self.sample_rate)
                    self.max_len = self.max_len if self.seq_len[key]< self.max_len else self.seq_len[key]

                    #keypose stuff
                    self.kp_vals[key], self.kp_loc[key] = keypose_module.run_keypose_init(self.p3d[key][:, self.dim_used], None, threshold=kp_threshold, save_kp=False)
                    self.kp_num[key] = self.kp_loc[key].shape[0]
                    self.max_kp_num = self.max_kp_num if self.kp_num[key]< self.max_kp_num else self.kp_num[key]
                    key += 1

    def __len__(self):
        if self.overfitting_exp:
            return 1        
        return len(self.p3d)

    def __getitem__(self, item):
        seq_len = self.seq_len[item]
        seq = torch.zeros([max_len, 32*3])

        seq[:seq_len, :] = self.p3d[item]

        kp_num = self.kp_num[item]
        kp_vals = torch.zeros([map_kp_num, 22*3])
        kp_locs = torch.zeros([map_kp_num])

        kp_vals[:kp_num, :] = self.kp_vals[item]
        kp_locs[:kp_num, :] = self.kp_loc[item]

        action_label = self.action_labels[item]

        return seq, kp_vals, kp_locs, seq_len, kp_num, action_label




class H36motion3D_SubSequences(Dataset):

    def __init__(self, opt, actions, split="train"):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 validation, 2 testing
        :param sample_rate:
        """
        self.path_to_data = opt.data_dir
        self.overfitting_exp = opt.overfitting_exp

        self.split = data_utils.splits[split]

        self.sample_rate = opt.sample_rate
        self.p3d = {}
        self.seq_len = {}
        self.data_idx= []
        self.action_labels={}

        subs = np.array([[1, 6, 7, 8, 9], [11], [5]])
        if self.overfitting_exp:
            subs = np.array([[1], [11], [5]])
    
        # ignore constant joints and joints at same position with other joints
        _, _, self.dim_used  = data_utils.dimension_info(dataset="h36m")

        subs = subs[self.split]
        key = 0 #increments over subjects, actions, subactions

        for subj in subs:
            for action_idx in np.arange(len(actions)):
                action = actions[action_idx]
                action_label = data_utils.actions_h36m.index(action)
                for subact in [1, 2]:  # subactions
                    self.action_labels[key] = torch.ones(1)*action_label
                    self.p3d[key], num_frames = load_human36m_full_data(self.path_to_data, subj, action, subact, self.sample_rate)

                    beginning_ind_list =  np.arange(0, num_frames - 512, 10)
                    tmp_data_idx_1 = [key] * len(beginning_ind_list) #which key
                    tmp_data_idx_2 = list(beginning_ind_list) #the middle frames
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                    key += 1

    def __len__(self):
        if self.overfitting_exp:
            return 1        
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, beginning_frame = self.data_idx[item]
        fs = np.arange(beginning_frame, beginning_frame+512)
        return self.p3d[key][fs], self.action_labels[key]

