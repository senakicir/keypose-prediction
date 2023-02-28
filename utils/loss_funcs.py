import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from utils import data_utils
from utils.utils import AccumLoss
from utils.data_utils import get_body_members, rotmat2euler_tensor, expmap2rotmat_tensor
from utils.forward_kinematics import seq_to_angles_transformer


class Loss_Function_Manager(object):
    def __init__(self, opt, cluster_centers):
        """
        This class keeps track of the loss value used for training.
        Also, it has the functions to transform the data between different representations:
        For example: from "distances to cluster centers" to probability distributions.
        """
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum")
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.binary_ce = torch.nn.BCEWithLogitsLoss(reduction="sum") 
        self.sigmoid = nn.Sigmoid()
        self.KL_divergence = nn.KLDivLoss(reduction="sum")
        self.mse = nn.MSELoss(reduction="sum")


        self.duration_category_num = opt.duration_category_num
        self.cluster_centers = cluster_centers.cuda()
         
        self.augment_output = opt.augment_output
        self.duration_lambda = opt.duration_lambda
        self.KL_lambda = opt.KL_lambda
        self.crossentropy_lambda = opt.crossentropy_lambda
        self.oracle_fixed_time = opt.oracle_fixed_time
        self.fixed_time = opt.fixed_time

        self.supervise_past = opt.supervise_past
        self.kd_t = opt.kd_t
        self.sampling_kd_t = opt.sampling_kd_t
        self.kd_cutoff= opt.kd_cutoff
        self.scale =opt.scale
        self.input_kp = opt.input_kp
        self.supervise_past = opt.supervise_past

        #560, 720, 880, 1000
        self.standard_eval_frame = np.array([14, 18, 22, 25])-1
        if opt.output_seq_n == 25:
            self.eval_frame = self.standard_eval_frame
        if opt.output_seq_n == 125:
            self.eval_frame = np.concatenate([self.standard_eval_frame, np.array([25,125])-1])
        else:
            self.eval_frame = self.standard_eval_frame
        self.diverse_seq_num = opt.diverse_seq_num

        self.beta = 0.0

        body_members = get_body_members(opt.dataset)
        self.angle_trans_function = seq_to_angles_transformer(body_members)

        #init loss dicts
        self.training_loss_keys = ["ce_loss", "KL_div_loss", "duration_loss", "loss"]

        self.eval_frame_keys = ["mse_"+str(frame+1) for frame in self.eval_frame]
        self.loss_dict = {}
        self.diverse_mse_dict = {}
        
        self.loss_dict_batch = {"kl_pred_gt_ave":AccumLoss(),"kl_gt_pred_ave":AccumLoss(),"kl_pred_gt_min":AccumLoss(),"kl_gt_pred_min":AccumLoss(), "mse":AccumLoss(), "mse_upto_1sec":AccumLoss()} 
        #mse loss dicts
        self.loss_dict_batch["diverse_mse_ave_125"]=AccumLoss()
        self.loss_dict_batch["diverse_mse_min_125"]=AccumLoss()
        self.loss_dict_batch["diverse_mse_min_25"]=AccumLoss()
        self.loss_dict_batch["diverse_mse_ave_25"]=AccumLoss()
        self.loss_dict_batch["diverse_mse_bestoverall_125"]=AccumLoss()
        self.loss_dict_batch["diverse_mse_bestoverall_25"]=AccumLoss()

        for key in self.training_loss_keys:
            self.loss_dict[key] = 0
            self.loss_dict_batch[key] = AccumLoss()

        for frame in self.eval_frame:
            self.loss_dict_batch["mse_"+str(frame+1)] = AccumLoss()
            self.diverse_mse_dict["diverse_mse_"+str(frame+1)] = {}
        self.diverse_mse_dict["diverse_mse"] ={}
        self.diverse_mse_dict["diverse_mse_upto25"] ={}
        
        self.diverse_pskl_dict = {}
        self.diverse_pskl_dict["kl_pred_gt"] = {}
        self.diverse_pskl_dict["kl_gt_pred"] = {}
        

    def reset_all(self):
        """
            Resets all loss values. Called at the beginning of the epoch.
        """
        for key, loss in self.loss_dict.items():
            self.loss_dict[key] = 0
        for key, loss in self.loss_dict_batch.items():
            self.loss_dict_batch[key] = AccumLoss()

        for frame in self.eval_frame:
            self.diverse_mse_dict["diverse_mse_"+str(frame+1)] = {}
        self.diverse_mse_dict["diverse_mse"] ={}
        self.diverse_mse_dict["diverse_mse_upto25"] ={}
        self.diverse_pskl_dict["kl_pred_gt_min"] = {}
        self.diverse_pskl_dict["kl_gt_pred_min"] = {}
        self.diverse_pskl_dict["kl_pred_gt_ave"] = {}
        self.diverse_pskl_dict["kl_gt_pred_ave"] = {}
    def reset_for_batch(self):
        """
            Resets loss_dict to be reused in every batch. Called at the beginning of each batch.
        """
        for key, loss in self.loss_dict.items():
            self.loss_dict[key] = 0

    def update_loss(self, ind, label_logit_pred, duration_pred, labels_prob_gt, label_gt, duration_probs_gt):
        """
            ind: Time step (kp index)
            label_logit_pred: predicted logits of the labels (output of network)
            duration_pred: predicted logits of the duration (output of network)
            labels_prob_gt: GT probability distribution of the labels
            durations_gt : GT distribution of the durations
        """
        # if supervise_past is set to true, then we will also have a loss on the predicted
        # PAST values. This is weighted by "past_weight".
        past_weight = 1.0
        if self.supervise_past:
            past_weight = min(ind/(self.input_kp-1), 1.0)

        durations_gt_label = torch.argmax(duration_probs_gt, dim=1)
       
        logprobs = self.logsoftmax(label_logit_pred)
        self.loss_dict["KL_div_loss"] += past_weight*self.KL_lambda*(-(labels_prob_gt * logprobs).sum())
        self.loss_dict["ce_loss"] += past_weight*self.crossentropy_lambda*self.cross_entropy(label_logit_pred, label_gt)

        logprobs_dur = self.logsoftmax(duration_pred)
        self.loss_dict["duration_loss"] += past_weight*self.duration_lambda*(-(duration_probs_gt * logprobs_dur).sum())

        
    def update_loss_batch(self, count):
        """
            Updates the loss for this batch.
        """
        for key, value in self.loss_dict.items():
            if torch.is_tensor(value):  
                self.loss_dict_batch[key].update(value.cpu().data.numpy(), count)
    
    def save_best_ave_loss(self):
        batch_size = self.diverse_mse_dict["diverse_mse"][0].shape[0]
        losses_upto125 = torch.zeros(self.diverse_seq_num, batch_size)
        losses_125 = torch.zeros(self.diverse_seq_num, batch_size)
        losses_upto25 = torch.zeros(self.diverse_seq_num, batch_size)
        losses_25 = torch.zeros(self.diverse_seq_num, batch_size)

        for diverse_ind in range(self.diverse_seq_num):
            losses_upto125[diverse_ind,:]=self.diverse_mse_dict["diverse_mse"][diverse_ind]
            losses_upto25[diverse_ind,:]=self.diverse_mse_dict["diverse_mse_upto25"][diverse_ind]
            losses_125[diverse_ind,:]=self.diverse_mse_dict["diverse_mse_125"][diverse_ind]            
            losses_25[diverse_ind,:]=self.diverse_mse_dict["diverse_mse_25"][diverse_ind]            

        min_losses_ind = torch.argmin(losses_upto125, dim=0)
        min_losses = losses_125[min_losses_ind, torch.arange(batch_size)]
        ave_losses = torch.mean(losses_125, dim=0)
        self.loss_dict_batch["diverse_mse_bestoverall_125"].update(torch.sum(min_losses), batch_size)
        self.loss_dict_batch["diverse_mse_ave_125"].update(torch.sum(ave_losses), batch_size)

        min_losses_ind = torch.argmin(losses_upto25, dim=0)
        min_losses = losses_25[min_losses_ind, torch.arange(batch_size)]
        ave_losses = torch.mean(losses_25, dim=0)
        self.loss_dict_batch["diverse_mse_bestoverall_25"].update(torch.sum(min_losses), batch_size)
        self.loss_dict_batch["diverse_mse_ave_25"].update(torch.sum(ave_losses), batch_size)


        min_losses_ind = torch.argmin(losses_25, dim=0)
        min_losses = losses_25[min_losses_ind, torch.arange(batch_size)]
        self.loss_dict_batch["diverse_mse_min_25"].update(torch.sum(min_losses), batch_size)
        min_losses_ind = torch.argmin(losses_125, dim=0)
        min_losses = losses_125[min_losses_ind, torch.arange(batch_size)]
        self.loss_dict_batch["diverse_mse_min_125"].update(torch.sum(min_losses), batch_size)

        min_losses_sorted_args = torch.argsort(losses_upto125, dim=0)
        include = [0,10,40,70,99]
        ind_list = [min_losses_sorted_args[ind] for ind in include]
        

        ### PSKL
        kl_pred_gt = torch.zeros(self.diverse_seq_num, batch_size)
        kl_gt_pred = torch.zeros(self.diverse_seq_num, batch_size)
        kl_comb = torch.zeros(self.diverse_seq_num, batch_size)
        for diverse_ind in range(self.diverse_seq_num):
            kl_pred_gt[diverse_ind,:]=self.diverse_pskl_dict["kl_pred_gt"][diverse_ind]
            kl_gt_pred[diverse_ind,:]=self.diverse_pskl_dict["kl_gt_pred"][diverse_ind]
            kl_comb[diverse_ind, :] = 0.5*(kl_pred_gt[diverse_ind,:]+kl_gt_pred[diverse_ind,:])
            
        min_losses_ind = torch.argmin(kl_comb, dim=0)
        pskl_pred_gt = kl_pred_gt[min_losses_ind, torch.arange(batch_size)]
        pskl_gt_pred = kl_gt_pred[min_losses_ind, torch.arange(batch_size)]
        self.loss_dict_batch["kl_pred_gt_min"].update(torch.sum(pskl_pred_gt), batch_size)
        self.loss_dict_batch["kl_gt_pred_min"].update(torch.sum(pskl_gt_pred), batch_size)
        pskl_pred_gt = torch.mean(kl_pred_gt, dim=0)
        pskl_gt_pred = torch.mean(kl_gt_pred, dim=0)
        self.loss_dict_batch["kl_pred_gt_ave"].update(torch.sum(pskl_pred_gt), batch_size)
        self.loss_dict_batch["kl_gt_pred_ave"].update(torch.sum(pskl_gt_pred), batch_size)
        return ind_list


    def update_mse_loss(self, pred, gt, diverse_ind=0):
        """
            Updates the MSE loss.
        """
        batch_size, num_frames, num_joints, _ = (pred.shape)
        loss = torch.sum(torch.mean(torch.norm(pred.reshape(batch_size*num_frames,num_joints,3)-
                        gt.reshape(batch_size*num_frames,num_joints,3), p=2,dim=2), dim=1))

        count = num_frames*batch_size
        self.loss_dict_batch["mse"].update(loss.cpu().data.numpy(), count)

        pred_resh = pred.reshape(batch_size,num_frames,num_joints,3)
        gt_resh = gt.reshape(batch_size,num_frames,num_joints,3)
        loss = torch.mean(torch.mean(torch.norm(pred_resh[:,:125]-gt_resh[:,:125], p=2,dim=3), dim=2), dim=1).cpu()
        self.diverse_mse_dict["diverse_mse"][diverse_ind] = loss
        loss = torch.mean(torch.mean(torch.norm(pred_resh[:,:25]-gt_resh[:,:25], p=2,dim=3), dim=2), dim=1).cpu()
        self.diverse_mse_dict["diverse_mse_upto25"][diverse_ind] = loss

        pred_tmp = pred_resh.permute(0,2,1,3).reshape(batch_size*num_joints, num_frames, 3)
        gt_tmp = gt_resh.permute(0,2,1,3).reshape(batch_size*num_joints, num_frames, 3)


        #also log MSE loss ONLY up to 1 second
        loss = torch.sum(torch.norm(pred[:,:25,:,:].reshape(-1,3)-gt[:,:25,:,:].reshape(-1,3), p=2,dim=1))
        count = 25*batch_size*num_joints
        self.loss_dict_batch["mse_upto_1sec"].update(loss.cpu().data.numpy(), count)

        for frame in self.eval_frame: 
            loss = torch.mean(torch.norm(pred[:,frame,:,:]-gt[:,frame,:,:], p=2,dim=2), dim=1).cpu()
            self.diverse_mse_dict["diverse_mse_"+str(frame+1)][diverse_ind] = loss
            self.loss_dict_batch["mse_"+str(frame+1)].update(torch.sum(loss), batch_size)


    def return_average_losses(self):
        my_losses = {}
        for key, value in self.loss_dict_batch.items():
            my_losses[key] = value.avg
        return my_losses

    def sum_losses(self, batch_size):
        for key in self.training_loss_keys:
            if key != "loss" and key!="mse_cc_loss":
                self.loss_dict["loss"]=self.loss_dict["loss"]+ self.loss_dict[key]/batch_size
        return self.loss_dict["loss"]

    def process_output_for_sampling(self, label_logits, duration_logits):
        """
            label_logits is the label prediction of the network. We run it through the heated softmax
            to convert it to a probability distribution from which we can sample. Used in diverse prediction mode.
                label_logits: shape (batch_size, num_clusters)
        """
        cluster_probabilities = knowledge_distillation(label_logits, self.sampling_kd_t)
        duration_probabilities = knowledge_distillation(duration_logits, self.sampling_kd_t)
        return cluster_probabilities, duration_probabilities

    def convert_output_to_tokens(self, output_network, input_noise):
        """
            output_network is the label prediction of the network. We convert it to a keypose label token
            in order to feed it back to the network for the next time step.
                output_network: shape (batch_size, num_clusters)
        """

        if input_noise is not None and self.augment_output:
            input_noise=input_noise
        else:
            input_noise=None

        pred_label = torch.argmax(output_network, dim=1).long()
        new_prob = self.convert_label_to_input(pred_label, self.cluster_centers, input_noise) 

        return new_prob

 

    def value_from_labels(self,labels,rng, mode):
        """
        Finds the pose value from the result of the network (logits_pred)
        params:
            logits_pred (torch.tensor): has shape (batch_size, cluster_n)
                for naive clustering.
        returns 
            val_pred (torch.tensor): has shape (batch_size, 3*num_joints).
        """
        #distilled logits has shape (batch_size, cluster_n)
        #self.cluster_centers has shape (cluster_n, 3*num_joints)
        val_pred = self.cluster_centers[labels]
        return val_pred



    def value_from_logits(self,logits_pred):
        """
        Finds the pose value from the result of the network (logits_pred)
        params:
            logits_pred (torch.tensor): has shape (batch_size, cluster_n)
                for naive clustering.
        returns 
            val_pred (torch.tensor): has shape (batch_size, 3*num_joints).
        """
        assert logits_pred.ndim == 2 

        #distilled logits with very low temp (workaround for argmax)
        #distilled_logits = knowledge_distillation(logits_pred, 0.001)
        label_pred = torch.argmax(logits_pred, dim=1).detach()
        
        #distilled logits has shape (batch_size, cluster_n)
        #self.cluster_centers has shape (cluster_n, 3*num_joints)
        val_pred = self.cluster_centers[label_pred]

        return val_pred

    def values_to_dist(self, values):
        """
            Goes from keypose values to a "distances to cluster centers" representation. 
             
            Params:
                values: torch tensor of shape (batch_size, seq_len, 3*num_joints)
                    or shape (batch_size, 3*num_joints)

            Returns: 
                keypose_dist: (batch_size, seq_len, num_clusters) or (batch_size, num_clusters) 
                labels: (batch_size, seq_len)  or (batch_size) 
        """
        dims = values.ndim
        if dims == 3:
            batch_n, seq_len = values.shape[0], values.shape[1]
            values = values.reshape(batch_n*seq_len, -1)
        else:
            batch_n, seq_len = values.shape[0], 1

        # find the distances to each cluster center.
        keypose_dist = torch.norm(self.cluster_centers.unsqueeze(0)-values.unsqueeze(1), dim=2)
        
        assert keypose_dist.shape == (batch_n*seq_len, self.cluster_centers.shape[0])
        
        # label of the cluster center with the minimum distance
        labels = torch.argmin(keypose_dist, dim=1)

        if dims == 3:
            keypose_dist = keypose_dist.reshape([batch_n, seq_len, -1])
            labels = labels.reshape([batch_n, seq_len])
        return keypose_dist, labels

    def convert_label_to_input(self, pred_label, cluster_centers, input_noise=None):
        """
            Converts labels to keypose tokens, by taking the distance of the cluster
            center represented by the label to all the other cluster centers. This distance
            representation is then converted to a probability distribution (keypose token.)
            Params:
                pred_label: Predicted labels of shape (batch,)
                cluster_centers: cluster centers of shape (n_cluster, 3*num_joints)
            Returns:
                new_prob: Probability distribution representation of label, shape (batch, n_cluster)
        """
        cc_pred = cluster_centers[pred_label, :]
        dist_pred = torch.norm(cc_pred.unsqueeze(1)-cluster_centers.unsqueeze(0), dim=2)

        assert dist_pred.shape == (cc_pred.shape[0], cluster_centers.shape[0])
        new_prob = dist_to_prob(dist_pred, self.kd_t, input_noise=input_noise)
        return new_prob

    def label_from_probabilities(self, prob_dist):
        label_pred = torch.argmax(prob_dist, dim=1).long()
        return label_pred

    def dist_to_prob(self, distances, input_noise=None):
        return dist_to_prob(distances, self.kd_t, input_noise)

    def compute_ent_metrics(self, gt_seqs, seqs, format='coords', diverse_ind=0):
        batch_size = seqs.shape[0]
        seq_len = seqs.shape[1]
        
        assert seqs.shape[3] ==3
        assert seq_len == 125, str(seq_len)

        #for seq_start, seq_end in [(s * (seq_len // 4), (s+1) * (seq_len // 4)) for s in range(4)] + [(0, seq_len)]:
        gt_seqs_tmp = gt_seqs.cpu().numpy()#gt_seqs[:,seq_start:seq_end, :]
        seqs_tmp = seqs.cpu().numpy()#seqs[:,seq_start:seq_end, :]

        gt_seqs_tmp = gt_seqs_tmp.reshape([batch_size*seq_len, -1, 3])
        seqs_tmp =  seqs_tmp.reshape([batch_size*seq_len, -1, 3])       

        # gt_cent_seqs = gt_seqs_tmp - gt_seqs_tmp[:, 0, np.newaxis, :]
        # cent_seqs = seqs_tmp - seqs_tmp[:, 0, np.newaxis, :]

        gt_angle_expmaps = self.angle_trans_function(gt_seqs_tmp)
        angle_expmaps = self.angle_trans_function(seqs_tmp)

        gt_angle_seqs = rotmat2euler_tensor(expmap2rotmat_tensor(gt_angle_expmaps))
        angle_seqs = rotmat2euler_tensor(expmap2rotmat_tensor(angle_expmaps))

        gt_angle_seqs = gt_angle_seqs.reshape([batch_size, seq_len, -1])
        angle_seqs = angle_seqs.reshape([batch_size, seq_len, -1])

        gt_seqs_fft = np.fft.fft(gt_angle_seqs, axis=1)
        gt_seqs_ps = np.abs(gt_seqs_fft) ** 2

        seqs_fft = np.fft.fft(angle_seqs, axis=1)
        seqs_ps = np.abs(seqs_fft) ** 2

        gt_seqs_ps_global = gt_seqs_ps + 1e-8
        gt_seqs_ps_global /= (gt_seqs_ps_global.sum(axis=1, keepdims=True))

        seqs_ps_global = seqs_ps + 1e-8
        seqs_ps_global /= (seqs_ps_global.sum(axis=1, keepdims=True))

        # seqs_ent_global = -np.sum(seqs_ps_global * np.log(seqs_ps_global), axis=1)
        
        seqs_kl_gen_gt = np.mean(np.mean(seqs_ps_global * np.log(seqs_ps_global / gt_seqs_ps_global), axis=2), axis=1)
        seqs_kl_gt_gen = np.mean(np.mean(gt_seqs_ps_global * np.log(gt_seqs_ps_global / seqs_ps_global), axis=2), axis=1)
        
        self.diverse_pskl_dict["kl_pred_gt"][diverse_ind] = torch.from_numpy(seqs_kl_gen_gt).float()
        self.diverse_pskl_dict["kl_gt_pred"][diverse_ind] = torch.from_numpy(seqs_kl_gt_gen).float()

    def categorize_durations(self, durations):
        return categorize_durations(durations, self.duration_category_num)

    def decategorize_durations(self,categorized_durations, duration_cat_sampled=None):
        return decategorize_durations(categorized_durations,self.duration_category_num, self.oracle_fixed_time, self.fixed_time, duration_cat_sampled)


def dist_to_prob(distances, kd_t, input_noise=None):
    """
    Convert distances (distance of keypose to each cluster center) 
    to probabilities using knowledge distillation
    
    params:
        distances: has shape  (batch_size, seq_len, num_clusters)
            for naive clustering.
        kd_t: float value, temperature of the heated softmax
        input_noise: input noise value to be added on top of prob. distribution
            (for data augmentation)
    returns 
        cluster_probabilities: has shape (batch_size, seq_len, num_clusters).
    """
    
    # flip distance values so that bigger values mean more probable clusters
    # and scale them so that max magnitude is 1
    max_values, _ = torch.max(torch.abs(distances), dim=distances.ndim-1)
    inverse_scaled_distances = -1.0*distances /(max_values.unsqueeze(distances.ndim-1) + 1e-7)
    assert torch.sum(torch.isnan(inverse_scaled_distances))==0

    # add noise to distribution
    if input_noise is not None:
        inverse_scaled_distances = inverse_scaled_distances + input_noise

    # knowledge distribution is a heated softmax. pass inverse_scaled_distances 
    # through heated softmax so that everything sums up to 1. 
    cluster_probabilities = knowledge_distillation(inverse_scaled_distances, kd_t)

    return cluster_probabilities


def add_noise_to_prob(cluster_probabilities, input_noise=None):
    """
        This function receives a probability distribution (only one), 
        shape either (batch_size, seq_len, cluster_num) or (batch_size, cluster_num)
    """

    # add noise for data augmentation
    if input_noise is not None:
        cluster_probabilities = cluster_probabilities + input_noise
        min_val, _ = torch.min(cluster_probabilities, dim=cluster_probabilities.ndim-1)
        cluster_probabilities = cluster_probabilities - min_val.unsqueeze(cluster_probabilities.ndim-1)

    #make sure either everything sums up to 1 or max is 1.
    cluster_probabilities_sum = torch.sum(cluster_probabilities, dim=cluster_probabilities.ndim-1)
    res = cluster_probabilities/cluster_probabilities_sum.unsqueeze(cluster_probabilities.ndim-1)  
    return res

def knowledge_distillation(input_mat, t):
    """
        Heated softmax.
    """
    res = torch.softmax(input_mat/t, dim=input_mat.ndim-1)
    assert torch.sum(torch.isnan(res))==0
    return res



def categorize_durations(durations, duration_category_num):
    """
        Go from integer value to categorized durations.
    """

    if durations.ndim==2:
        flattened_durations = durations.reshape(durations.shape[0]*durations.shape[1]).cuda()
    elif durations.ndim==1:
        flattened_durations = durations

    if duration_category_num == 5:
        d = {"vs":5.0, "s":10.0, "l":14.0, "vl":25.0}

        categorized = torch.zeros(flattened_durations.shape[0],len(d)+1).cuda()
        categorized[flattened_durations<=d["vs"], 0] = 0.90
        categorized[flattened_durations<=d["vs"], 1] = 0.10

        categorized[torch.logical_and(flattened_durations>d["vs"], flattened_durations<=d["s"]), 0] = 0.10
        categorized[torch.logical_and(flattened_durations>d["vs"], flattened_durations<=d["s"]), 1] = 0.80
        categorized[torch.logical_and(flattened_durations>d["vs"], flattened_durations<=d["s"]), 2] = 0.10

        categorized[torch.logical_and(flattened_durations>d["s"], flattened_durations<=d["l"]), 1] = 0.10
        categorized[torch.logical_and(flattened_durations>d["s"], flattened_durations<=d["l"]), 2] = 0.80
        categorized[torch.logical_and(flattened_durations>d["s"], flattened_durations<=d["l"]), 3] = 0.10

        categorized[torch.logical_and(flattened_durations>d["l"], flattened_durations<=d["vl"]), 2] = 0.10
        categorized[torch.logical_and(flattened_durations>d["l"], flattened_durations<=d["vl"]), 3] = 0.80
        categorized[torch.logical_and(flattened_durations>d["l"], flattened_durations<=d["vl"]), 4] = 0.10

        categorized[flattened_durations>d["vl"], 3] = 0.10
        categorized[flattened_durations>d["vl"], 4] = 0.90

    elif duration_category_num == 4:
        d = {"s":6.0,"m":15.0, "l":25.0}

        categorized = torch.zeros(flattened_durations.shape[0],len(d)+1).cuda()
        categorized[flattened_durations<=d["s"], 0] = 0.90
        categorized[flattened_durations<=d["s"], 1] = 0.10

        categorized[torch.logical_and(flattened_durations>d["s"], flattened_durations<=d["m"]), 0] = 0.10
        categorized[torch.logical_and(flattened_durations>d["s"], flattened_durations<=d["m"]), 1] = 0.80
        categorized[torch.logical_and(flattened_durations>d["s"], flattened_durations<=d["m"]), 2] = 0.10

        categorized[torch.logical_and(flattened_durations>d["m"], flattened_durations<=d["l"]), 1] = 0.10
        categorized[torch.logical_and(flattened_durations>d["m"], flattened_durations<=d["l"]), 2] = 0.80
        categorized[torch.logical_and(flattened_durations>d["m"], flattened_durations<=d["l"]), 3] = 0.10

        categorized[flattened_durations>d["l"], 2] = 0.10
        categorized[flattened_durations>d["l"], 3] = 0.90


    if durations.ndim==2 and durations.shape[1]!=1:
        categorized = categorized.reshape(durations.shape[0],durations.shape[1],len(d)+1).cuda()
    return categorized

def decategorize_durations(categorized_durations, duration_category_num, oracle_fixed_time, fixed_time, duration_cat_sampled=None):
    """
        Go from categorized durations to integer value.
    """
    # (this was used for an ablation study)
    # return 10*torch.ones((categorized_durations.shape[0], 1)).float().cuda()
    resh = False
    if categorized_durations.ndim==3:
        B, N, C = categorized_durations.shape
        categorized_durations = categorized_durations.reshape([B*N, C]).cuda()
        resh = True
    assert categorized_durations.ndim==2

    if duration_category_num == 5:
        duration_key_pairs = {0:3, 1:6, 2:12, 3:16, 4:25}
        # duration_key_pairs = {0:5, 1:8, 2:12, 3:16, 4:25}

    elif duration_category_num == 4:
        duration_key_pairs = {0:6, 1:10, 2:16, 3:25}

    if oracle_fixed_time:
        durations = fixed_time*torch.ones((categorized_durations.shape[0], 1)).cuda()
        return durations 

    #duration_key_pairs = {0:3, 1:6, 2:10, 3:15, 4:20, 5:25}
    durations = torch.zeros((categorized_durations.shape[0], 1)).cuda()
    if duration_cat_sampled is None:
        duration_labels = torch.argmax(categorized_durations, dim=1)
    else:
        duration_labels = duration_cat_sampled
        
    for key, val in duration_key_pairs.items():
        durations[duration_labels == key]=val
    
    if resh:
        durations = durations.reshape([B, N, 1])
 
    return durations
