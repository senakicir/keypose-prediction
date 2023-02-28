import numpy as np
import torch
from ..utils import AccumLoss
from ..data_utils import get_body_members, rotmat2euler_tensor, expmap2rotmat_tensor
from ..forward_kinematics import seq_to_angles_transformer

import torch.nn as nn

from scipy.spatial.distance import pdist

class Loss_Function_Manager_Action_Recognition(object):
    def __init__(self, actions,ar_seq_len, only_motion=False, dataset="h36m", prefix =None):
        if prefix is None:
            self.prefix =  "_omac" if only_motion else ""
        else:
            self.prefix =  prefix + "_omac" if only_motion else ""


        self.num_classes = len(actions)
        self.actions = actions
        self.ar_seq_len=ar_seq_len
        self.loss_dict_batch = {"ce_loss":AccumLoss(),self.prefix+"accuracy":AccumLoss(), 
                                self.prefix+"accuracy2":AccumLoss(), self.prefix+"accuracy3":AccumLoss(),
                                self.prefix+"accuracy5":AccumLoss(), "global_mean":AccumLoss(),
                                "kl_gen_gt":AccumLoss(), 
                                "kl_gt_gen":AccumLoss(), 
                                "diversity": AccumLoss()}
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum")
        self.conf_mat= torch.zeros((self.num_classes, self.num_classes)).float()

        body_members = get_body_members(dataset)
        self.angle_trans_function = seq_to_angles_transformer(body_members)


    def reset_all(self):
        self.loss_dict_batch = {"ce_loss":AccumLoss(), self.prefix+"accuracy":AccumLoss(), 
                                self.prefix+"accuracy2":AccumLoss(), self.prefix+"accuracy3":AccumLoss(),
                                self.prefix+"accuracy5":AccumLoss(),  "global_mean":AccumLoss(),
                                "kl_gen_gt":AccumLoss(), 
                                "kl_gt_gen":AccumLoss(),
                                "diversity": AccumLoss()}
        self.conf_mat=torch.zeros((self.num_classes, self.num_classes))

    def update_loss(self, outputs, labels, batch_size):
        loss = self.cross_entropy(outputs, labels)
        self.loss_dict_batch["ce_loss"].update(loss.detach(), batch_size)
        return loss

    def update_conf_mat_and_acc(self, outputs, labels, batch_size):
        detached_outputs = outputs.detach()
        _, topkinds = torch.topk(detached_outputs, 5, dim=1)
        pred_labels = topkinds[:,0]

        conf_mat = torch.zeros((self.num_classes, self.num_classes)).float()
        # #add this confusion_matrix on top of ones from other batches
        self.conf_mat = self.conf_mat+conf_mat

        ## accuracy
        acc = torch.sum(pred_labels == labels).cpu()
        self.loss_dict_batch[self.prefix+"accuracy"].update(acc.float(), batch_size)

        ## accuracy top 2
        acctop2 = acc + torch.sum(topkinds[:,1] == labels).cpu()
        self.loss_dict_batch[self.prefix+"accuracy2"].update(acctop2.float(), batch_size)

        ## accuracy top 3
        acctop3 = acctop2 + torch.sum(topkinds[:,2] == labels).cpu()
        self.loss_dict_batch[self.prefix+"accuracy3"].update(acctop3.float(), batch_size)

        ## accuracy top 5
        acctop4 = acctop3 + torch.sum(topkinds[:,3] == labels).cpu()
        acctop5 = acctop4 + torch.sum(topkinds[:,4] == labels).cpu()
        self.loss_dict_batch[self.prefix+"accuracy5"].update(acctop5.float(), batch_size)

    def return_average_losses(self):
        my_losses = {}
        for key, value in self.loss_dict_batch.items():
            my_losses[key] = value.avg

        return my_losses

    def compute_ent_metrics(self, gt_seqs, seqs, format='coords'):
        batch_size = seqs.shape[0]
        seq_len = seqs.shape[1]
        
        assert seqs.shape[3] ==3
        assert seq_len == 125, str(seq_len)

        gt_seqs_tmp = gt_seqs.cpu().numpy()
        seqs_tmp = seqs.cpu().numpy()

        gt_seqs_tmp = gt_seqs_tmp.reshape([batch_size*seq_len, -1, 3])
        seqs_tmp =  seqs_tmp.reshape([batch_size*seq_len, -1, 3])       

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
        
        seqs_kl_gen_gt = np.mean(np.mean(seqs_ps_global * np.log(seqs_ps_global / gt_seqs_ps_global), axis=2), axis=1)
        seqs_kl_gt_gen = np.mean(np.mean(gt_seqs_ps_global * np.log(gt_seqs_ps_global / seqs_ps_global), axis=2), axis=1)
        
        self.loss_dict_batch["kl_gen_gt"].update(seqs_kl_gen_gt.sum(), batch_size)
        self.loss_dict_batch["kl_gt_gen"].update(seqs_kl_gt_gen.sum(), batch_size)

    """metrics"""

    def compute_diversity(self, pred, *args):
        if pred.shape[0] == 1:
            return 0.0
        batch_size = pred.shape[1] 
        for batch_ind in range(batch_size):
            dist = pdist(pred[:, batch_ind, :, :].reshape([pred.shape[0], -1]))
            diversity = dist.mean().item()
            self.loss_dict_batch["diversity"].update(diversity, 1)
        return diversity


def compute_ade(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2).mean(axis=1)
    return dist.min()


def compute_fde(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1]
    return dist.min()


def compute_mmade(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_ade(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist


def compute_mmfde(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_fde(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist

