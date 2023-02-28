#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import torch
from torch.autograd.variable import Variable
import os
import yaml

from . import forward_kinematics

from . import keypose_extract as keypose_module

splits = {"train":0, "test": 2, "val":1, "all":3}



def rotmat2euler(R):
    """
    Converts a rotation matrix to Euler angles
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

    Args
      R: a 3x3 rotation matrix
    Returns
      eul: a 3x1 Euler angle representation of R
    """
    if R[0, 2] == 1 or R[0, 2] == -1:
        # special case
        E3 = 0  # set arbitrarily
        dlta = np.arctan2(R[0, 1], R[0, 2]);

        if R[0, 2] == -1:
            E2 = np.pi / 2;
            E1 = E3 + dlta;
        else:
            E2 = -np.pi / 2;
            E1 = -E3 + dlta;

    else:
        E2 = -np.arcsin(R[0, 2])
        E1 = np.arctan2(R[1, 2] / np.cos(E2), R[2, 2] / np.cos(E2))
        E3 = np.arctan2(R[0, 1] / np.cos(E2), R[0, 0] / np.cos(E2))

    eul = np.array([E1, E2, E3]);
    return eul


def rotmat2quat(R):
    """
    Converts a rotation matrix to a quaternion
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

    Args
      R: 3x3 rotation matrix
    Returns
      q: 1x4 quaternion
    """
    rotdiff = R - R.T;

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2;
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps);

    costheta = (np.trace(R) - 1) / 2;

    theta = np.arctan2(sintheta, costheta);

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R));


def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

    Args
      r: 1x3 exponential map
    Returns
      R: 3x3 rotation matrix
    """
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x);
    return R



def expmap2rotmat_tensor(r):
    """
    Converts an exponential map angle to a rotation matrix
    Tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      r: (..., 3) exponential map Tensor
    Returns:
      R: (..., 3, 3) rotation matrix Tensor
    """
    base_shape = [int(d) for d in r.shape][:-1]
    zero_dim = np.zeros(base_shape)

    theta = np.sqrt(np.sum(np.square(r), axis=-1, keepdims=True) + 1e-8)
    r0 = r / theta

    r0x = np.reshape(
        np.stack([zero_dim, -1.0 * r0[..., 2], r0[..., 1],
                  zero_dim, zero_dim, -1.0 * r0[..., 0],
                  zero_dim, zero_dim, zero_dim], axis=-1),
        base_shape + [3, 3]
    )
    trans_dims = list(range(len(r0x.shape)))
    trans_dims[-1], trans_dims[-2] = trans_dims[-2], trans_dims[-1]
    r0x = r0x - np.transpose(r0x, trans_dims)

    tile_eye = np.tile(np.reshape(np.eye(3), [1 for _ in base_shape] + [3, 3]), base_shape + [1, 1])
    theta = np.expand_dims(theta, axis=-1)

    R = tile_eye + np.sin(theta) * r0x + (1.0 - np.cos(theta)) * np.matmul(r0x, r0x)
    return R

def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

    Args
      q: 1x4 quaternion
    Returns
      r: 1x3 exponential map
    Raises
      ValueError if the l2 norm of the quaternion is not close to 1
    """
    if (np.abs(np.linalg.norm(q) - 1) > 1e-3):
        raise (ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]

    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0

    r = r0 * theta
    return r

def quat2expmap_tensor(q):
    """
    Converts a quaternion to an exponential map
    Tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
        q: (..., 4) quaternion Tensor
    Returns:
        r: (..., 3) exponential map Tensor
    Raises:
        ValueError if the l2 norm of the quaternion is not close to 1
    """
    # if (np.abs(np.linalg.norm(q)-1)>1e-3):
    # raise(ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.sqrt(np.sum(np.square(q[..., 1:]), axis=-1, keepdims=True) + 1e-8)
    coshalftheta = np.expand_dims(q[..., 0], axis=-1)

    r0 = q[..., 1:] / sinhalftheta
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2*np.pi, 2*np.pi)

    condition = np.greater(theta, np.pi)
    theta = np.where(condition, 2 * np.pi - theta, theta)
    r0 = np.where(np.tile(condition, [1 for _ in condition.shape[:-1]] + [3]), -r0, r0)
    r = r0 * theta

    return r


def rotmat2quat_torch(R):
    """
    Converts a rotation matrix to a quaternion
    Tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      R: (..., 3, 3) rotation matrix Tensor
    Returns:
      q: (..., 4) quaternion Tensor
    """
    trans_dims = list(range(len(R.shape)))
    trans_dims[-1], trans_dims[-2] = trans_dims[-2], trans_dims[-1]
    rotdiff = R - np.transpose(R, trans_dims)

    r = np.stack([-rotdiff[..., 1, 2], rotdiff[..., 0, 2], -rotdiff[..., 0, 1]], axis=-1)
    rnorm = np.sqrt(np.sum(np.square(r), axis=-1, keepdims=True) + 1e-8)
    sintheta = rnorm / 2.0
    r0 = r / rnorm

    costheta = np.expand_dims((np.trace(R) - 1.0) / 2.0, axis=-1)

    theta = np.arctan2(sintheta, costheta)

    q = np.concatenate([np.cos(theta / 2),  r0 * np.sin(theta / 2)], axis=-1)

    return q


def rotmat2euler_tensor(R):
    """
    Converts a rotation matrix to Euler angles
    Tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      R: a (..., 3, 3) rotation matrix Tensor
    Returns:
      eul: a (..., 3) Euler angle representation of R
    """
    base_shape = [int(d) for d in R.shape][:-2]
    zero_dim = np.zeros(base_shape)
    one_dim = np.ones(base_shape)

    econd0 = np.equal(R[..., 0, 2], one_dim)
    econd1 = np.equal(R[..., 0, 2], -1.0 * one_dim)
    econd = np.logical_or(econd0, econd1)

    e2 = np.where(
        econd,
        np.where(econd1, one_dim * np.pi / 2.0, one_dim * -np.pi / 2.0),
        -np.arcsin(R[..., 0, 2])
    )
    e1 = np.where(
        econd,
        np.arctan2(R[..., 1, 2], R[..., 0, 2]),
        np.arctan2(R[..., 1, 2] / np.cos(e2), R[..., 2, 2] / np.cos(e2))
    )
    e3 = np.where(
        econd,
        zero_dim,
        np.arctan2(R[..., 0, 1] / np.cos(e2), R[..., 0, 0] / np.cos(e2))
    )

    eul = np.stack([e1, e2, e3], axis=-1)
    return eul
def quaternion_between(u, v):
    """
    Finds the quaternion between two tensor of 3D vectors.
    See:
    http://www.euclideanspace.com/maths/algebra/vectors/angleBetween/index.htm
    Args:
        u: A `np.array` of rank R, the last dimension must be 3.
        v: A `np.array` of rank R, the last dimension must be 3.
    Returns:
        A `Quaternion` of Rank R with the last dimension being 4.
        returns 1, 0, 0, 0 quaternion if either u or v is 0, 0, 0
    Raises:
        ValueError, if the last dimension of u and v is not 3.
    """
    if u.shape[-1] != 3 or v.shape[-1] != 3:
        raise ValueError("The last dimension of u and v must be 3.")

    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")

    def _vector_batch_dot(a, b):
        return np.sum(np.multiply(a, b), axis=-1, keepdims=True)

    def _length_2(a):
        return np.sum(np.square(a), axis=-1, keepdims=True)

    def _normalize(a):
        norm_quart= a / np.sqrt(_length_2(a))
        return norm_quart

    base_shape = [int(d) for d in u.shape]
    base_shape[-1] = 1
    zero_dim = np.zeros(base_shape)
    one_dim = np.ones(base_shape)
    w = np.sqrt(_length_2(u) * _length_2(v)) + _vector_batch_dot(u, v)

    q = np.where(
            np.tile(np.equal(np.sum(u, axis=-1, keepdims=True), zero_dim), [1 for _ in u.shape[:-1]] + [4]),
            np.concatenate([one_dim, u], axis=-1),
            np.where(
                np.tile(np.equal(np.sum(v, axis=-1, keepdims=True), zero_dim), [1 for _ in u.shape[:-1]] + [4]),
                np.concatenate([one_dim, v], axis=-1),
                np.where(
                    np.tile(np.less(w, 1e-4), [1 for _ in u.shape[:-1]] + [4]),
                    np.concatenate([zero_dim, np.stack([-u[..., 2], u[..., 1], u[..., 0]], axis=-1)], axis=-1),
                    np.concatenate([w, np.cross(u, v)], axis=-1)
                )
            )
        )

    return _normalize(q)

def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore, actions, one_hot):
    """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

    Args
      normalizedData: nxd matrix with normalized data
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
      actions: list of strings with the encoded actions
      one_hot: whether the data comes with one-hot encoding
    Returns
      origData: data originally used to
    """
    T = normalizedData.shape[0]
    D = data_mean.shape[0]

    origData = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = []
    for i in range(D):
        if i in dimensions_to_ignore:
            continue
        dimensions_to_use.append(i)
    dimensions_to_use = np.array(dimensions_to_use)

    if one_hot:
        origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
    else:
        origData[:, dimensions_to_use] = normalizedData

    # potentially ineficient, but only done once per experiment
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    origData = np.multiply(origData, stdMat) + meanMat
    return origData


def revert_output_format(poses, data_mean, data_std, dim_to_ignore, actions, one_hot):
    """
    Converts the output of the neural network to a format that is more easy to
    manipulate for, e.g. conversion to other format or visualization

    Args
      poses: The output from the TF model. A list with (seq_length) entries,
      each with a (batch_size, dim) output
    Returns
      poses_out: A tensor of size (batch_size, seq_length, dim) output. Each
      batch is an n-by-d sequence of poses.
    """
    seq_len = len(poses)
    if seq_len == 0:
        return []

    batch_size, dim = poses[0].shape

    poses_out = np.concatenate(poses)
    poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
    poses_out = np.transpose(poses_out, [1, 0, 2])

    poses_out_list = []
    for i in xrange(poses_out.shape[0]):
        poses_out_list.append(
            unNormalizeData(poses_out[i, :, :], data_mean, data_std, dim_to_ignore, actions, one_hot))

    return poses_out_list


def readCSVasFloat(filename):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    returnArray = []
    my_file = open(filename)
    lines = my_file.readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    my_file.close()
    return returnArray


def normalize_data(data, data_mean, data_std, dim_to_use, actions, one_hot):
    """
    Normalize input data by removing unused dimensions, subtracting the mean and
    dividing by the standard deviation

    Args
      data: nx99 matrix with data to normalize
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dim_to_use: vector with dimensions used by the model
      actions: list of strings with the encoded actions
      one_hot: whether the data comes with one-hot encoding
    Returns
      data_out: the passed data matrix, but normalized
    """
    data_out = {}
    nactions = len(actions)

    if not one_hot:
        # No one-hot encoding... no need to do anything special
        for key in data.keys():
            data_out[key] = np.divide((data[key] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]

    else:
        # TODO hard-coding 99 dimensions for un-normalized human poses
        for key in data.keys():
            data_out[key] = np.divide((data[key][:, 0:99] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]
            data_out[key] = np.hstack((data_out[key], data[key][:, -nactions:]))

    return data_out


def normalization_stats(completeData):
    """"
    Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33

    Args
      completeData: nx99 matrix with data to normalize
    Returns
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
      dimensions_to_use: vector with dimensions used by the model
    """
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use



actions_h36m = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]

actions_cmu = ["basketball", "basketball_signal", "directing_traffic", 
                "jumping", "soccer", "washwindow"]
                #  "soccer", "washwindow"]


def load_mean_poses(data_dir):
    mp=torch.load(data_dir+"/mean_poses.pkl")
    return mp

def define_actions(action, dataset, overfitting_exp = False):
    """
    Define the list of actions we are using.

    Args
      action: String with the passed action. Could be "all"
    Returns
      actions: List of strings of actions
    Raises
      ValueError if the action is not included in H3.6M
    """
    if dataset =="h36m":
        if overfitting_exp:
            return ["walking"]

        if action in actions_h36m:
            return [action]

        if action == "all":
            return actions_h36m

        if action == "all_srnn":
            return ["walking", "eating", "smoking", "discussion"]
        
        if action == "all_walking":
            return ["walking", "walkingtogether", "walkingdog"]

        raise (ValueError, "Unrecognized action: %d" % action)
    
    elif dataset =="cmu":
        if overfitting_exp:
            return ["walking"]

        if action in actions_cmu:
            return [action]

        if action == "all":
            return actions_cmu


        raise (ValueError, "Unrecognized action: %d" % action)



def rotmat2euler_torch(R):
    """
    Converts a rotation matrix to euler angles
    batch pytorch version ported from the corresponding numpy method above

    :param R:N*3*3
    :return: N*3
    """
    n = R.data.shape[0]
    eul = Variable(torch.zeros(n, 3).float()).cuda()
    idx_spec1 = (R[:, 0, 2] == 1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    idx_spec2 = (R[:, 0, 2] == -1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    if len(idx_spec1) > 0:
        R_spec1 = R[idx_spec1, :, :]
        eul_spec1 = Variable(torch.zeros(len(idx_spec1), 3).float()).cuda()
        eul_spec1[:, 2] = 0
        eul_spec1[:, 1] = -np.pi / 2
        delta = torch.atan2(R_spec1[:, 0, 1], R_spec1[:, 0, 2])
        eul_spec1[:, 0] = delta
        eul[idx_spec1, :] = eul_spec1

    if len(idx_spec2) > 0:
        R_spec2 = R[idx_spec2, :, :]
        eul_spec2 = Variable(torch.zeros(len(idx_spec2), 3).float()).cuda()
        eul_spec2[:, 2] = 0
        eul_spec2[:, 1] = np.pi / 2
        delta = torch.atan2(R_spec2[:, 0, 1], R_spec2[:, 0, 2])
        eul_spec2[:, 0] = delta
        eul[idx_spec2] = eul_spec2

    idx_remain = np.arange(0, n)
    idx_remain = np.setdiff1d(np.setdiff1d(idx_remain, idx_spec1), idx_spec2).tolist()
    if len(idx_remain) > 0:
        R_remain = R[idx_remain, :, :]
        eul_remain = Variable(torch.zeros(len(idx_remain), 3).float()).cuda()
        eul_remain[:, 1] = -torch.asin(R_remain[:, 0, 2])
        eul_remain[:, 0] = torch.atan2(R_remain[:, 1, 2] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 2, 2] / torch.cos(eul_remain[:, 1]))
        eul_remain[:, 2] = torch.atan2(R_remain[:, 0, 1] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 0, 0] / torch.cos(eul_remain[:, 1]))
        eul[idx_remain, :] = eul_remain

    return eul


def rotmat2quat_torch(R):
    """
    Converts a rotation matrix to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N * 3 * 3
    :return: N * 4
    """
    rotdiff = R - R.transpose(1, 2)
    r = torch.zeros_like(rotdiff[:, 0])
    r[:, 0] = -rotdiff[:, 1, 2]
    r[:, 1] = rotdiff[:, 0, 2]
    r[:, 2] = -rotdiff[:, 0, 1]
    r_norm = torch.norm(r, dim=1)
    sintheta = r_norm / 2
    r0 = torch.div(r, r_norm.unsqueeze(1).repeat(1, 3) + 0.00000001)
    t1 = R[:, 0, 0]
    t2 = R[:, 1, 1]
    t3 = R[:, 2, 2]
    costheta = (t1 + t2 + t3 - 1) / 2
    theta = torch.atan2(sintheta, costheta)
    q = Variable(torch.zeros(R.shape[0], 4)).float().cuda()
    q[:, 0] = torch.cos(theta / 2)
    q[:, 1:] = torch.mul(r0, torch.sin(theta / 2).unsqueeze(1).repeat(1, 3))

    return q


def expmap2quat_torch(exp):
    """
    Converts expmap to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N*3
    :return: N*4
    """
    theta = torch.norm(exp, p=2, dim=1).unsqueeze(1)
    v = torch.div(exp, theta.repeat(1, 3) + 0.0000001)
    sinhalf = torch.sin(theta / 2)
    coshalf = torch.cos(theta / 2)
    q1 = torch.mul(v, sinhalf.repeat(1, 3))
    q = torch.cat((coshalf, q1), dim=1)
    return q


def expmap2rotmat_torch(r, is_cuda=True):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    m = torch.mul(
            torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
            (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    if is_cuda:
        R = Variable(torch.eye(3, 3).repeat(n, 1, 1)).float().cuda() + m
    else:
        R = Variable(torch.eye(3, 3).repeat(n, 1, 1)).float() + m
    return R


def expmap2xyz_torch(expmap, is_cuda=True):
    """
    convert expmaps to joint locations
    :param expmap: N*99
    :return: N*32*3
    """
    parent, offset, rotInd, expmapInd = forward_kinematics._some_variables()
    xyz = forward_kinematics.fkl_torch(expmap, parent, offset, rotInd, expmapInd, is_cuda)
    return xyz


def expmap2xyz_torch_cmu(expmap, is_cuda):
    parent, offset, rotInd, expmapInd = forward_kinematics._some_variables_cmu()
    xyz = forward_kinematics.fkl_torch(expmap, parent, offset, rotInd, expmapInd, is_cuda)
    return xyz


def find_keyposes_in_batch(keypose_list, selected_indices):
    keyposes_in_batch = [keypose for keypose in keypose_list if keypose["loc"] in selected_indices]
    keyposes_in_batch[0]["offset"] = keyposes_in_batch[0]["loc"]-selected_indices[0]
    return keyposes_in_batch

def find_sequence_in_keyposes(keypose_batch, the_sequence, input_kp, output_seq_n):
    initial_keypose_loc = keypose_batch[0]["loc"]
    final_keypose_loc = keypose_batch[-1]["loc"]
    input_keypose_ending_loc = keypose_batch[input_kp-1]["loc"]
    kp_seq_len = final_keypose_loc-input_keypose_ending_loc ##future frames
    ## if number of future frames we have is larger than what we need to predict
    if kp_seq_len >= output_seq_n:
        subsampled_seq = the_sequence[initial_keypose_loc:final_keypose_loc, :]
        predicted_seq_len = kp_seq_len
    else:
        if input_keypose_ending_loc+output_seq_n <= the_sequence.shape[0]:
            subsampled_seq = the_sequence[initial_keypose_loc:input_keypose_ending_loc+output_seq_n, :]
            predicted_seq_len = output_seq_n
        else:
            return None, 0, 0

    seq_len = subsampled_seq.shape[0]
    return subsampled_seq, seq_len, predicted_seq_len


def find_keyposes_in_sequence(keypose_locations, test_index_point, input_kp, output_kp):
    first_future_index = len(keypose_locations)
    kp_ind_tensor = torch.zeros([input_kp+output_kp]).long()
    for keypose_ind, keypose_loc in enumerate(keypose_locations):
        if keypose_loc > test_index_point:
            first_future_index = keypose_ind
            break
    last_past_idx = first_future_index-1

    kp_list_len = 0
    keypose_ind = 0
    offset = None

    if not last_past_idx-input_kp>=0:
        while kp_list_len<(input_kp-last_past_idx-1):
            kp_ind_tensor[kp_list_len] =  0
            kp_list_len += 1

    while kp_list_len != input_kp + output_kp:
        if keypose_ind < len(keypose_locations):
            if keypose_ind >= first_future_index-input_kp:
                kp_ind_tensor[kp_list_len] = keypose_ind
                kp_list_len += 1

            if keypose_ind == last_past_idx:
                offset = test_index_point-keypose_locations[keypose_ind]
            keypose_ind += 1
        
        #keep adding same keypose (the end one)
        elif keypose_ind == len(keypose_locations):
            kp_ind_tensor[kp_list_len] = len(keypose_locations)-1
            kp_list_len += 1

    assert offset is not None
    if offset > 0:
        import pdb; pdb.set_trace()
    return kp_ind_tensor, offset

def find_future_keyposes_in_sequence(keypose_locations, test_index_point, output_kp):
    kp_ind_tensor = torch.zeros([output_kp]).long()
    kp_list_len = 0
    for keypose_ind, keypose_loc in enumerate(keypose_locations):
        if keypose_loc > test_index_point:
            kp_ind_tensor[kp_list_len] = keypose_ind
            kp_list_len += 1
        if kp_list_len == output_kp:
            break
    offset = keypose_locations[kp_ind_tensor[0]]-test_index_point
    if kp_list_len < output_kp:
        kp_ind_tensor[kp_list_len:] = len(keypose_locations)-1
        
    return kp_ind_tensor, offset


def find_keyposes_in_sequence_old(keypose_list, the_sequence, test_index_point, input_kp, output_kp):
    first_future_index = 0
    for keypose_ind, keypose in enumerate(keypose_list):
        if keypose["loc"] > test_index_point:
            first_future_index = keypose_ind
            break
    
    if not (first_future_index-input_kp>=0):
        return None, None

    kp_list = []
    kp_list_len = 0
    keypose_ind = 0
    
    while kp_list_len != input_kp + output_kp:
        if keypose_ind < len(keypose_list):
            keypose = keypose_list[keypose_ind]
            if keypose_ind >= first_future_index-input_kp:
                kp_list.append(keypose)
                kp_list_len += 1

            if keypose_ind == first_future_index-1:
                offset = test_index_point-keypose["loc"]

            keypose_ind += 1
        
        #keep adding same keypose (the end one)
        elif keypose_ind == len(keypose_list):
            keypose = keypose_list[-1]
            kp_list.append(keypose)
            kp_list_len += 1

    return kp_list, offset

def cluster_keyposes(cluster_n, keypose_dir, all_subjs, training_subjs):
    ## CLUSTERING:
    n_clusters = cluster_n
    clf = KMeans(n_clusters=n_clusters)

    training_data = np.zeros((0, len(dimensions_to_use)))
    test_data = np.zeros((0,len(dimensions_to_use)))
    validation_data = np.zeros((0,len(dimensions_to_use)))

    # load training data
    for subj in training_subjs:
        for act in acts:
            for subact in [1,2]:
                keypose_filename = '{0}/S{1}/{2}_{3}.yaml'.format(keypose_dir, subj, act, subact)
                if (os.path.isfile(keypose_filename)):
                    data = {}
                    with open(keypose_filename, 'r') as yaml_file:
                        data_loaded =  yaml.load(yaml_file, Loader=yaml.FullLoader)
                        keyposes = np.array(data_loaded["keyposes"])
                        training_data = np.concatenate([training_data, keyposes.transpose()])


    # cluster training data and save cluster centers
    clf.fit(training_data)
    cluster_centers = {}
    for cluster in range(n_clusters):
        cluster_centers[cluster] = (clf.cluster_centers_[cluster]).tolist()

    keypose_filename = '{0}/cluster_centers_{1}.yaml'.format(keypose_dir, cluster_n)
    with open(keypose_filename, 'w') as yaml_file:
        yaml.dump(cluster_centers, yaml_file)

    # predict on all data
    for subj in all_subjs:
        for act in acts:
            for subact in [1,2]:
                keypose_filename = '{0}/S{1}/{2}_{3}.yaml'.format(opt.keypose_dir, subj, act, subact)
                if (os.path.isfile(keypose_filename)):
                    data = {}
                    with open(keypose_filename, 'r') as yaml_file:
                        data_loaded =  yaml.load(yaml_file, Loader=yaml.FullLoader)
                        keyposes = np.array(data_loaded["keyposes"])
                        labels = clf.predict(keyposes.transpose())
                        data["keyposes"] = data_loaded["keyposes"]
                        data["loc"] = data_loaded["loc"]
                        data["labels"] = labels.tolist()
                    with open(keypose_filename, 'w') as keypose_yaml_file:
                        yaml.dump(data, keypose_yaml_file)


# def find_indices_256(frame_num1, frame_num2):
#     """
#     Adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478

#     which originaly from
#     In order to find the same action indices as in SRNN.
#     https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
#     """

#     # Used a fixed dummy seed, following
#     # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29


#     SEED = 123456725690
#     rng = np.random.RandomState(SEED)

#     T1 = frame_num1 - (175+125+1)
#     T2 = frame_num2 - (175+125+1)

#     index_list1 = []
#     index_list2 = []
#     for _ in np.arange(0, 128):
#         idx_ran1 = rng.randint(16, T1)
#         idx_ran2 = rng.randint(16, T2)
#         index_list1.append(idx_ran1 + 175)
#         index_list2.append(idx_ran2 + 175)
#     return index_list1, index_list2


def find_indices(frame_num, num_ind, short_seq=False):
    """
    Adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478

    which originaly from
    In order to find the same action indices as in SRNN.
    https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29

    SEED = 12345
    rng = np.random.RandomState(SEED)
    if short_seq:
        T1 = frame_num - (150+25+1)
    else:
        T1 = frame_num - (175+125+1)

    index_list1 = []
    for _ in np.arange(0, num_ind):
        idx_ran1 = rng.randint(16, T1)
        index_list1.append(idx_ran1 + 175) #this is middle frame
    return index_list1


def find_indices_kp(kp_locs, seq_len):
    """
    Adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478

    which originaly from
    In order to find the same action indices as in SRNN.
    https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29

    input_n_margin = 175
    output_n_margin = 125

    SEED = 123456
    rng = np.random.RandomState(SEED)

    begin_n = 0
    end_n = len(kp_locs)-1
    for m in range(len(kp_locs)):
        if kp_locs[m] > input_n_margin:
            begin_n = m
            break
    for m in range(len(kp_locs)):
        if kp_locs[m] > seq_len-output_n_margin:
            end_n = m
            break

    index_list1 = []
    for _ in np.arange(0, 128):
        idx_ran1 = rng.randint(begin_n, end_n)
        index_list1.append(kp_locs[idx_ran1])
    return index_list1


def find_indices_srnn(frame_num1, frame_num2, seq_len, input_n):
    """
    Adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478

    which originaly from
    In order to find the same action indices as in SRNN.
    https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = frame_num1 - 150
    T2 = frame_num2 - 150  # seq_len

    index_list1 = []
    index_list2 = []
    for _ in np.arange(0, 4):
        idx_ran1 = rng.randint(16, T1)
        idx_ran2 = rng.randint(16, T2)
        index_list1.append(idx_ran1 + 50)
        index_list2.append(idx_ran2 + 50)
    return index_list1, index_list2

def kp_dict_to_input(kp_seq, n_clusters, kp_dim):
    data = torch.zeros([len(kp_seq), n_clusters]).float()
    durs = torch.zeros([len(kp_seq)]).int()
    locs = torch.zeros([len(kp_seq)]).int()
    vals = torch.zeros([len(kp_seq), kp_dim])
    labels = torch.zeros([len(kp_seq)]).long()

    for ind in range(0, len(kp_seq)):
        kp = kp_seq[ind]
        transition_time = kp["duration"]
        durs[ind] = transition_time
        locs[ind] = kp["loc"]
        vals[ind] = torch.from_numpy(kp["keyposes"]).float()
        data[ind, :] = torch.from_numpy(kp["naive_distances"]).float()
        labels[ind] = kp["naive_labels"]

    return vals, data, locs, durs, labels


def kp_dict_to_one_hot(list_of_kp_seq, n_keyposes, n_clusters):
    data = torch.zeros([len(list_of_kp_seq), n_keyposes, n_clusters+1]).float()
    labels = torch.zeros([len(list_of_kp_seq), n_keyposes]).long()
    
    for kp_seq_ind, kp_seq in enumerate(list_of_kp_seq):
        for ind, kp in enumerate(kp_seq):
            data[kp_seq_ind, ind, -1]  = kp["duration"]            
            data[kp_seq_ind, ind, :-1] = torch.from_numpy(kp["naive_distances"]).float()
            labels[kp_seq_ind, ind] = kp["naive_labels"]

 
    return data, labels

def one_hot_to_kp_dict(labels, durations):
    batch_size, seq_length, n_clusters = labels.shape
    kp_batches = []
    for batch in range(batch_size):
        kp_seq = []
        for ind in range(seq_length):
            kp_dict = {}
            kp_dict["label"] = torch.argmax(labels[batch,ind])
            kp_dict["duration"] = durations[batch,ind] 
            kp_seq.append(kp_dict)
        kp_batches.append(kp_seq)
    return kp_batches


#Computes the eucledean distance between each keypose and each cluster center
def compute_distances(keyposes, cluster_centers) :
    distances = np.array([])
    for pose in keyposes :
        pose_distance = np.array([])
        for idx in range(len(cluster_centers)) :
            pose_distance = np.append(pose_distance, np.linalg.norm(cluster_centers[idx] - pose))
        distances = np.append(distances, pose_distance)
    return distances




def load_bone_length(subject_list, path_to_keyposes, path_to_dataset, overwrite=True):
    bone_dict = {}
    for subject in subject_list:
        skeleton_filename = '{0}/S{1}/bone_len.yaml'.format(path_to_keyposes, subject)
        if (os.path.isfile(skeleton_filename)) and not overwrite:
            with open(skeleton_filename, 'r') as yaml_file:
                bone_len = np.array(yaml.load(yaml_file, Loader=yaml.FullLoader))
        else:
            ##load all data
            all_poses = np.zeros([0,32*3])
            for act in  define_actions('all'):
                for subact in [1,2]:
                    new_seq, _ =load_human36m_full_data(path_to_dataset, subject, act, subact, sample_rate=2, is_cuda=False)
                    all_poses = np.concatenate([all_poses, new_seq], axis=0)
            transposed_seq = all_poses.reshape((all_poses.shape[0],32,3)).transpose(0,2,1)
            all_bone_len = find_bone_lengths(transposed_seq, all_bone_connections())
            bone_len = np.mean(all_bone_len, axis=0)

            with open(skeleton_filename, 'w') as yaml_file:
                yaml.dump(bone_len.tolist(), yaml_file)
        bone_dict[subject] = bone_len.copy()
    return bone_dict


def process_bone(poses, start_joint, end_joint, old_val, goal_bone_len):
    bone_vec = poses[:, :, end_joint]-old_val
    new_old_val = poses[:, :, end_joint].clone()
    new_bone_vector = goal_bone_len*bone_vec/torch.norm(bone_vec, 2, dim=1).unsqueeze(1)
    poses[:,:,end_joint] = poses[:, :, start_joint] + new_bone_vector     
    return poses, new_old_val

def traverse_body(poses, start_joint, old_val, bone_connections, visited_list, goal_bone_len):
    for count in range(len(bone_connections)):
        list_start_joint, list_end_joint = bone_connections[count]
        if start_joint == list_start_joint and not visited_list[count]:
            visited_list[count] = True
            poses, new_old_val = process_bone(poses, list_start_joint, list_end_joint, old_val, goal_bone_len[count])
            traverse_body(poses, list_end_joint, new_old_val, bone_connections, visited_list, goal_bone_len)
    return poses


def find_bone_lengths(poses, bone_connections):
    ##poses are of format (N,3,D)
    bone_len = torch.zeros([poses.shape[0], len(bone_connections)])
    bone_connections = np.array(bone_connections)
    i = bone_connections[:,0]
    j = bone_connections[:,1]
    bone_len = torch.norm(poses[:,:,i]-poses[:,:,j], 2, dim=1)
    assert bone_len.shape==(poses.shape[0], len(bone_connections))
    return bone_len

def convert_to_skeleton(poses, goal_bone_len, bone_connections):
    initial_poses = poses.clone()

    #make skeleton independent
    visited_list = [False]*len(bone_connections)

    #convert poses (this is a recursive function)
    converted_poses = traverse_body(initial_poses, bone_connections[0][0], initial_poses[:, :, bone_connections[0][0]], bone_connections, visited_list, goal_bone_len)
    return converted_poses

def joint_num(dataset="h36m"):
    if dataset == "h36m":
        return 32
    elif dataset=="cmu":
        return 38

def nonmoving_joints(dataset="h36m"):
    ## 16 20 23 24 28 31 are duplicate joints
    ## 0 and 11 are both hip
    ## 0 is left hip and 6 is right hip i guess they also do not move?
    if dataset == "h36m":
        return np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
    elif dataset == "cmu":
        return np.array([0, 1, 2, 7, 8, 13, 16, 20, 29, 24, 27, 33, 36])

def dimension_info(dataset="h36m"):
    joint_to_ignore = nonmoving_joints(dataset)
    dimensions_to_ignore = convert_joints_to_dimensions(joint_to_ignore)
    num_joints = joint_num(dataset)
    dimensions_to_use = np.setdiff1d(np.arange(num_joints*3), dimensions_to_ignore)
    return joint_to_ignore, dimensions_to_ignore, dimensions_to_use


def all_bone_connections(dataset="h36m"):
    if dataset == "h36m":
        i = np.array([0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 12,13,14,13,17,18,13,25,26,19,19,27,27])
        j = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,12,13,14,15,17,18,19,25,26,27,21,22,29,30])
    elif dataset == "cmu":
        i = np.array([1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14, 15, 16, 17, 18, 19, 
                16, 21, 22, 23, 24, 25, 26, 24, 28, 16, 30, 31, 32, 33, 34, 35, 33, 37]) - 1
        j = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  
                20, 21, 22, 23, 25, 24, 26, 27, 28, 29, 30, 31, 32,33, 34, 35, 36, 37, 38]) - 1
    return list(zip(i.tolist(), j.tolist()))

def all_bone_connections_after_subsampling(part="all", dataset="h36m"):
    if dataset == "h36m":
        if part == "all":
            i = np.array([0, 1, 2, 4, 5, 6, 8, 9,  10,  9, 12, 13,  9, 17, 18, 14, 19, 8, 8])
            j = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18, 19, 15, 21, 0, 4])
        if part == "upper":
            i = np.array([8, 9,  10,  9, 12, 13, 14, 9,  17, 18, 19])
            j = np.array([9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21])
        if part == "lower":
            i = np.array([0, 1, 2, 4, 5, 6])
            j = np.array([1, 2, 3, 5, 6, 7])
    elif dataset == "cmu":
        i = np.array([0, 1, 2, 4, 5, 6, 8, 9,  10, 11,9, 13, 14, 15, 16,15,9,19, 20,21,22,21, 8, 8])
        j = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 12,13, 14, 15, 16, 17,18,19,20,21,22,23,24, 0, 4])
    return list(zip(i.tolist(), j.tolist()))

def lower_body_joints_after_subsampling():
    return np.array([0,1,2,3,4,5,6,7])

def upper_body_joints_after_subsampling():
    return np.setdiff1d(np.arange(22), lower_body_joints_after_subsampling())

def convert_joints_to_dimensions(joint_array):
    return np.concatenate((joint_array * 3, joint_array * 3 + 1, joint_array * 3 + 2))

def set_equal_joints(pose, dataset): 
    new_pose = pose.clone()
    joint_equal, joint_to_ignore = equal_joint_pairs(dataset)
    index_to_ignore = convert_joints_to_dimensions(joint_to_ignore)
    index_to_equal =  convert_joints_to_dimensions(joint_equal)
    new_pose[:, :, index_to_ignore] = new_pose[:, :, index_to_equal]
    return new_pose

def equal_joint_pairs(dataset):
    if dataset == "h36m":
        return np.array([[13, 19, 22, 13, 27, 30],[16, 20, 23, 24, 28, 31]])
    elif dataset == "cmu":
        return np.array([[15, 15, 15, 23, 23, 32, 32],[16, 20, 29, 24, 27, 33, 36]])

def get_body_members(dataset="h36m"):
    if dataset=="h36m":
        body_members = {
            'left_arm': {'joints': [13, 17, 18, 19], 'side': 'left'},
            'right_arm': {'joints': [13, 25, 26, 27], 'side': 'right'},
            'head': {'joints': [13, 14, 15], 'side': 'right'},
            'torso': {'joints': [0, 12, 13], 'side': 'right'},
            'left_leg': {'joints': [0, 6, 7, 8], 'side': 'left'},
            'right_leg': {'joints': [0, 1, 2, 3], 'side': 'right'},
        }
    elif dataset == "cmu":
        body_members = {
            'left_arm': {'joints': [20, 30, 31, 32], 'side': 'left'},
            'right_arm': {'joints': [20, 21, 22, 23], 'side': 'right'},
            'head': {'joints': [20, 17, 18], 'sxide': 'right'},
            'torso': {'joints': [0, 14, 20], 'side': 'right'},
            'left_leg': {'joints': [0, 8, 9, 10], 'side': 'left'},
            'right_leg': {'joints': [0, 2, 3, 4], 'side': 'right'},
        }
    return body_members
