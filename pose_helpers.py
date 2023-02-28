import torch as torch
import numpy as np
import time
import math
from scipy.spatial.transform import Rotation as R

def get_rotation_a_to_b_3d(a, b):
    a, b = a / np.linalg.norm(a, ord=2), b / np.linalg.norm(b, ord=2)
    return 2 * np.outer(a + b, a + b) / np.dot(a + b, a + b) - np.eye(3)


def normalize_and_rotate_poses_new(poses, joint_names):
    poses = poses.numpy()
    # Rotate such that Thorax is at [0, 0, x] for some x
    R_tensor = np.apply_along_axis(
        lambda x: get_rotation_a_to_b_3d(x, np.array([0, 0, 1])), 1, poses[:, :, joint_names.index('lowerneck')])
    poses = R_tensor @ poses

    # Rotate around z such that shoulders vector is parallel to yz plane
    tmp = poses[:, 0:2,  joint_names.index('lclavicle')] - poses[:, 0:2,  joint_names.index('rclavicle')]
    R_tensor = np.apply_along_axis(
        lambda x: R.from_euler('z', -np.arctan(x[1] / x[0]), degrees=False).as_matrix(), 1, tmp)
    poses = R_tensor @ poses

    tmp = poses[:, 0:2, joint_names.index('ltibia')] - poses[:, 0:2, 10]  # 10 is index of LToe
    tmp += poses[:, 0:2, joint_names.index('rtibia')] - poses[:, 0:2, 5]  # 5 in index of RToe
    R_tensor = np.apply_along_axis(
        lambda x: R.from_euler('z', 180, degrees=True).as_matrix() if x[1] < 0 else np.eye(3), 1, tmp)
    poses = R_tensor @ poses
    return torch.from_numpy(poses).float()

def normalize_body(poses, start_joint, end_joint, old_val, s1_bone_len):
    bone_vec = old_val-poses[:, :, end_joint]
    new_old_val = poses[:, :, end_joint].clone()
    new_bone_vector = s1_bone_len*bone_vec/torch.norm(bone_vec, dim=1, keepdim=True)
    poses[:,:,end_joint] = poses[:, :, start_joint] + new_bone_vector     
    return poses, new_old_val


def traverse_body(poses, start_joint, old_val, bone_connections, visited_list, subject_1_bone_len):
    count = 0
    for list_start_joint, list_end_joint in bone_connections:
        if start_joint == list_start_joint and not visited_list[count]:
            visited_list[count] = True
            poses, new_old_val = normalize_body(poses, list_start_joint, list_end_joint, old_val, subject_1_bone_len[count])
            return traverse_body(poses, list_end_joint, new_old_val, bone_connections, visited_list, subject_1_bone_len)
        count += 1
    return poses


def find_bone_lengths(bone_connections, pose):
    bone_len = []
    for list_start_joint, list_end_joint in bone_connections:
        bone_len.append(torch.norm(pose[:, list_start_joint]-pose[:, list_end_joint]))
    return bone_len

def convert_to_skeleton(poses, joint_names, to_skeleton, bone_connections):
    pass

def centralize_normalize_rotate_poses(poses, joint_names, subject_1_pose, bone_connections):  
    hip_index = joint_names.index('root')

    #centralize
    hip_pose = poses[:, :, hip_index]
    normalized_poses = poses - hip_pose.unsqueeze(2)
    num_of_poses = poses.shape[0]

    fail_msg  = "normalization created nans"
    assert not torch.isnan(normalized_poses).any(), fail_msg

    #make skeleton indep
    visited_list = [False]*len(bone_connections)
    subject_1_bone_len = find_bone_lengths(bone_connections, subject_1_pose)
    #print(find_bone_lengths(bone_connections, normalized_poses[5,:,:]))

    #normalized_poses = traverse_body(normalized_poses, 0, normalized_poses[:, :, hip_index], bone_connections, visited_list, subject_1_bone_len)
    assert torch.allclose(torch.FloatTensor(find_bone_lengths(bone_connections, normalized_poses[5,:,:])), torch.FloatTensor(subject_1_bone_len))

    #first rotation: make everyone's shoulder vector [0, 1]
    shoulder_vector = normalized_poses[:, :, joint_names.index('lclavicle')] - normalized_poses[:, :, joint_names.index('rclavicle')] 
    spine_vector = normalized_poses[:, :, joint_names.index('lowerneck')] - normalized_poses[:, :, joint_names.index('root')] 

    shoulder_vector = shoulder_vector/torch.norm(shoulder_vector, dim=1, keepdim=True)
    spine_vector = spine_vector/torch.norm(spine_vector, dim=1,  keepdim=True)

   
    normal_vector = torch.cross(shoulder_vector, spine_vector, dim=1)
    spine_vector = torch.cross(normal_vector, shoulder_vector, dim=1)
    assert normal_vector.shape == shoulder_vector.shape
    
    inv_rotation_matrix = torch.inverse(torch.cat([shoulder_vector.unsqueeze(2), normal_vector.unsqueeze(2), spine_vector.unsqueeze(2)], dim=2))

    rotated_poses =  torch.bmm(inv_rotation_matrix, normalized_poses)

    fail_msg  = "first rotation created nans"
    assert not torch.isnan(rotated_poses).any(), fail_msg

    #second rotation: make everyone's shoulder vector [0, 1]
    new_shoulder_vector = rotated_poses[:, :, joint_names.index('lclavicle')] - rotated_poses[:, :, joint_names.index('rclavicle')]
    new_shoulder_vector = new_shoulder_vector/torch.norm(new_shoulder_vector, dim=1, keepdim=True)
    new_spine_vector = rotated_poses[:, :, joint_names.index('lowerneck')] - rotated_poses[:, :, joint_names.index('root')] 
    new_spine_vector = new_spine_vector/torch.norm(new_spine_vector, dim=1, keepdim=True)
    new_normal_vector = torch.cross(new_shoulder_vector, new_spine_vector, dim=1)
    new_spine_vector = torch.cross(new_normal_vector, new_shoulder_vector, dim=1)

    assert (torch.allclose(torch.mean(new_shoulder_vector[:, 1:], dim=0), torch.FloatTensor([0,0])))
    assert (torch.allclose(torch.mean(new_normal_vector[:,[0,2]], dim=0), torch.FloatTensor([0,0])))
    assert (torch.allclose(torch.mean(new_spine_vector[:,:-1], dim=0), torch.FloatTensor([0,0])))
    assert (torch.allclose(torch.mean(rotated_poses[:,:, hip_index]), torch.FloatTensor([0])))

    return rotated_poses

def interpolate(input_motion, indices, num_of_joints):
    seq_len = input_motion.shape[0]
    output_motion = torch.zeros([seq_len, 3, num_of_joints])

    indices[indices <= 0] = 1
    indices[indices >= seq_len-1] = seq_len-2

    pose_1 = input_motion[0]
    ind_1 = 0
    for count in range(0, len(indices)+1):
        if count == len(indices):
            ind_2 = input_motion.shape[0] #last pose index
            pose_2 = input_motion[-1,:,:]
        else:
            ind_2 = indices[count]
            pose_2 = input_motion[int(ind_2)]*(ind_2%1) + input_motion[int(ind_2)+1]*(1-ind_2%1) 

        #pose is 3XJ
        delta_pose = ((pose_2-pose_1)/(ind_2-ind_1)).unsqueeze(0)
        time = ( ( (torch.arange(int(ind_2)-int(ind_1))).unsqueeze(1) ).unsqueeze(2)).repeat(1, 3, num_of_joints )
       
        #import pdb; pdb.set_trace()
       
        if count == len(indices):
            output_motion[int(ind_1):, :, :] = delta_pose*time + pose_1
        else:
            output_motion[int(ind_1):int(ind_2), :, :] = delta_pose*time + pose_1

        pose_1 = pose_2
        ind_1 = ind_2

    return output_motion


def interpolate_clusters(keyposes, labels, indices, first_pose, last_pose, num_of_joints, seq_len):
    output_motion = torch.zeros([seq_len, 3, num_of_joints])

    indices[indices<=0] =1
    indices[indices>=seq_len-1] =seq_len-2

    pose_1 = first_pose
    ind_1 = 0
    for count in range(0, len(indices)+1):
        if count == len(indices):
            ind_2 = input_motion.shape[0]-1 #last pose index
            pose_2 = input_motion[-1,:,:]

        else:
            ind_2 = indices[count]
            pose_2 = torch.mean(keyposes*labels[count], dim=0)

        
        #pose is 3XJ
        delta_pose = ((pose_2-pose_1)/(ind_2-ind_1)).unsqueeze(0)
        time = ( ( (torch.arange(int(ind_2)-int(ind_1))).unsqueeze(1) ).unsqueeze(2)).repeat(1, 3, num_of_joints )

        if count == len(indices):
            output_motion[int(ind_1):, :, :] = delta_pose*time + pose_1
        else:
            print(delta_pose.shape, pose_1.shape)
            output_motion[int(ind_1):int(ind_2), :, :] = delta_pose*time + pose_1

        pose_1 = pose_2
        ind_1 = ind_2

    return output_motion