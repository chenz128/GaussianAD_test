import numpy as np
from pyquaternion import Quaternion
import torch


def get_rm(angle, axis, deg=False):
    if deg:
        angle = np.deg2rad(angle)
    rm = np.eye(3)
    if axis == 'x':
        rm[1, 1] = np.cos(angle)
        rm[2, 2] = np.cos(angle)
        rm[1, 2] = - np.sin(angle)
        rm[2, 1] = np.sin(angle)
    elif axis == 'y':
        rm[0, 0] = np.cos(angle)
        rm[2, 2] = np.cos(angle)
        rm[0, 2] = np.sin(angle)
        rm[2, 0] = - np.sin(angle)
    elif axis == 'z':
        rm[0, 0] = np.cos(angle)
        rm[1, 1] = np.cos(angle)
        rm[0, 1] = - np.sin(angle)
        rm[1, 0] = np.sin(angle)
    return rm


def get_xyz(pose_dict):
    return np.array(pose_dict['translation'])

def get_img2global(calib_dict, pose_dict):
    
    cam2img = np.eye(4)
    cam2img[:3, :3] = np.asarray(calib_dict['camera_intrinsic'])
    img2cam = np.linalg.inv(cam2img)

    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    cam2ego[:3, 3] = np.asarray(calib_dict['translation']).T

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict['translation']).T

    img2global = ego2global @ cam2ego @ img2cam
    return img2global

def get_lidar2global(calib_dict, pose_dict):

    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(calib_dict['translation']).T

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict['translation']).T

    lidar2global = ego2global @ lidar2ego
    return lidar2global


def _can_pad_on_first_dim(shapes):
    if not shapes:
        return False
    ndim = len(shapes[0])
    if ndim == 0:
        return True
    trailing_shape = shapes[0][1:]
    return all(len(shape) == ndim and shape[1:] == trailing_shape for shape in shapes)


def _pad_and_stack_tensors(tensors, pad_value=0):
    shapes = [tuple(tensor.shape) for tensor in tensors]
    if len(set(shapes)) == 1:
        return torch.stack(tensors)

    if not _can_pad_on_first_dim(shapes):
        raise ValueError(f'cannot pad tensors with shapes: {shapes}')

    max_instances = max(shape[0] for shape in shapes)
    padded = tensors[0].new_full(
        (len(tensors), max_instances, *shapes[0][1:]),
        pad_value,
    )
    for batch_index, tensor in enumerate(tensors):
        padded[batch_index, :tensor.shape[0]] = tensor
    return padded


def _collate_numeric_values(values):
    tensors = []
    for value in values:
        if isinstance(value, np.ndarray):
            if value.dtype.kind in {'O', 'U', 'S'}:
                return values
            tensors.append(torch.from_numpy(value))
        else:
            tensors.append(value)

    try:
        return _pad_and_stack_tensors(tensors)
    except ValueError:
        return values


def custom_collate_fn_temporal(instances):
    return_dict = {}
    for k, v in instances[0].items():
        if isinstance(v, np.ndarray):
            return_dict[k] = _collate_numeric_values([
                instance[k] for instance in instances])
        elif isinstance(v, torch.Tensor):
            return_dict[k] = _collate_numeric_values([
                instance[k] for instance in instances])
        elif v is None:
            return_dict[k] = [None] * len(instances)
        else:
            return_dict[k] = [instance[k] for instance in instances]
    return return_dict
