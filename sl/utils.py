import os
import time
import shutil
from collections import OrderedDict

import numpy as np
import cv2
import torch
import torch.nn.functional as F


def convert_pred_tp_map_to_one_channel_tp_map(pred_tp_map):
    type_map = F.softmax(pred_tp_map, dim=-1)
    type_map = torch.argmax(type_map, dim=-1, keepdim=True).type(torch.float32)
    return type_map


def ensure_array(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, list):
        return np.array(data)
    assert False, 'the type of data should be one of ndarray, Tensor or list'


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def bhwc_to_bchw(tensor):
    return tensor.permute(0, 3, 1, 2).contiguous()


def bchw_to_bhwc(tensor):
    return tensor.permute(0, 2, 3, 1).contiguous()


def device():
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def empty_tensor():
    return torch.tensor(0, device=device())


def is_empty_tensor(t):
    return len(t.shape) == 0


def cur_time_str():
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())


def interpolate_vector(vector, length):
    vector = ensure_array(vector)
    vector = torch.tensor(vector)
    vector = vector.unsqueeze(0).unsqueeze(0)
    vector = F.interpolate(vector, size=(1, length))
    vector = vector.squeeze(0).squeeze(0)
    return ensure_array(vector)


def one_hot(data, num_classes):
    data_tensor = torch.as_tensor(ensure_array(data), dtype=torch.int64)
    data_tensor = F.one_hot(data_tensor, num_classes=num_classes)
    return ensure_array(data_tensor)


def create_subtype_map(instance_map, pseudo_labels):
    subtype_map = np.zeros(instance_map.shape, dtype=np.int64)
    inst_id_list = np.unique(instance_map)[1:]
    for idx, inst_id in enumerate(inst_id_list):
        subtype_map[instance_map == inst_id] = np.argmax(pseudo_labels[idx])
    return subtype_map


def create_ann_inst_type_map(inst_map, type_map, num_types, reserve_background=False):
    ann_inst_type_map = np.zeros((inst_map.shape[0], inst_map.shape[1], num_types), dtype=np.int64)
    inst_id_list = np.unique(inst_map)[1:]
    for inst_id in inst_id_list:
        ann_inst_type_map[inst_map == inst_id, np.unique(type_map * (inst_map == inst_id))[1:]] = inst_id
    if not reserve_background:
        ann_inst_type_map = ann_inst_type_map[..., 1:]
    else:
        ann_inst_type_map[inst_map == 0, 0] = 1
    return ann_inst_type_map


def convert_inst_info_dict_ndarray(inst_info_dict):
    inst_centroids = np.zeros((len(inst_info_dict.keys()), 2), dtype=np.float32)
    inst_types = np.zeros((len(inst_info_dict.keys()), 1), dtype=np.int64)
    for i, key in enumerate(inst_info_dict.keys()):
        inst_info = inst_info_dict[key]
        inst_centroids[i] = inst_info['centroid']
        inst_types[i] = inst_info['type']

    return inst_centroids, inst_types
