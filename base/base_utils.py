import os
import time
import shutil
from collections import OrderedDict

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.ndimage import measurements
from loguru import logger


def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)


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


def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.

    Args:
        x: input array
        crop_shape: dimensions of cropped array

    Returns:
        x: cropped array

    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0: h0 + crop_shape[0], w0: w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0: h0 + crop_shape[0], w0: w0 + crop_shape[1]]
    return x

def cur_time_str():
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

def bhwc_to_bchw(tensor):
    return tensor.permute(0, 3, 1, 2).contiguous()

def bchw_to_bhwc(tensor):
    return tensor.permute(0, 2, 3, 1).contiguous()

def fix_mirror_padding(ann):
    """Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.).

    """
    current_max_id = np.amax(ann)
    inst_list = list(np.unique(ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(ann == inst_id, np.uint8)
        remapped_ids = measurements.label(inst_map)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(ann)
    return ann

def rm_n_mkdir(dir_path):
    """Remove and make directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def mkdir(dir_path):
    """Make directory."""
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def get_inst_centroid(inst_map):
    """Get instance centroids given an input instance map.

    Args:
        inst_map: input instance map

    Returns:
        array of centroids

    """
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)


def center_pad_to_shape(img, size, cval=255):
    """Pad input image."""
    # rounding down, add 1
    pad_h = size[0] - img.shape[0]
    pad_w = size[1] - img.shape[1]
    pad_h = (pad_h // 2, pad_h - pad_h // 2)
    pad_w = (pad_w // 2, pad_w - pad_w // 2)
    if len(img.shape) == 2:
        pad_shape = (pad_h, pad_w)
    else:
        pad_shape = (pad_h, pad_w, (0, 0))
    img = np.pad(img, pad_shape, "constant", constant_values=cval)
    return img


def color_deconvolution(rgb, stain_mat):
    """Apply colour deconvolution."""
    log255 = np.log(255)  # to base 10, not base e
    rgb_float = rgb.astype(np.float64)
    log_rgb = -((255.0 * np.log((rgb_float + 1) / 255.0)) / log255)
    output = np.exp(-(log_rgb @ stain_mat - 255.0) * log255 / 255.0)
    output[output > 255] = 255
    output = np.floor(output + 0.5).astype("uint8")
    return output


def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image.

    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`

    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x

def binarize_pannuke(x):
    '''
    convert multichannel (multiclass) instance segmetation tensor
    to binary instance segmentation (bg and nuclei),
    :param x: B*B*C (for PanNuke 256*256*5 )
    :return: Instance segmentation
    '''
    out = np.zeros([x.shape[0], x.shape[1]])
    count = 1
    for i in range(x.shape[2]):
        x_ch = x[:,:,i]
        unique_vals = np.unique(x_ch)
        unique_vals = unique_vals.tolist()
        unique_vals.remove(0)
        for j in unique_vals:
            x_tmp = x_ch == j
            x_tmp_c = 1- x_tmp
            out *= x_tmp_c
            out += count*x_tmp
            count += 1
    out = out.astype('int32')
    return out

def crop_to_shape(x, y, data_format="NCHW"):
    """Centre crop x so that x has shape of y. y dims must be smaller than x dims.

    Args:
        x: input array
        y: array with desired shape.

    """
    assert (
            y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1]
    ), "Ensure that y dimensions are smaller than x dimensions!"

    x_shape = x.size()
    y_shape = y.size()
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])
    return crop_op(x, crop_shape, data_format)


def get_sobel_kernel(size):
    """Get sobel kernel with a given size."""
    assert size % 2 == 1, "Must be odd, get size=%d" % size

    h_range = torch.arange(
        -size // 2 + 1,
        size // 2 + 1,
        dtype=torch.float32,
        device=device(),
        requires_grad=False,
    )
    v_range = torch.arange(
        -size // 2 + 1,
        size // 2 + 1,
        dtype=torch.float32,
        device=device(),
        requires_grad=False,
    )
    h, v = torch.meshgrid(h_range, v_range)
    kernel_h = h / (h * h + v * v + 1.0e-15)
    kernel_v = v / (h * h + v * v + 1.0e-15)
    return kernel_h, kernel_v


def get_gradient_hv(hv):
    """For calculating gradient."""
    kernel_h, kernel_v = get_sobel_kernel(5)
    kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
    kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

    h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
    v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

    # can only apply in NCHW mode
    h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
    v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
    dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
    dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
    return dhv


def device():
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def ensure_array(array):
    if isinstance(array, np.ndarray):
        return array
    if isinstance(array, torch.Tensor):
        return array.cpu().detach().numpy()
    return np.array(array)


def gen_np_map(ann, crop_shape):
    np_map = ann.copy()
    np_map[np_map > 0] = 1
    np_map = cropping_center(np_map, crop_shape)
    return np_map

def remove_useless_logs(log_dir):
    dirnames = os.listdir(log_dir)
    dir_paths = [os.path.join(log_dir, dirname) for dirname in dirnames]

    for dir_path in dir_paths:
        checkpoint_dir = os.path.join(dir_path, 'checkpoints')
        if len(os.listdir(checkpoint_dir)) == 0:
            shutil.rmtree(dir_path)
            logger.info(f'Remove useless log {dir_path}')


def remove_useless_records(record_dir):
    record_names = os.listdir(record_dir)
    record_paths = [os.path.join(record_dir, record_name) for record_name in record_names]

    for path in record_paths:
        count = -1
        with open(path, 'r') as f:
            count = len(f.readlines())

        if count == 0 or count == 1:
            os.remove(path)
            logger.info(f'Remove useless record {path}')

def create_ann_inst_type_map(inst_map, type_map, num_types):
    ann_inst_type_map = np.zeros((inst_map.shape[0], inst_map.shape[1], num_types), dtype=np.int64)
    inst_id_list = np.unique(inst_map)[1:]
    for inst_id in inst_id_list:
        ann_inst_type_map[inst_map == inst_id, np.unique(type_map * (inst_map == inst_id))[1:]] = inst_id
    ann_inst_type_map = ann_inst_type_map[..., 1:]
    return ann_inst_type_map

def get_true_inst_info_dict_for_painting_nuclei(true_inst_map, true_type_map):
    inst_id_list = np.unique(true_inst_map)[1:]
    inst_info_dict = OrderedDict()
    for inst_id in inst_id_list:
        inst_map = true_inst_map == inst_id
        rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
        inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
        inst_map = inst_map[inst_bbox[0][0]: inst_bbox[1][0], inst_bbox[0][1]: inst_bbox[1][1]]
        inst_map = inst_map.astype(np.uint8)
        inst_moment = cv2.moments(inst_map)
        inst_contour = cv2.findContours(
            inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
        if inst_contour.shape[0] < 2:
            continue
        if len(inst_contour.shape) != 2:
            continue
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid = np.array(inst_centroid)
        inst_contour[:, 0] += inst_bbox[0][1]  # X
        inst_contour[:, 1] += inst_bbox[0][0]  # Y
        inst_centroid[0] += inst_bbox[0][1]  # X
        inst_centroid[1] += inst_bbox[0][0]  # Y

        type_map = true_type_map.copy()
        type_map[true_inst_map != inst_id] = 0

        type_id = np.unique(type_map)[1:].tolist()[0]
        inst_info_dict[inst_id] = {
            'centroid': inst_centroid,
            'type': type_id,
            'contour': inst_contour
        }

    return inst_info_dict

def get_true_inst_info_dict_for_computing_indexes(true_inst_type_map):
    inst_info_dict = OrderedDict()
    for channel in range(true_inst_type_map.shape[-1]):
        true_inst_map = true_inst_type_map[..., channel]
        inst_id_list = np.unique(true_inst_map)[1:]
        for inst_id in inst_id_list:
            inst_map = true_inst_map == inst_id
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_info_dict[inst_id] = {
                'centroid': inst_centroid,
                'type': channel + 1
            }
    return inst_info_dict

def convert_inst_info_dict_ndarray(inst_info_dict):
    inst_centroids = np.zeros((len(inst_info_dict.keys()), 2), dtype=np.float32)
    inst_types = np.zeros((len(inst_info_dict.keys()), 1), dtype=np.int64)
    for i, key in enumerate(inst_info_dict.keys()):
        inst_info = inst_info_dict[key]
        inst_centroids[i] = inst_info['centroid']
        inst_types[i] = inst_info['type']

    return inst_centroids, inst_types

def convert_true_tp_onehot_to_true_np_map(tp_one_hot):
    # 0 denotes background
    return 1 - tp_one_hot[..., 0]


def convert_pred_tp_map_to_one_channel_tp_map(pred_tp_map):
    type_map = F.softmax(pred_tp_map, dim=-1)
    type_map = torch.argmax(type_map, dim=-1, keepdim=True).type(torch.float32)
    return type_map


def convert_pred_tp_map_to_np_map(pred_tp_map):
    type_map = convert_pred_tp_map_to_one_channel_tp_map(pred_tp_map).type(torch.int64)
    return (type_map > 0).type(torch.float32)
