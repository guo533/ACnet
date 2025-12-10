import numpy as np

from scipy.ndimage import measurements
from skimage import morphology as morph

import base.base_utils as utils


def gen_instance_hv_map(ann, crop_shape):
    """Input annotation must be of original shape.

    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.
    """
    orig_ann = ann.copy()
    fixed_ann = utils.fix_mirror_padding(orig_ann)
    crop_ann = utils.cropping_center(fixed_ann, crop_shape)
    crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_ann))
    inst_list.remove(0)
    for inst_id in inst_list:
        inst_map = np.array(fixed_ann == inst_id, np.uint8)
        inst_box = utils.get_bounding_box(inst_map)
        inst_box[0] = max(inst_box[0] - 2, 0)
        inst_box[2] = max(inst_box[2] - 2, 0)
        inst_box[1] = min(inst_box[1] + 2, inst_map.shape[0])
        inst_box[3] = min(inst_box[3] + 2, inst_map.shape[1])
        inst_map = inst_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]
        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue

        inst_com = list(measurements.center_of_mass(inst_map))

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])

        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        x_map_box = x_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = np.dstack([x_map, y_map])
    return hv_map


def gen_targets(ann, crop_shape, **kwargs):
    """Generate the targets for the network."""
    hv_map = gen_instance_hv_map(ann, crop_shape)
    np_map = ann.copy()
    np_map[np_map > 0] = 1

    hv_map = utils.cropping_center(hv_map, crop_shape)
    np_map = utils.cropping_center(np_map, crop_shape)

    target_dict = {
        "hv_map": hv_map,
        "np_map": np_map,
    }
    return target_dict


def gen_np_map(ann, crop_shape):
    np_map = ann.copy()
    np_map[np_map > 0] = 1

    np_map = utils.cropping_center(np_map, crop_shape)

    target_dict = {
        "np_map": np_map,
    }
    return target_dict
