import numpy as np
from collections import OrderedDict
import cv2

import torch
from skimage.morphology import remove_small_objects
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (binary_erosion, binary_dilation, binary_fill_holes)

import base.base_utils as utils


def produce_inst_type_maps(pred_map, num_types=6, return_centroids=False):
    """Post processing script for image tiles.
    Args:
        pred_map: combined with type_map and instance_map
        num_types: number of types

    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
        pred_type_out: pixel-wise nuclear type prediction
    """
    pred_type = pred_map[..., 0]
    pred_inst = pred_map[..., 1]
    pred_type = pred_type.astype(np.int32)
    pred_type = np.squeeze(pred_type)

    pred_inst = np.squeeze(pred_inst)

    produced_inst_type_map = np.zeros((pred_type.shape[0], pred_type.shape[1], num_types), dtype=np.int64)

    inst_info_dict = OrderedDict()
    if return_centroids or num_types is not None:
        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
        inst_info_dict = OrderedDict()
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            # TODO: chane format of bbox output
            rmin, rmax, cmin, cmax = utils.get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                       inst_bbox[0][0]: inst_bbox[1][0], inst_bbox[0][1]: inst_bbox[1][1]
                       ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue  # ! check for trickery shape
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }

    if num_types is not None:
        #### * Get class of each instance id, stored at index id-1
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = (
                    inst_map_crop == inst_id
            )  # TODO: duplicated operation, may be expensive
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            produced_inst_type_map[pred_inst == inst_id, int(inst_type)] = inst_id
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

    # print('here')
    # ! WARNING: ID MAY NOT BE CONTIGUOUS
    # inst_id in the dict maps to the same value in the `pred_inst`
    produced_inst_type_map = produced_inst_type_map[..., 1:]
    return produced_inst_type_map


def post_process(pred):
    """Postprocess the predicted type mask
    Parameters
    ----------
    pred: torch.Tensor or numpy.ndarray
        The predicted type mask, the format of pred is [B, H, W, C]

    Returns
    -------
    """
    pred = utils.ensure_array(pred).copy()
    pred = pred[..., 1:]
    if len(pred.shape) == 3:
        pred = pred[None, :, :, :]
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    pred = pred.astype('int64')
    num_patches = pred.shape[0]

    instance_maps = np.zeros(pred.shape[:3], dtype=np.int64)
    type_maps = np.zeros(pred.shape[:3], dtype=np.int64)
    new_inst_id = 1
    for i in range(num_patches):
        pred_i = pred[i]
        for class_idx in range(pred.shape[3]):
            pred_i[..., class_idx] = binary_fill_holes(pred_i[..., class_idx])
            pred_i[..., class_idx] = measurements.label(pred_i[..., class_idx])[0]
            pred_i[..., class_idx] = remove_small_objects(pred_i[..., class_idx], min_size=10)
            type_maps[i][pred_i[..., class_idx] > 0] = class_idx + 1
            inst_id_list = np.unique(pred_i[..., class_idx])[1:]
            for inst_id in inst_id_list:
                instance_maps[i][pred_i[..., class_idx] == inst_id] = new_inst_id
                new_inst_id += 1
    return instance_maps, type_maps


def process(pred_map, num_types=None, return_centroids=False):
    """Post processing script for image tiles.
    Args:
        pred_map: commbined output of tp, np and hv branches, in the same order
        num_types: number of types considered at output of nc branch

    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
        pred_type_out: pixel-wise nuclear type prediction
    """
    if num_types is not None:
        pred_type = pred_map[..., :1]
        pred_inst = pred_map[..., 1:]
        pred_type = pred_type.astype(np.int32)
        pred_type = np.squeeze(pred_type)
    else:
        pred_inst = pred_map

    pred_inst = np.squeeze(pred_inst)
    inst_info_dict = OrderedDict()
    if return_centroids or num_types is not None:
        inst_id_list = np.unique(pred_inst)[1:]  # exclude background
        inst_info_dict = OrderedDict()
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id

            # TODO: chane format of bbox output
            rmin, rmax, cmin, cmax = utils.get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                       inst_bbox[0][0]: inst_bbox[1][0], inst_bbox[0][1]: inst_bbox[1][1]
                       ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue  # ! check for trickery shape
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }

    if num_types is not None:
        #### * Get class of each instance id, stored at index id-1
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = (
                    inst_map_crop == inst_id
            )  # TODO: duplicated operation, may be expensive
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

    # print('here')
    # ! WARNING: ID MAY NOT BE CONTIGUOUS
    # inst_id in the dict maps to the same value in the `pred_inst`
    return pred_inst, inst_info_dict
