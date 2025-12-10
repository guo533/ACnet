#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import time
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
from skimage.morphology import remove_small_objects
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (binary_erosion, binary_dilation, binary_fill_holes)
from torch.utils.data import DataLoader

from model.unet import UNet
import base.base_utils as utils
import base.metrics as metrics
from base.metrics import NucleiTypeStat
from base.base_dataset import FileLoaderExt
from base.base_dataset import LUSCTestFileLoaderExt


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
            # print(inst_type)
            produced_inst_type_map[pred_inst == inst_id, int(inst_type)] = inst_id
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

    # print('here')
    # ! WARNING: ID MAY NOT BE CONTIGUOUS
    # inst_id in the dict maps to the same value in the `pred_inst`
    produced_inst_type_map = produced_inst_type_map[..., 1:]
    return produced_inst_type_map


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


def do_eval_model_pannuke(opt):
    net = UNet(in_channels=3, num_types=opt.num_types).to(utils.device())
    assert os.path.exists(opt.checkpoint_path), 'pretrained model ' + opt.checkpoint_path + ' does not exist'

    net_state_dict = torch.load(opt.checkpoint_path, map_location=utils.device())['net_state_dict']
    load_feedback = net.load_state_dict(net_state_dict, strict=True)
    print(f'Load checkpoint from {opt.checkpoint_path}')
    print(f'Missing variables: \n{load_feedback[0]}')
    print(f'Detected unknown variables: \n{load_feedback[1]}')
    net.eval()

    filenames = os.listdir(opt.dataset_dir)
    filenames.sort(key=lambda x: int(x[:x.find('.npy')]))

    file_list = [os.path.join(opt.dataset_dir, filename) for filename in filenames]

    val_input_dataset = FileLoaderExt(file_list, with_type=True, input_shape=(opt.patch_size, opt.patch_size), mask_shape=(opt.mask_size, opt.mask_size), mode='test', gen_hv=False)
    dataloader = DataLoader(
        val_input_dataset,
        num_workers=opt.workers,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False
    )

    dice_record = np.array([], dtype=np.float32)
    aji_record = np.array([], dtype=np.float32)
    pq_record = np.array([], dtype=np.float32)
    dq_record = np.array([], dtype=np.float32)
    sq_record = np.array([], dtype=np.float32)

    neo_pq_record = np.array([], dtype=np.float32)
    inflam_pq_record = np.array([], dtype=np.float32)
    conn_pq_record = np.array([], dtype=np.float32)
    dead_pq_record = np.array([], dtype=np.float32)
    nonneo_pq_record = np.array([], dtype=np.float32)
    mpq_record = np.array([], dtype=np.float32)

    nuclei_type_stat = NucleiTypeStat()

    cnt = 0

    file_idx = 0
    for batch_idx, batch_data in enumerate(dataloader):
        images = batch_data['image'].to(utils.device()).type(torch.float32) / 255.0
        anns = utils.ensure_array(batch_data['ann'].to(utils.device()).type(torch.int64))
        instance_maps = batch_data['instance_map'].type(torch.int64)
        type_maps = batch_data['type_map'].type(torch.int64)

        images = utils.bhwc_to_bchw(images)

        with torch.no_grad():
            raw_pred_dict = net(images)
            pred_dict = OrderedDict([(k, utils.bchw_to_bhwc(v)) for k, v in raw_pred_dict.items()])
            pred_dict['type_map'] = F.softmax(pred_dict['type_map'], dim=-1)
            # Process outputs
            pred_instance_maps, pred_type_maps = post_process(pred_dict['type_map'])

        pred_maps = np.concatenate((pred_type_maps[..., None], pred_instance_maps[..., None]), axis=-1)

        num_patches = images.shape[0]

        instance_maps = utils.ensure_array(instance_maps)
        type_maps = utils.ensure_array(type_maps)
        for i in range(num_patches):
            instance_map = instance_maps[i]
            true_type_map = np.squeeze(anns[i])[..., 1]
            true_inst_type_map = utils.create_ann_inst_type_map(instance_map, true_type_map, opt.num_types)
            true_inst_info_dict = utils.get_true_inst_info_dict_for_computing_indexes(true_inst_type_map)

            cnt += 1

            # pred_map contains type_map and binary_map
            pred_map = pred_maps[i]
            pred_inst_type_map = produce_inst_type_maps(pred_map, num_types=opt.num_types)
            pred_inst, pred_inst_info_dict = process(pred_map, num_types=opt.num_types, return_centroids=True)

            pred_inst = metrics.remap_label(pred_inst)

            pq_tmps = []
            for j in range(opt.num_types - 1):
                pred_tmp = pred_inst_type_map[..., j].astype('int64')
                true_tmp = true_inst_type_map[..., j].astype('int64')
                pred_tmp = metrics.remap_label(pred_tmp)
                true_tmp = metrics.remap_label(true_tmp)
                if len(np.unique(true_tmp)) == 1:
                    pq_tmps.append(np.nan)
                else:
                    pq_tmps.append(metrics.get_fast_pq(true_tmp, pred_tmp)[0][2])

            if len(np.unique(instance_map)) == 1:
                dice = np.nan
                aji = np.nan
                dq = np.nan
                sq = np.nan
                pq = np.nan
            else:
                instance_map = metrics.remap_label(instance_map)
                dice = metrics.get_dice_1(instance_map, pred_inst)
                aji = metrics.get_fast_aji(instance_map, pred_inst)
                dq, sq, pq = metrics.get_fast_pq(instance_map, pred_inst)[0]

            neo_pq_record = np.append(neo_pq_record, pq_tmps[0])
            inflam_pq_record = np.append(inflam_pq_record, pq_tmps[1])
            conn_pq_record = np.append(conn_pq_record, pq_tmps[2])
            dead_pq_record = np.append(dead_pq_record, pq_tmps[3])
            nonneo_pq_record = np.append(nonneo_pq_record, pq_tmps[4])
            mpq_record = np.append(mpq_record, np.nanmean(pq_tmps))

            dice_record = np.append(dice_record, dice)
            aji_record = np.append(aji_record, aji)
            dq_record = np.append(dq_record, dq)
            sq_record = np.append(sq_record, sq)
            pq_record = np.append(pq_record, pq)
            print(f'{file_idx}.npy\tDICE: {dice: 0.6}\tAJI: {aji: 0.6}\tPQ: {pq: 0.6}\tneo pq: {pq_tmps[0]: 0.6}\tinflam pq: {pq_tmps[1]: 0.6}\tconn pq: {pq_tmps[2]: 0.6}\tdead pq: {pq_tmps[3]: 0.6}\tnoneo pq: {pq_tmps[4]: 0.6}\tmpq: {np.nanmean(pq_tmps): 0.6}')
            file_idx += 1

            pred_inst_centroid, pred_inst_type = utils.convert_inst_info_dict_ndarray(pred_inst_info_dict)
            true_inst_centroid, true_inst_type = utils.convert_inst_info_dict_ndarray(true_inst_info_dict)
            nuclei_type_stat.process(true_inst_centroid, true_inst_type, pred_inst_centroid, pred_inst_type)

    avg_dice = np.nanmean(dice_record)
    avg_aji = np.nanmean(aji_record)
    avg_pq = np.nanmean(pq_record)
    avg_dq = np.nanmean(dq_record)
    avg_sq = np.nanmean(sq_record)

    avg_neo_pq = np.nanmean(neo_pq_record)
    avg_inflam_pq_record = np.nanmean(inflam_pq_record)
    avg_conn_pq_record = np.nanmean(conn_pq_record)
    avg_dead_pq_record = np.nanmean(dead_pq_record)
    avg_nonneo_pq_record = np.nanmean(nonneo_pq_record)
    avg_mpq_record = np.nanmean(mpq_record)

    nuclei_type_stat_results = nuclei_type_stat.finish()

    print(f'Avg DICE: {avg_dice: 0.6}\tavg AJI: {avg_aji: 0.6}\tavg PQ: {avg_pq: 0.6}\tavg neo pq: {avg_neo_pq: 0.6}\tavg inflam pq: {avg_inflam_pq_record: 0.6}\tavg conn pq: {avg_conn_pq_record: 0.6}\tavg dead pq: {avg_dead_pq_record: 0.6}\tavg nonneo pq: {avg_nonneo_pq_record: 0.6}\tmpq:{avg_mpq_record: 0.6}')
    print(nuclei_type_stat_results)
    return avg_dice, avg_aji, avg_dq, avg_sq, avg_pq, *nuclei_type_stat_results, avg_neo_pq, avg_inflam_pq_record, avg_conn_pq_record, avg_dead_pq_record, avg_nonneo_pq_record, avg_mpq_record, (avg_neo_pq + avg_inflam_pq_record + avg_conn_pq_record + avg_dead_pq_record + avg_nonneo_pq_record) / 5

def do_eval_model_lizard(opt):
    net = UNet(in_channels=3, num_types=opt.num_types).to(utils.device())
    assert os.path.exists(opt.checkpoint_path), 'pretrained model ' + opt.checkpoint_path + ' does not exist'

    net_state_dict = torch.load(opt.checkpoint_path, map_location=utils.device())['net_state_dict']
    load_feedback = net.load_state_dict(net_state_dict, strict=True)
    net.eval()

    filenames = os.listdir(opt.dataset_dir)
    filenames.sort(key=lambda x: int(x[:x.find('.npy')]))

    file_list = [os.path.join(opt.dataset_dir, filename) for filename in filenames]

    val_input_dataset = FileLoaderExt(file_list, with_type=True, input_shape=(opt.patch_size, opt.patch_size), mask_shape=(opt.mask_size, opt.mask_size), mode='test', gen_hv=False)
    dataloader = DataLoader(
        val_input_dataset,
        num_workers=opt.workers,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False
    )

    dice_record = np.array([], dtype=np.float32)
    aji_record = np.array([], dtype=np.float32)
    pq_record = np.array([], dtype=np.float32)
    dq_record = np.array([], dtype=np.float32)
    sq_record = np.array([], dtype=np.float32)

    neu_pq_record = np.array([], dtype=np.float32)
    epi_pq_record = np.array([], dtype=np.float32)
    lym_pq_record = np.array([], dtype=np.float32)
    pla_pq_record = np.array([], dtype=np.float32)
    eos_pq_record = np.array([], dtype=np.float32)
    con_pq_record = np.array([], dtype=np.float32)

    nuclei_type_stat = NucleiTypeStat()

    file_idx = 0
    for batch_idx, batch_data in enumerate(dataloader):
        images = batch_data['image'].to(utils.device()).type(torch.float32) / 255.0
        anns = utils.ensure_array(batch_data['ann'].to(utils.device()).type(torch.int64))
        instance_maps = batch_data['instance_map'].type(torch.int64)
        type_maps = batch_data['type_map'].type(torch.int64)

        images = utils.bhwc_to_bchw(images)

        with torch.no_grad():
            raw_pred_dict = net(images)
            pred_dict = OrderedDict([(k, utils.bchw_to_bhwc(v)) for k, v in raw_pred_dict.items()])
            pred_dict['type_map'] = F.softmax(pred_dict['type_map'], dim=-1)
            # Process outputs
            pred_instance_maps, pred_type_maps = post_process(pred_dict['type_map'])

        pred_maps = np.concatenate((pred_type_maps[..., None], pred_instance_maps[..., None]), axis=-1)

        num_patches = images.shape[0]

        instance_maps = utils.ensure_array(instance_maps)
        type_maps = utils.ensure_array(type_maps)
        for i in range(num_patches):
            instance_map = instance_maps[i]
            true_type_map = np.squeeze(anns[i])[..., 1]
            true_inst_type_map = utils.create_ann_inst_type_map(instance_map, true_type_map, opt.num_types)
            true_inst_info_dict = utils.get_true_inst_info_dict_for_computing_indexes(true_inst_type_map)

            # pred_map contains type_map and binary_map
            pred_map = pred_maps[i]
            pred_inst_type_map = produce_inst_type_maps(pred_map, num_types=opt.num_types)
            pred_inst, pred_inst_info_dict = process(pred_map, num_types=opt.num_types, return_centroids=True)

            pred_inst = metrics.remap_label(pred_inst)

            pq_tmps = []
            for j in range(opt.num_types - 1):
                pred_tmp = pred_inst_type_map[..., j].astype('int64')
                true_tmp = true_inst_type_map[..., j].astype('int64')
                pred_tmp = metrics.remap_label(pred_tmp)
                true_tmp = metrics.remap_label(true_tmp)
                if len(np.unique(true_tmp)) == 1:
                    pq_tmps.append(np.nan)
                else:
                    pq_tmps.append(metrics.get_fast_pq(true_tmp, pred_tmp)[0][2])

            if len(np.unique(instance_map)) == 1:
                dice = np.nan
                aji = np.nan
                dq = np.nan
                sq = np.nan
                pq = np.nan
            else:
                instance_map = metrics.remap_label(instance_map)
                dice = metrics.get_dice_1(instance_map, pred_inst)
                aji = metrics.get_fast_aji(instance_map, pred_inst)
                dq, sq, pq = metrics.get_fast_pq(instance_map, pred_inst)[0]

            neu_pq_record = np.append(neu_pq_record, pq_tmps[0])
            epi_pq_record = np.append(epi_pq_record, pq_tmps[1])
            lym_pq_record = np.append(lym_pq_record, pq_tmps[2])
            pla_pq_record = np.append(pla_pq_record, pq_tmps[3])
            eos_pq_record = np.append(eos_pq_record, pq_tmps[4])
            con_pq_record = np.append(con_pq_record, pq_tmps[5])

            dice_record = np.append(dice_record, dice)
            aji_record = np.append(aji_record, aji)
            dq_record = np.append(dq_record, dq)
            sq_record = np.append(sq_record, sq)
            pq_record = np.append(pq_record, pq)
            file_idx += 1

            pred_inst_centroid, pred_inst_type = utils.convert_inst_info_dict_ndarray(pred_inst_info_dict)
            true_inst_centroid, true_inst_type = utils.convert_inst_info_dict_ndarray(true_inst_info_dict)
            nuclei_type_stat.process(true_inst_centroid, true_inst_type, pred_inst_centroid, pred_inst_type)

    avg_dice = np.nanmean(dice_record)
    avg_aji = np.nanmean(aji_record)
    avg_pq = np.nanmean(pq_record)
    avg_dq = np.nanmean(dq_record)
    avg_sq = np.nanmean(sq_record)

    avg_neu_pq_record = np.nanmean(neu_pq_record)
    avg_epi_pq_record = np.nanmean(epi_pq_record)
    avg_lym_pq_record = np.nanmean(lym_pq_record)
    avg_pla_pq_record = np.nanmean(pla_pq_record)
    avg_eos_pq_record = np.nanmean(eos_pq_record)
    avg_con_pq_record = np.nanmean(con_pq_record)

    nuclei_type_stat_results = nuclei_type_stat.finish()

    print(f'Avg DICE: {avg_dice: 0.6}\tavg AJI: {avg_aji: 0.6}\tavg PQ: {avg_pq: 0.6}\tavg neu pq: {avg_neu_pq_record: 0.6}\tavg epi pq: {avg_epi_pq_record: 0.6}\tavg lym pq: {avg_lym_pq_record: 0.6}\tavg pla pq: {avg_pla_pq_record: 0.6}\tavg eos pq: {avg_eos_pq_record: 0.6}\tavg con pq: {avg_con_pq_record: 0.6}\t')
    print(nuclei_type_stat_results)
    return avg_dice, avg_aji, avg_dq, avg_sq, avg_pq, *nuclei_type_stat_results, avg_neu_pq_record, avg_epi_pq_record, avg_lym_pq_record, avg_pla_pq_record, avg_eos_pq_record, avg_con_pq_record, (avg_neu_pq_record + avg_epi_pq_record + avg_lym_pq_record + avg_pla_pq_record + avg_eos_pq_record + avg_con_pq_record) / (opt.num_types - 1)

def do_eval_model_lusc(opt):
    net = UNet(in_channels=3, num_types=opt.num_types).to(utils.device())
    net_state_dict = torch.load(opt.checkpoint_path, map_location=utils.device())['net_state_dict']
    net.load_state_dict(net_state_dict)
    net.eval()

    filenames = os.listdir(opt.dataset_dir)
    filenames.sort(key=lambda x: int(x))
    file_list = [os.path.join(opt.dataset_dir, filename) for filename in filenames]

    patch_file_list = []
    for cur_file_path in file_list:
        names = os.listdir(cur_file_path)
        names.sort(key=lambda x: int(x[:x.find('.npy')]))
        patch_files = []
        for name in names:
            full_path = os.path.join(cur_file_path, name)
            patch_files.append(full_path)
        patch_file_list.append(patch_files)

    ann_filenames = os.listdir(opt.lusc_ann_dir)
    ann_filenames.sort(key=lambda x: int(x[:x.find('.npy')]))
    ann_file_list = [os.path.join(opt.lusc_ann_dir, ann_filename) for ann_filename in ann_filenames]

    val_dataloaders = []
    for cur_file_patch_list in patch_file_list:
        val_input_dataset = LUSCTestFileLoaderExt(cur_file_patch_list, input_shape=(opt.patch_size, opt.patch_size),
                                              mask_shape=(opt.mask_size, opt.mask_size))
        val_dataloader = DataLoader(
            val_input_dataset,
            num_workers=opt.workers,
            batch_size=opt.batch_size,
            shuffle=False,
            drop_last=False
        )
        val_dataloaders.append(val_dataloader)

    dice_record = np.array([], dtype=np.float32)
    aji_record = np.array([], dtype=np.float32)
    pq_record = np.array([], dtype=np.float32)
    dq_record = np.array([], dtype=np.float32)
    sq_record = np.array([], dtype=np.float32)

    pos_pq_record = np.array([], dtype=np.float32)
    neg_pq_record = np.array([], dtype=np.float32)

    nuclei_type_stat = NucleiTypeStat()

    file_idx = 0
    for full_file_idx, val_dataloader in enumerate(val_dataloaders):
        for batch_idx, batch_data in enumerate(val_dataloader):
            imgs = batch_data['img'].to(utils.device()).type(torch.float32)
            anns = utils.ensure_array(batch_data['ann'].to(utils.device()).type(torch.int64))
            imgs = imgs.permute(0, 3, 1, 2).contiguous()
            imgs /= 255.0
            with torch.no_grad():
                raw_pred_dict = net(imgs)
                raw_pred_dict = OrderedDict([(k, v.permute(0, 2, 3, 1).contiguous()) for k, v in raw_pred_dict.items()])
                pred_dict = OrderedDict()
                pred_dict['tp'] = raw_pred_dict['type_map']
                pred_dict['tp'] = F.softmax(pred_dict['tp'], dim=-1)

            pred = utils.ensure_array(torch.cat(list(pred_dict.values()), axis=-1))
            pred_list = np.split(pred, pred.shape[0], axis=0)
            full_pred_map = np.zeros((1, 1024, 1024, 3))

            the_count = 0
            for col in range(0, 1024, 256):
                for row in range(0, 1024, 256):
                    full_pred_map[0, col: col + 256, row: row + 256, ...] = pred_list[the_count]
                    the_count += 1
            fix_full_pred_map = full_pred_map[0, :1000, :1000, ...]
            ann = np.load(ann_file_list[full_file_idx])[..., 3:]
            ann = ann[:1000, :1000, ...]
            true_inst_map = np.squeeze(ann)[..., 0]
            true_type_map = np.squeeze(ann)[..., 1]

            pred_instance_maps, pred_type_maps = post_process(fix_full_pred_map)
            pred_instance_maps = pred_instance_maps[0]
            pred_type_maps = pred_type_maps[0]
            fix_full_pred_map = np.concatenate((pred_type_maps[..., None], pred_instance_maps[..., None]), axis=-1)
            pred_inst_type_map = produce_inst_type_maps(fix_full_pred_map, num_types=opt.num_types)
            true_inst_type_map = utils.create_ann_inst_type_map(true_inst_map, true_type_map, opt.num_types)
            pred_inst, inst_info_dict = process(fix_full_pred_map, num_types=opt.num_types, return_centroids=True)
            true_inst_info_dict = utils.get_true_inst_info_dict_for_computing_indexes(true_inst_type_map)
            pred_inst = metrics.remap_label(pred_inst)

            pq_tmps = []
            for j in range(opt.num_types - 1):
                pred_tmp = pred_inst_type_map[..., j].astype('int64')
                true_tmp = true_inst_type_map[..., j].astype('int64')
                pred_tmp = metrics.remap_label(pred_tmp)
                true_tmp = metrics.remap_label(true_tmp)
                if len(np.unique(true_tmp)) == 1:
                    pq_tmps.append(np.nan)
                else:
                    pq_tmps.append(metrics.get_fast_pq(true_tmp, pred_tmp)[0][2])

            if len(np.unique(true_inst_map)) == 1:
                dice = np.nan
                aji = np.nan
                dq = np.nan
                sq = np.nan
                pq = np.nan
            else:
                true_inst_map = metrics.remap_label(true_inst_map)
                dice = metrics.get_dice_1(true_inst_map, pred_inst)
                aji = metrics.get_fast_aji(true_inst_map, pred_inst)
                dq, sq, pq = metrics.get_fast_pq(true_inst_map, pred_inst)[0]

            pos_pq_record = np.append(pos_pq_record, pq_tmps[0])
            neg_pq_record = np.append(neg_pq_record, pq_tmps[1])
            mpq_record = np.append(mpq_record, np.nanmean(pq_tmps))

            dice_record = np.append(dice_record, dice)
            aji_record = np.append(aji_record, aji)
            pq_record = np.append(pq_record, pq)
            dq_record = np.append(dq_record, dq)
            sq_record = np.append(sq_record, sq)
            print(f'{filenames[file_idx]}\tDICE: {dice: 0.6}\tAJI: {aji: 0.6}\tPQ: {pq: 0.6}\tpos pq: {pq_tmps[0]: 0.6}\tneg pq: {pq_tmps[1]: 0.6}')
            file_idx += 1

            pred_inst_centroid, pred_inst_type = utils.convert_inst_info_dict_ndarray(inst_info_dict)
            true_inst_centroid, true_inst_type = utils.convert_inst_info_dict_ndarray(true_inst_info_dict)
            nuclei_type_stat.process(true_inst_centroid, true_inst_type, pred_inst_centroid, pred_inst_type)

    avg_dice = np.nanmean(dice_record)
    avg_aji = np.nanmean(aji_record)
    avg_pq = np.nanmean(pq_record)
    avg_dq = np.nanmean(dq_record)
    avg_sq = np.nanmean(sq_record)

    avg_pos_pq_record = np.nanmean(pos_pq_record)
    avg_neg_pq_record = np.nanmean(neg_pq_record)

    nuclei_type_stat_results = nuclei_type_stat.finish()
    print(f'Avg DICE: {avg_dice: 0.6}\tavg AJI: {avg_aji: 0.6}\tavg PQ: {avg_pq: 0.6}\tavg pos pq: {avg_pos_pq_record: 0.6}\tavg neg pq: {avg_neg_pq_record: 0.6}')
    print(nuclei_type_stat_results)
    return avg_dice, avg_aji, avg_dq, avg_sq, avg_pq, *nuclei_type_stat_results, avg_pos_pq_record, avg_neg_pq_record, (avg_pos_pq_record + avg_neg_pq_record) / 2


def eval_model_pannuke(opt):
    utils.mkdir(opt.record_dir)
    utils.remove_useless_records(opt.record_dir)
    record_path = os.path.join(opt.record_dir, f'{utils.cur_time_str()}.csv')
    with open(record_path, 'w') as f:
        f.write('dice,aji,dq,sq,pq,f_d,acc,f_c_n,f_c_i,f_c_c,f_c_d,f_c_e,neo_pq,inflam_pq,conn_pq,dead_pq,nonneo_pq,tmp\n')
    avg_dice, avg_aji, avg_dq, avg_sq, avg_pq, f_d, acc, f_c_n, f_c_i, f_c_c, f_c_d, f_c_e, neo_pq,inflam_pq,conn_pq,dead_pq,nonneo_pq,tmp = do_eval_model_pannuke(opt)
    with open(record_path, 'a') as f:
        f.write(f'{avg_dice:0.6},{avg_aji:0.6},{avg_dq:0.6},{avg_sq:0.6},{avg_pq:0.6},{f_d:0.6},{acc:0.6},{f_c_n:0.6},{f_c_i:0.6},{f_c_c:0.6},{f_c_d:0.6},{f_c_e:0.6},{neo_pq:0.6},{inflam_pq:0.6},{conn_pq:0.6},{dead_pq:0.6},{nonneo_pq:0.6},{tmp:0.6}\n')

def eval_model_lizard(opt):
    utils.mkdir(opt.record_dir)
    utils.remove_useless_records(opt.record_dir)
    record_path = os.path.join(opt.record_dir, f'{utils.cur_time_str()}.csv')
    with open(record_path, 'w') as f:
        f.write('dice,aji,dq,sq,pq,f_d,acc,f_c_neu,f_c_epi,f_c_lym,f_c_pla,f_c_eos,f_c_con,neu_pq,epi_pq,lym_pq,pla_pq,eos_pq,con_pq,tmp\n')
    avg_dice, avg_aji, avg_dq, avg_sq, avg_pq, f_d, acc, f_c_neu, f_c_epi, f_c_lym, f_c_pla, f_c_eos, f_c_con, neu_pq,epi_pq,lym_pq,pla_pq,eos_pq,con_pq,tmp = do_eval_model_lizard(opt)
    with open(record_path, 'a') as f:
        f.write(f'{avg_dice:0.6},{avg_aji:0.6},{avg_dq:0.6},{avg_sq:0.6},{avg_pq:0.6},{f_d:0.6},{acc:0.6},{f_c_neu:0.6},{f_c_epi:0.6},{f_c_lym:0.6},{f_c_pla:0.6},{f_c_eos:0.6},{f_c_con:0.6},{neu_pq:0.6},{epi_pq:0.6},{lym_pq:0.6},{pla_pq:0.6},{eos_pq:0.6},{con_pq:0.6},{tmp:0.6}\n')

def eval_model_lusc(opt):
    utils.mkdir(opt.record_dir)
    utils.remove_useless_records(opt.record_dir)
    record_path = os.path.join(opt.record_dir, f'{utils.cur_time_str()}.csv')
    with open(record_path, 'w') as f:
        f.write('dice,aji,dq,sq,pq,f_d,acc,f_c_pos,f_c_neg,pos_pq,neg_pq,tmp\n')
    avg_dice, avg_aji, avg_dq, avg_sq, avg_pq, f_d, acc, f_c_pos, f_c_neg, pos_pq,neg_pq,tmp = do_eval_model_lusc(opt)
    with open(record_path, 'a') as f:
        f.write(f'{avg_dice:0.6},{avg_aji:0.6},{avg_dq:0.6},{avg_sq:0.6},{avg_pq:0.6},{f_d:0.6},{acc:0.6},{f_c_pos:0.6},{f_c_neg:0.6},{pos_pq:0.6},{neg_pq:0.6},{tmp:0.6}\n')


def parse_pannuke_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default=r'')
    parser.add_argument('--checkpoint-path', type=str, default=r'')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--num-types', type=int, default=6)
    parser.add_argument('--record-dir', type=str, default=r'./outputs/records')
    return parser.parse_args()

def parse_lizard_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default=r'')
    parser.add_argument('--checkpoint-path', type=str, default=r'')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--num-types', type=int, default=7)
    parser.add_argument('--record-dir', type=str, default=r'./outputs/records')
    return parser.parse_args()

def parse_lusc_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default=r'D:\CJH\data\generated\半监督LUSC\new_data_patches\unlabeled_test')
    parser.add_argument('--lusc-ann-dir', type=str, default=r'D:\CJH\data\generated\半监督LUSC\new_data_formal\unlabeled')
    parser.add_argument('--checkpoint-path', type=str, default=r'')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--num-types', type=int, default=3)
    parser.add_argument('--record-dir', type=str, default=r'./outputs/records')
    return parser.parse_args()

def main():
    dataset_name = 'pannuke'
    if dataset_name == 'pannuke':
        opt = parse_pannuke_opt()
    elif dataset_name == 'lizard':
        opt = parse_lizard_opt()
    elif dataset_name == 'lusc':
        opt = parse_lusc_opt()
    else:
        print("Unknown dataset name")
        exit(1)
    opt.dataset_name = dataset_name

    if dataset_name == 'pannuke':
        eval_model_pannuke(opt)
    elif dataset_name == 'lizard':
        eval_model_lizard(opt)
    elif dataset_name == 'lusc':
        eval_model_lusc(opt)


if __name__ == '__main__':
    main()
