#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import time
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loguru import logger
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.use('Agg')

import base.base_utils as utils
import base.post_proc as post_proc
import base.metrics as metrics
from base.metrics import NucleiTypeStat
from base.base_dataset import FileLoaderExt, LUSCTestFileLoaderExt
from sl.models.edda import EDDA
from base.viz import visualize_instances_dict

def do_paint_model_pannuke(opt):
    net = EDDA(input_ch=3, num_types=opt.num_types, pretrained_encoder_path=opt.pretrained_backbone_path).to(utils.device())
    net_state_dict = torch.load(opt.checkpoint_path, map_location=utils.device())['net_state_dict']
    net.load_state_dict(net_state_dict)
    net.eval()

    filenames = os.listdir(opt.dataset_dir)
    filenames.sort(key=lambda x: int(x[:x.find('.npy')]))

    file_list = [os.path.join(opt.dataset_dir, filename) for filename in filenames]

    val_input_dataset = FileLoaderExt(file_list, with_type=True, input_shape=(opt.patch_size, opt.patch_size),
                                   mask_shape=(opt.mask_size, opt.mask_size), mode='test', gen_hv=False)
    val_dataloader = DataLoader(
        val_input_dataset,
        num_workers=opt.workers,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False
    )

    file_idx = 0
    for batch_idx, batch_data in enumerate(val_dataloader):
        imgs = batch_data['img'].to(utils.device()).type(torch.float32)
        anns = utils.ensure_array(batch_data['ann'].to(utils.device()).type(torch.int64))
        imgs = imgs.permute(0, 3, 1, 2).contiguous()
        imgs /= 255.0
        with torch.no_grad():
            raw_pred_dict = net(imgs)
            raw_pred_dict = OrderedDict([(k, v.permute(0, 2, 3, 1).contiguous()) for k, v in raw_pred_dict.items()])
            raw_pred_dict['tp'] = utils.convert_pred_tp_map_to_one_channel_tp_map(raw_pred_dict['tp'])
            raw_pred_dict['np'] = F.softmax(raw_pred_dict['np'], dim=-1)[..., 1:]
            pred_dict = OrderedDict()
            pred_dict['tp'] = raw_pred_dict['tp']
            pred_dict['np'] = raw_pred_dict['np']
            pred_dict['hv'] = raw_pred_dict['hv']

        pred = utils.ensure_array(torch.cat(list(pred_dict.values()), axis=-1))
        pred_list = np.split(pred, pred.shape[0], axis=0)
        num_patches = len(pred_list)

        for i in range(num_patches):
            pred_map = pred_list[i]
            true_inst_map = np.squeeze(anns[i])[..., 0]
            true_type_map = np.squeeze(anns[i])[..., 1]
            pred_inst_type_map = post_proc.produce_inst_type_maps_for_lizard_and_lusc(pred_map, num_types=opt.num_types)
            true_inst_type_map = utils.create_ann_inst_type_map(true_inst_map, true_type_map, opt.num_types)
            pred_inst, inst_info_dict = post_proc.process_for_lizard_and_lusc(pred_map, num_types=opt.num_types, return_centroids=True)
            true_inst_info_dict = utils.get_true_inst_info_dict_for_painting_nuclei(true_inst_type_map)

            type_info_dict = {
                0: ('background', (0, 0, 0)),
                1: ('neoplastic', (254, 0,0)),
                2: ('inflammatory', (0, 253, 253)),
                3: ('connective', (178, 20, 238)),
                4: ('dead', (0, 255, 0)),
                5: ('non-neoplastic epithelial', (236, 226, 149))
            }
            pred_overlay = visualize_instances_dict(batch_data['img'][i], inst_info_dict, type_colour=type_info_dict)
            real_overlay = visualize_instances_dict(batch_data['img'][i], true_inst_info_dict, type_colour=type_info_dict)
            print(f'{file_idx}')
            np.save(os.path.join(opt.paint_dir, rf'/pred_{file_idx}.npy'), pred_overlay)
            np.save(os.path.join(opt.paint_dir, rf'/real_{file_idx}.npy'), real_overlay)
            file_idx += 1

def do_paint_model_lizard(opt):
    net = EDDA(input_ch=3, num_types=opt.num_types, pretrained_encoder_path=opt.pretrained_backbone_path).to(utils.device())
    net_state_dict = torch.load(opt.checkpoint_path, map_location=utils.device())['net_state_dict']
    net.load_state_dict(net_state_dict)
    net.eval()

    filenames = os.listdir(opt.dataset_dir)
    filenames.sort(key=lambda x: int(x[:x.find('.npy')]))

    file_list = [os.path.join(opt.dataset_dir, filename) for filename in filenames]

    val_input_dataset = FileLoaderExt(file_list, input_shape=(opt.patch_size, opt.patch_size),
                                   mask_shape=(opt.mask_size, opt.mask_size), mode='test', gen_hv=False)
    val_dataloader = DataLoader(
        val_input_dataset,
        num_workers=opt.workers,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False
    )

    file_idx = 0
    for batch_idx, batch_data in enumerate(val_dataloader):
        imgs = batch_data['img'].to(utils.device()).type(torch.float32)
        anns = utils.ensure_array(batch_data['ann'].to(utils.device()).type(torch.int64))
        imgs = imgs.permute(0, 3, 1, 2).contiguous()
        imgs /= 255.0
        with torch.no_grad():
            raw_pred_dict = net(imgs)
            raw_pred_dict = OrderedDict([(k, v.permute(0, 2, 3, 1).contiguous()) for k, v in raw_pred_dict.items()])
            raw_pred_dict['tp'] = utils.convert_pred_tp_map_to_one_channel_tp_map(raw_pred_dict['tp'])
            raw_pred_dict['np'] = F.softmax(raw_pred_dict['np'], dim=-1)[..., 1:]

            pred_dict = OrderedDict()
            pred_dict['tp'] = raw_pred_dict['tp']
            pred_dict['np'] = raw_pred_dict['np']
            pred_dict['hv'] = raw_pred_dict['hv']

        pred = utils.ensure_array(torch.cat(list(pred_dict.values()), axis=-1))
        pred_list = np.split(pred, pred.shape[0], axis=0)
        num_patches = len(pred_list)

        for i in range(num_patches):
            pred_map = pred_list[i]
            true_inst_map = np.squeeze(anns[i])[..., 0]
            true_type_map = np.squeeze(anns[i])[..., 1]
            pred_inst_type_map = post_proc.produce_inst_type_maps_for_lizard_and_lusc(pred_map, num_types=opt.num_types)
            true_inst_type_map = utils.create_ann_inst_type_map(true_inst_map, true_type_map, opt.num_types)
            pred_inst, inst_info_dict = post_proc.process_for_lizard_and_lusc(pred_map, num_types=opt.num_types, return_centroids=True)
            true_inst_info_dict = utils.get_true_inst_info_dict_for_painting_nuclei(true_inst_type_map)

            type_info_dict = {
                0: ('background', (0, 0, 0)),
                1: ('neutrophil', (129, 0, 0)),
                2: ('epithelial', (49, 100, 250)),
                3: ('lymphocyte', (255, 147, 59)),
                4: ('plasma', (255, 255, 0)),
                5: ('eosinophil', (0, 0, 129)),
                6: ('connective', (178, 20, 238))
            }
            pred_overlay = visualize_instances_dict(batch_data['img'][i], inst_info_dict, type_colour=type_info_dict)
            real_overlay = visualize_instances_dict(batch_data['img'][i], true_inst_info_dict, type_colour=type_info_dict)
            print(f'{file_idx}')
            np.save(os.path.join(opt.paint_dir, rf'/pred_{file_idx}.npy'), pred_overlay)
            np.save(os.path.join(opt.paint_dir, rf'/real_{file_idx}.npy'), real_overlay)
            file_idx += 1

def do_paint_model_lusc(opt):
    net = EDDA(input_ch=3, num_types=opt.num_types, pretrained_encoder_path=opt.pretrained_backbone_path).to(utils.device())
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

    file_idx = 0
    cost_time = 0.0
    for full_file_idx, val_dataloader in enumerate(val_dataloaders):
        for batch_idx, batch_data in enumerate(val_dataloader):
            imgs = batch_data['img'].to(utils.device()).type(torch.float32)
            imgs = imgs.permute(0, 3, 1, 2).contiguous()
            imgs /= 255.0
            with torch.no_grad():
                start_time = time.time()
                raw_pred_dict = net(imgs)
                raw_pred_dict = OrderedDict([(k, v.permute(0, 2, 3, 1).contiguous()) for k, v in raw_pred_dict.items()])
                raw_pred_dict['tp'] = utils.convert_pred_tp_map_to_one_channel_tp_map(raw_pred_dict['tp'])
                raw_pred_dict['np'] = F.softmax(raw_pred_dict['np'], dim=-1)[..., 1:]

                pred_dict = OrderedDict()
                pred_dict['tp'] = raw_pred_dict['tp']
                pred_dict['np'] = raw_pred_dict['np']
                pred_dict['hv'] = raw_pred_dict['hv']

            pred = utils.ensure_array(torch.cat(list(pred_dict.values()), axis=-1))
            pred_list = np.split(pred, pred.shape[0], axis=0)
            full_pred_map = np.zeros((1, 1024, 1024, 4))
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
            pred_inst_type_map = post_proc.produce_inst_type_maps_for_lizard_and_lusc(fix_full_pred_map, num_types=opt.num_types)
            true_inst_type_map = utils.create_ann_inst_type_map(true_inst_map, true_type_map, opt.num_types)
            pred_inst, inst_info_dict = post_proc.process_for_lizard_and_lusc(fix_full_pred_map, num_types=opt.num_types, return_centroids=True)
            cost_time += (time.time() - start_time)
            true_inst_info_dict = utils.get_true_inst_info_dict_for_painting_nuclei(true_inst_type_map)

            type_info_dict = {
                0: ('background', (0, 0, 0)),
                1: ('pos', (75, 0, 130)),
                2: ('neg', (85, 108, 48))
            }

            full_size_ann = np.load(ann_file_list[full_file_idx])
            pred_overlay = visualize_instances_dict(full_size_ann[..., :3], inst_info_dict, type_colour=type_info_dict)[:1000, :1000]
            real_overlay = visualize_instances_dict(full_size_ann[..., :3], true_inst_info_dict, type_colour=type_info_dict)[:1000, :1000]
            print(f'{file_idx}')
            np.save(os.path.join(opt.paint_dir, rf'/pred_{file_idx}.npy'), pred_overlay)
            np.save(os.path.join(opt.paint_dir, rf'/real_{file_idx}.npy'), real_overlay)
            file_idx += 1


def parse_pannuke_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--original-image-size', type=int, default=256)
    parser.add_argument('--pretrained-backbone-path', type=str, default=r'pretrained/resnet50-0676ba61.pth')
    parser.add_argument('--checkpoint-path', type=str, default=r'')
    parser.add_argument('--num-types', type=int, default=6)
    parser.add_argument('--dataset-dir', type=str, default=r'')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--mask-size', type=int, default=256)
    parser.add_argument('--paint-dir', type=str, default=r'')
    return parser.parse_args()

def parse_lizard_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--original-image-size', type=int, default=256)
    parser.add_argument('--pretrained-backbone-path', type=str, default=r'pretrained/resnet50-0676ba61.pth')
    parser.add_argument('--checkpoint-path', type=str, default=r'')
    parser.add_argument('--num-types', type=int, default=7)
    parser.add_argument('--dataset-dir', type=str, default=r'')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--mask-size', type=int, default=256)
    parser.add_argument('--paint-dir', type=str, default=r'')
    return parser.parse_args()

def parse_lusc_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--original-image-size', type=int, default=256)
    parser.add_argument('--pretrained-backbone-path', type=str, default=r'pretrained/resnet50-0676ba61.pth')
    parser.add_argument('--checkpoint-path', type=str, default=r'')
    parser.add_argument('--num-types', type=int, default=3)
    parser.add_argument('--dataset-dir', type=str, default=r'D:\CJH\data\generated\半监督LUSC\new_data_patches\unlabeled_test')
    parser.add_argument('--lusc-ann-dir', type=str, default=r'D:\CJH\data\generated\半监督LUSC\new_data_formal\unlabeled')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--mask-size', type=int, default=256)
    parser.add_argument('--paint-dir', type=str, default=r'')
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
        logger.error("Unknown dataset name")
        exit(1)
    opt.dataset_name = dataset_name

    if dataset_name == 'pannuke':
        do_paint_model_pannuke(opt)
    elif dataset_name == 'lizard':
        do_paint_model_lizard(opt)
    elif dataset_name == 'lusc':
        do_paint_model_lusc(opt)


if __name__ == '__main__':
    main()
