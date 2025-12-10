#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
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
import hovernet_benchmark.utils as hb_utils
from hovernet_benchmark.data.dataloader import FileLoader
import hovernet_benchmark.process.post_proc as post_proc
import hovernet_benchmark.metrics as metrics
from hovernet_benchmark.model.hovernet_ext import HoVerNetExt

import base.base_utils as utils
from sl.models.edda import EDDA
from base.viz import visualize_instances_dict

from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.nn.functional as F
import numpy as np
import requests
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM

# matplotlib.use('Agg')

import time

def binarize(x):
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

def produce_cam(opt):
    net = EDDA(input_ch=3, num_types=opt.num_types, pretrained_encoder_path=opt.pretrained_backbone_path).to(hb_utils.device())
    net_state_dict = torch.load(opt.checkpoint_path, map_location=hb_utils.device())['net_state_dict']
    net.load_state_dict(net_state_dict)
    net.eval()

    filenames = os.listdir(opt.dataset_dir)
    filenames.sort(key=lambda x: int(x[:x.find('.npy')]))

    file_list = [os.path.join(opt.dataset_dir, filename) for filename in filenames]
    file_list = file_list[opt.file_index:opt.file_index + 1]

    val_input_dataset = FileLoader(file_list, input_shape=(opt.patch_size, opt.patch_size),
                                   mask_shape=(opt.mask_size, opt.mask_size), mode='test', setup_augmentor=False, use_affine=False)
    val_dataloader = DataLoader(
        val_input_dataset,
        num_workers=opt.workers,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False
    )

    file_idx = 0
    cost_time = 0.0
    for batch_idx, batch_data in enumerate(val_dataloader):
        imgs = batch_data['img'].to(hb_utils.device()).type(torch.float32)
        anns = hb_utils.ensure_array(batch_data['ann'].to(hb_utils.device()).type(torch.int64))
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

        pred = hb_utils.ensure_array(torch.cat(list(pred_dict.values()), axis=-1))
        pred_list = np.split(pred, pred.shape[0], axis=0)
        num_patches = len(pred_list)

        for i in range(num_patches):
            pred_map = pred_list[i]
            pred_inst, inst_info_dict = post_proc.process(pred_map, num_types=opt.num_types, return_centroids=True)
            pred_mask = np.array(pred_inst > 0)

            class SegmentationModelOutputWrapper(torch.nn.Module):
                def __init__(self, model):
                    super(SegmentationModelOutputWrapper, self).__init__()
                    self.model = model

                def forward(self, x):
                    return self.model(x)["np"]

            class SemanticSegmentationTarget:
                def __init__(self, mask):
                    self.mask = torch.from_numpy(mask)
                    if torch.cuda.is_available():
                        self.mask = self.mask.cuda()

                def __call__(self, model_output):
                    return (model_output[1, :, :] * self.mask).sum()

            model = SegmentationModelOutputWrapper(net)
            target_layers = [model.model.decoder['np'][-1][2]]
            targets = [SemanticSegmentationTarget(pred_mask)]
            # input_tensor = torch.permute(batch_data['img'], (0, 3, 1, 2))
            with GradCAM(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=imgs, targets=targets)[0, :]
                cam_image = show_cam_on_image(batch_data['img'][0].numpy() / 255.0, grayscale_cam, use_rgb=True)

            # plt.figure(dpi=100)
            # plt.subplot(131)
            # plt.imshow(batch_data['img'][0].numpy())
            # # plt.figure(dpi=100)
            # plt.subplot(132)
            # plt.imshow(cam_image)
            # # plt.figure(dpi=100)
            # plt.subplot(133)
            # # plt.imshow(anns[0, ..., 0])
            # plt.imshow(pred_mask)
            # plt.show()
            # exit(0)

            plt.figure(dpi=100)
            plt.imshow(cam_image)
            plt.axis('off')
            plt.savefig(f'{opt.image_dir}/{opt.file_index}.png', bbox_inches='tight', pad_inches=0)
            plt.show()
            exit(0)


the_cur_time_str = utils.cur_time_str()

def val_single_checkpoint(opt):
    produce_cam(opt)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--original-image-size', type=int, default=256)
    parser.add_argument('--pretrained-backbone-path', type=str, default=r'G:\CJH\workspace\deep_learning\cell_segmentation_v5\cell_segmentation\pretrained\resnet50-0676ba61.pth')
    parser.add_argument('--checkpoint-path', type=str, default=r'')
    parser.add_argument('--record-dir', type=str, default=r'./records')
    parser.add_argument('--num-types', type=int, default=7)
    parser.add_argument('--file-index', type=int, default=0)
    parser.add_argument('--image-dir', type=str, default=r'')
    parser.add_argument('--dataset-dir', type=str, default=r'D:\CJH\data\generated\半监督Lizard\unlabeled')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--mask-size', type=int, default=256)
    return parser.parse_args()


def main():
    opt = parse_opt()
    val_single_checkpoint(opt)

if __name__ == '__main__':
    main()
