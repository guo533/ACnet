#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import albumentations as A
from sl.models.simsiam.net import SimSiam
import sl.data.datasets as datasets
import base.base_utils as utils
import base.base_utils as hb_utils
import sl.losses as sl_losses
import sl.env as env_utils
from sl.models.edda import EDDA
import base.losses as hb_losses
from sl.data.fileloader import FileLoaderExt
import base.post_proc as post_proc


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_augmentation():
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([A.GaussianBlur(), A.MedianBlur(), A.GaussNoise()], p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10,
                             val_shift_limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1,
                                   contrast_limit=0.1, p=0.5),
    ])
    return transforms


def create_perturbation_counterparts(unlabeled_images, raw_pred_dict):
    transforms = get_augmentation()
    num_types = raw_pred_dict['tp'].shape[-1]
    raw_pred_dict['tp'] = hb_utils.convert_pred_tp_map_to_one_channel_tp_map(raw_pred_dict['tp'])
    raw_pred_dict['np'] = F.softmax(raw_pred_dict['np'], dim=-1)[..., 1:]

    pred_dict = OrderedDict()
    pred_dict['tp'] = raw_pred_dict['tp']
    pred_dict['np'] = raw_pred_dict['np']
    pred_dict['hv'] = raw_pred_dict['hv']

    pred = hb_utils.ensure_array(torch.cat(list(pred_dict.values()), axis=-1))
    pred_list = np.split(pred, pred.shape[0], axis=0)
    num_patches = len(pred_list)

    pert_unlabeled_image_list = []
    pert_unlabeled_pseudo_type_map_list = []
    unlabeled_images = utils.ensure_array(unlabeled_images)

    for i in range(num_patches):
        unlabeled_image = unlabeled_images[i]
        pred_map = pred_list[i]
        pred_inst_type_map = post_proc.produce_inst_type_maps(pred_map, num_types=num_types)

        tmp_map = np.zeros((pred_inst_type_map.shape[0], pred_inst_type_map.shape[1]), dtype=np.int64)
        for channel_index in range(pred_inst_type_map.shape[-1]):
            channel_map = (pred_inst_type_map[..., channel_index] > 0).astype('int64')
            tmp_map += channel_map
        background_map = np.zeros((pred_inst_type_map.shape[0], pred_inst_type_map.shape[1]), dtype=np.int64)
        background_map[tmp_map == 0] = 1
        # remove instance ids
        pred_inst_type_map[pred_inst_type_map > 0] = 1
        background_map = background_map[..., None]
        unlabeled_pred_type_map = np.concatenate((background_map, pred_inst_type_map), axis=-1)
        unlabeled_pred_type_map = np.argmax(unlabeled_pred_type_map, axis=-1)
        transformed = transforms(image=unlabeled_image, masks=[unlabeled_pred_type_map])
        pert_unlabeled_image_list.append(transformed['image'][None, ...])
        pert_unlabeled_pseudo_type_map_list.append(transformed['masks'][0][None, ...])

    pert_unlabeled_images = torch.from_numpy(np.concatenate(pert_unlabeled_image_list, axis=0)).to(utils.device()).type(
        torch.float32)
    pert_unlabeled_pseudo_type_maps = torch.from_numpy(np.concatenate(pert_unlabeled_pseudo_type_map_list, axis=0)).to(
        utils.device()).type(torch.int64)
    pert_unlabeled_pseudo_type_maps = F.one_hot(pert_unlabeled_pseudo_type_maps, num_classes=num_types)
    return pert_unlabeled_images, pert_unlabeled_pseudo_type_maps, None, None

def create_postprocessed_output(raw_pred_dict):
    num_types = raw_pred_dict['tp'].shape[-1]
    raw_pred_dict['tp'] = utils.convert_pred_tp_map_to_one_channel_tp_map(raw_pred_dict['tp'])
    raw_pred_dict['np'] = F.softmax(raw_pred_dict['np'], dim=-1)[..., 1:]

    pred_dict = OrderedDict()
    pred_dict['tp'] = raw_pred_dict['tp']
    pred_dict['np'] = raw_pred_dict['np']
    pred_dict['hv'] = raw_pred_dict['hv']

    pred = hb_utils.ensure_array(torch.cat(list(pred_dict.values()), axis=-1))
    pred_list = np.split(pred, pred.shape[0], axis=0)
    num_patches = len(pred_list)

    pred_inst_type_map_list = []
    for i in range(num_patches):
        pred_map = pred_list[i]
        pred_inst_type_map = post_proc.produce_inst_type_maps(pred_map, num_types=num_types)
        pred_inst_type_map_list.append(pred_inst_type_map[None, ...])
    return np.concatenate(pred_inst_type_map_list, axis=0)

def calc_loss_if_not_none(loss_func, true, pred):
    if true is None or pred is None:
        return torch.from_numpy(np.array(0)).to(utils.device())
    return loss_func(true, pred)

def train(opt):
    memory_bank = [0, 0, 0, 0, 0, 0]
    memory_bank_count = [0, 0, 0, 0, 0, 0]
    # Prepare environment
    checkpoint_dir, record_dir = env_utils.setup(opt)

    ################# Networks ################
    simsiam_net = SimSiam(in_dim=opt.num_types).to(utils.device())
    net = EDDA(input_ch=3, num_types=opt.num_types, pretrained_encoder_path=opt.pretrained_encoder_path).to(hb_utils.device())
    net.train()
    simsiam_net.train()

    ################# Optimizers ################
    net_optimizer = optim.Adam(net.parameters(), lr=opt.gen_learning_rate)
    simsiam_net_optimizer = optim.Adam(simsiam_net.parameters(), lr=opt.simsiam_learning_rate)

    net_state_dict = torch.load(opt.pretrained_checkpoint_path, map_location=utils.device())['net_state_dict']
    load_feedback = net.load_state_dict(net_state_dict, strict=True)
    net_optimizer.load_state_dict(torch.load(opt.pretrained_checkpoint_path, map_location=utils.device())['optimizer'])
    print(f'Load checkpoint from {opt.pretrained_checkpoint_path}')
    print(f'Missing variables: \n{load_feedback[0]}')
    print(f'Detected unknown variables: \n{load_feedback[1]}')

    ################# Schedulers ################
    simsiam_net_scheduler = optim.lr_scheduler.StepLR(simsiam_net_optimizer, opt.simsiam_learning_rate_change_freq, verbose=True)

    labeled_file_list = datasets.get_file_list(opt.labeled_dataset_dir)
    unlabeled_file_list = datasets.get_file_list(opt.unlabeled_dataset_dir)

    labeled_dataloader = DataLoader(FileLoaderExt(labeled_file_list, mode='train', with_type=True, input_shape=(256, 256),
                                        mask_shape=(256, 256), use_affine=False), batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=False)
    unlabeled_dataloader = DataLoader(FileLoaderExt(unlabeled_file_list, mode='val', with_type=True, input_shape=(256, 256),
                                        mask_shape=(256, 256), use_affine=False), batch_size=2, shuffle=False, num_workers=opt.workers, pin_memory=False, drop_last=True)
    unlabeled_enumerator = enumerate(unlabeled_dataloader)
    net_scaler = amp.GradScaler()
    simsiam_scaler = amp.GradScaler()
    tfwriter = SummaryWriter(log_dir=record_dir)

    for epoch in range(opt.warmup_epoch + 1, opt.epochs):
        batch_total_loss_record = np.array([], dtype=np.float32)

        if epoch == 49:
            adjust_learning_rate(net_optimizer, (opt.gen_learning_rate / 10))

        for batch_idx, batch_data in enumerate(labeled_dataloader):
            net_optimizer.zero_grad()
            simsiam_net_optimizer.zero_grad()

            labeled_images = batch_data['img'].to(hb_utils.device()).type(torch.float32) / 255.0
            labeled_true_np = batch_data["np_map"].to(hb_utils.device()).type(torch.int64)
            labeled_true_np_onehot = (F.one_hot(labeled_true_np, num_classes=2)).type(torch.float32)
            labeled_true_instance_map = batch_data["instance_map"].to(hb_utils.device()).type(torch.int64)
            labeled_true_tp = batch_data['tp_map'].to(hb_utils.device()).type(torch.int64)
            labeled_true_tp_onehot = F.one_hot(labeled_true_tp, num_classes=opt.num_types).type(torch.float32)
            labeled_true_hv = batch_data["hv_map"].to(hb_utils.device()).type(torch.float32)

            # Convert [B, H, W, C] to [B, C, H, W] if necessary
            labeled_images = utils.bhwc_to_bchw(labeled_images)

            with amp.autocast():
                labeled_pred_dict = net(labeled_images)
                # Convert [B, C, H, W] to [B, H, W, C] if necessary
                labeled_pred_dict = OrderedDict([(k, utils.bchw_to_bhwc(v)) for k, v in labeled_pred_dict.items()])

                # Softmax predicted maps
                labeled_pred_np = F.softmax(labeled_pred_dict['np'], dim=-1)
                labeled_pred_tp = F.softmax(labeled_pred_dict['tp'], dim=-1)
                labeled_pred_hv = labeled_pred_dict['hv']

                labeled_np_loss = hb_losses.xentropy_loss(labeled_true_np_onehot, labeled_pred_np) + hb_losses.dice_loss(labeled_true_np_onehot, labeled_pred_np)
                labeled_tp_loss = hb_losses.xentropy_loss(labeled_true_tp_onehot, labeled_pred_tp) + hb_losses.dice_loss(labeled_true_tp_onehot, labeled_pred_tp)
                labeled_hv_loss = hb_losses.mse_loss(labeled_true_hv, labeled_pred_hv) + hb_losses.msge_loss(labeled_true_hv, labeled_pred_hv, labeled_true_np_onehot[..., 1])

                labeled_loss = labeled_np_loss + labeled_tp_loss + 2 * labeled_hv_loss

                if epoch > 49:
                    labeled_true_instance_map_array = utils.ensure_array(labeled_true_instance_map)
                    labeled_true_tp_array = utils.ensure_array(labeled_true_tp)
                    batch_size = labeled_true_instance_map_array.shape[0]
                    true_inst_type_maps = np.zeros((batch_size, labeled_true_instance_map_array.shape[1],
                                                    labeled_true_instance_map_array.shape[2], opt.num_types - 1),
                                                   dtype=np.int64)

                    for b in range(batch_size):
                        true_inst_type_map = utils.create_ann_inst_type_map(labeled_true_instance_map_array[b],
                                                                            labeled_true_tp_array[b], opt.num_types)
                        true_inst_type_maps[b] = true_inst_type_map
                    labeled_tp_feat = labeled_pred_dict['tp']

                    # calculate prototypes stored in memory bank
                    for c in range(true_inst_type_maps.shape[-1]):
                        class_feat = None
                        class_instance_num = 0
                        for b in range(batch_size):
                            inst_list = np.unique(true_inst_type_maps[b, ..., c])
                            if len(inst_list) > 1:
                                inst_list = inst_list[1:]
                                for inst_id in inst_list:
                                    tmp_matrix = (true_inst_type_maps[b, ..., c] == inst_id)
                                    tmp_matrix = tmp_matrix[..., None]
                                    instance_feat = labeled_tp_feat[b, ...] * torch.from_numpy(tmp_matrix).to(
                                        utils.device())
                                    instance_feat = torch.masked_select(instance_feat,
                                                                        torch.from_numpy(tmp_matrix).to(utils.device()))
                                    instance_feat = F.interpolate(torch.unsqueeze(torch.unsqueeze(instance_feat, 0), 0),
                                                                  5000)
                                    instance_feat = torch.squeeze(instance_feat)
                                    if class_feat is None:
                                        class_feat = instance_feat
                                    else:
                                        class_feat = class_feat + instance_feat
                                    class_instance_num += 1
                                    # instance_feat_list[c].append(instance_feat)
                        if class_feat is not None:
                            ground_truth_guided_prototype = class_feat / class_instance_num
                            if memory_bank[c] is None:
                                memory_bank[c] = ground_truth_guided_prototype.detach()
                            else:
                                memory_bank[c] = memory_bank[c] + ground_truth_guided_prototype.detach()
                            memory_bank_count[c] += 1

                    # calculate labeled confidence prototypes
                    labeled_pred_postprocessed_output = create_postprocessed_output(labeled_pred_dict)
                    labeled_pred_postprocessed_output_prototypes = [None, None, None, None, None, None]
                    for c in range(labeled_pred_postprocessed_output.shape[-1]):
                        class_feat = None
                        class_instance_num = 0
                        for b in range(batch_size):
                            inst_list = np.unique(labeled_pred_postprocessed_output[b, ..., c])
                            if len(inst_list) > 1:
                                inst_list = inst_list[1:]
                                for inst_id in inst_list:
                                    tmp_matrix = (labeled_pred_postprocessed_output[b, ..., c] == inst_id)
                                    tmp_matrix = tmp_matrix[..., None]
                                    instance_feat = labeled_tp_feat[b, ...] * torch.from_numpy(tmp_matrix).to(
                                        utils.device())
                                    instance_feat = torch.masked_select(instance_feat,
                                                                        torch.from_numpy(tmp_matrix).to(
                                                                            utils.device()))
                                    instance_feat = F.interpolate(
                                        torch.unsqueeze(torch.unsqueeze(instance_feat, 0), 0), 5000)
                                    instance_feat = torch.squeeze(instance_feat)
                                    if class_feat is None:
                                        class_feat = instance_feat
                                    else:
                                        class_feat = class_feat + instance_feat
                                    class_instance_num += 1
                                    # instance_feat_list[c].append(instance_feat)
                        if class_feat is not None:
                            labeled_pred_postprocessed_output_prototypes[c] = class_feat / class_instance_num
                    labeled_intra_loss = calc_loss_if_not_none(hb_losses.mse_loss,
                                                               memory_bank[0] / (memory_bank_count[0] + 1e-8),
                                                               labeled_pred_postprocessed_output_prototypes[0]) \
                                         + calc_loss_if_not_none(hb_losses.mse_loss,
                                                                 memory_bank[1] / (memory_bank_count[1] + 1e-8),
                                                                 labeled_pred_postprocessed_output_prototypes[1]) \
                                         + calc_loss_if_not_none(hb_losses.mse_loss,
                                                                 memory_bank[2] / (memory_bank_count[2] + 1e-8),
                                                                 labeled_pred_postprocessed_output_prototypes[2]) \
                                         + calc_loss_if_not_none(hb_losses.mse_loss,
                                                                 memory_bank[3] / (memory_bank_count[3] + 1e-8),
                                                                 labeled_pred_postprocessed_output_prototypes[3]) \
                                         + calc_loss_if_not_none(hb_losses.mse_loss,
                                                                 memory_bank[4] / (memory_bank_count[4] + 1e-8),
                                                                 labeled_pred_postprocessed_output_prototypes[4]) \
                                         + calc_loss_if_not_none(hb_losses.mse_loss,
                                                                 memory_bank[5] / (memory_bank_count[5] + 1e-8),
                                                                 labeled_pred_postprocessed_output_prototypes[5])
                    labeled_loss = labeled_loss + labeled_intra_loss * opt.labeled_intra_lambda

            if epoch <= opt.warmup_epoch:
                net_scaler.scale(labeled_loss).backward()
                net_scaler.step(net_optimizer)
                net_scaler.update()
                batch_total_loss_record = np.append(batch_total_loss_record, labeled_loss.item())
            else:
                try:
                    unlabeled_batch_idx, unlabeled_batch_data = next(unlabeled_enumerator)
                except:
                    unlabeled_enumerator = enumerate(unlabeled_dataloader)
                    unlabeled_batch_idx, unlabeled_batch_data = next(unlabeled_enumerator)

                unlabeled_images = unlabeled_batch_data['img'].to(utils.device()).type(torch.float32) / 255.0
                unlabeled_images = utils.bhwc_to_bchw(unlabeled_images)

                net.eval()
                with torch.no_grad():
                    unlabeled_pred_dict = net(unlabeled_images)
                # net.train()
                unlabeled_pred_dict = OrderedDict([(k, utils.bchw_to_bhwc(v)) for k, v in unlabeled_pred_dict.items()])

                # create perturbation
                pert_unlabeled_images, pert_unlabeled_pseudo_type_maps, unlabeled_pred_decoder_outputs, unlabeled_pred_type_maps = create_perturbation_counterparts(
                    unlabeled_batch_data['img'], unlabeled_pred_dict)

                pert_unlabeled_images = pert_unlabeled_images / 255.0
                pert_unlabeled_images = utils.bhwc_to_bchw(pert_unlabeled_images)

                with amp.autocast():
                    pert_unlabeled_pred_dict = net(pert_unlabeled_images)
                    pert_unlabeled_pred_dict = OrderedDict(
                        [(k, utils.bchw_to_bhwc(v)) for k, v in pert_unlabeled_pred_dict.items()])
                    pert_unlabeled_pred_type_maps = F.softmax(pert_unlabeled_pred_dict['tp'], dim=-1)
                    pert_unlabeled_tp_feat = pert_unlabeled_pred_dict['tp']
                    unlabeled_xentropy_loss = hb_losses.xentropy_loss(pert_unlabeled_pseudo_type_maps, pert_unlabeled_pred_type_maps)

                    # augmented images and the counterpart masks
                    x1 = utils.bhwc_to_bchw(pert_unlabeled_pred_type_maps)
                    # unaugmented images and the counterpart masks
                    x2 = utils.bhwc_to_bchw(pert_unlabeled_pseudo_type_maps)
                    # p1, p2, z1, z2 = simsiam_net(x1, x2)
                    p1, p2, z1, z2 = simsiam_net(x1.type(torch.float32), x2.type(torch.float32))
                    unlabeled_loss = sl_losses.simsiam_loss(p1, z2, p2, z1)
                    gen_loss = labeled_loss

                    if epoch > 49:
                        # calculate labeled confidence prototypes
                        pert_unlabeled_pred_postprocessed_output = create_postprocessed_output(pert_unlabeled_pred_dict)
                        pert_unlabeled_pred_postprocessed_output_prototypes = [None, None, None, None, None, None]
                        pert_batch_size = pert_unlabeled_tp_feat.shape[0]
                        for c in range(pert_unlabeled_pred_postprocessed_output.shape[-1]):
                            class_feat = None
                            class_instance_num = 0
                            for b in range(pert_batch_size):
                                # print(pert_unlabeled_pred_postprocessed_output.shape)
                                inst_list = np.unique(pert_unlabeled_pred_postprocessed_output[b, ..., c])
                                if len(inst_list) > 1:
                                    inst_list = inst_list[1:]
                                    for inst_id in inst_list:
                                        tmp_matrix = (pert_unlabeled_pred_postprocessed_output[b, ..., c] == inst_id)
                                        tmp_matrix = tmp_matrix[..., None]
                                        instance_feat = pert_unlabeled_tp_feat[b, ...] * torch.from_numpy(
                                            tmp_matrix).to(
                                            utils.device())
                                        instance_feat = torch.masked_select(instance_feat,
                                                                            torch.from_numpy(tmp_matrix).to(
                                                                                utils.device()))
                                        instance_feat = F.interpolate(
                                            torch.unsqueeze(torch.unsqueeze(instance_feat, 0), 0),
                                            5000)
                                        instance_feat = torch.squeeze(instance_feat)
                                        if class_feat is None:
                                            class_feat = instance_feat
                                        else:
                                            class_feat = class_feat + instance_feat
                                        class_instance_num += 1
                                        # instance_feat_list[c].append(instance_feat)
                            if class_feat is not None:
                                pert_unlabeled_pred_postprocessed_output_prototypes[c] = class_feat / class_instance_num
                        pert_unlabeled_intra_loss = calc_loss_if_not_none(
                            hb_losses.mse_loss, memory_bank[0] / (memory_bank_count[0] + 1e-8),
                            pert_unlabeled_pred_postprocessed_output_prototypes[0]) \
                                                    + calc_loss_if_not_none(
                            hb_losses.mse_loss, memory_bank[1] / (memory_bank_count[1] + 1e-8),
                            pert_unlabeled_pred_postprocessed_output_prototypes[1]) \
                                                    + calc_loss_if_not_none(
                            hb_losses.mse_loss, memory_bank[2] / (memory_bank_count[2] + 1e-8),
                            pert_unlabeled_pred_postprocessed_output_prototypes[2]) \
                                                    + calc_loss_if_not_none(
                            hb_losses.mse_loss, memory_bank[3] / (memory_bank_count[3] + 1e-8),
                            pert_unlabeled_pred_postprocessed_output_prototypes[3]) \
                                                    + calc_loss_if_not_none(
                            hb_losses.mse_loss, memory_bank[4] / (memory_bank_count[4] + 1e-8),
                            pert_unlabeled_pred_postprocessed_output_prototypes[4]) \
                                                    + calc_loss_if_not_none(
                            hb_losses.mse_loss, memory_bank[5] / (memory_bank_count[5] + 1e-8),
                            pert_unlabeled_pred_postprocessed_output_prototypes[5])

                net_scaler.scale(gen_loss).backward()
                if epoch > 49:
                    loss2 = unlabeled_loss * opt.unlabeled_lambda + unlabeled_xentropy_loss * opt.unlabeled_cross_entropy_lambda + pert_unlabeled_intra_loss * opt.pert_unlabeled_intra_lambda
                    simsiam_scaler.scale(loss2).backward()
                    batch_total_loss_record = np.append(batch_total_loss_record, gen_loss.item() + loss2.item())
                else:
                    loss2 = unlabeled_loss * opt.unlabeled_lambda + unlabeled_xentropy_loss * opt.unlabeled_cross_entropy_lambda
                    simsiam_scaler.scale(loss2).backward()
                    batch_total_loss_record = np.append(batch_total_loss_record, gen_loss.item() + loss2.item())

                net_scaler.step(net_optimizer)
                simsiam_scaler.step(simsiam_net_optimizer)

                net_scaler.update()
                simsiam_scaler.update()

                net_optimizer.zero_grad()
                simsiam_net.zero_grad()

        avg_epoch_total_loss = np.mean(batch_total_loss_record)

        if epoch > opt.warmup_epoch:
            simsiam_net_scheduler.step()
        tfwriter.add_scalar('loss', avg_epoch_total_loss, epoch)
        tfwriter.flush()

        print(f"""epoch: {epoch}
        loss: {avg_epoch_total_loss: 0.6}""")


        if epoch >= opt.start_save_epoch:
            checkpoint_name = f'net_epoch={epoch}.tar'
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save({
                'net_state_dict': net.state_dict(),
                'gen_optimizer': net_optimizer.state_dict(),
                'simsiam_state_dict': simsiam_net.state_dict(),
                'simsiam_optimizer': simsiam_net_optimizer.state_dict(),
            }, checkpoint_path)
            print(f'save checkpoint at {checkpoint_path}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled-dataset-dir', type=str, default=r'D:\CJH\data\generated\半监督Lizard\labeled')
    parser.add_argument('--unlabeled-dataset-dir', type=str, default=r'D:\CJH\data\generated\半监督Lizard\unlabeled')
    parser.add_argument('--pretrained-encoder-path', type=str, default=r'pretrained/resnet50-0676ba61.pth')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--pretrained-checkpoint-path', type=str, default=r'')
    parser.add_argument('--gen-learning-rate', type=float, default=1e-5)
    parser.add_argument('--dis-learning-rate', type=float, default=1e-5)
    parser.add_argument('--simsiam-learning-rate', type=float, default=1e-4)
    parser.add_argument('--num-types', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('--dis-learning-rate-change-freq', type=int, default=30)
    parser.add_argument('--simsiam-learning-rate-change-freq', type=int, default=10)
    parser.add_argument('--unlabeled-lambda', type=float, default=0.01)
    parser.add_argument('--unlabeled-cross-entropy-lambda', type=float, default=0.01)
    parser.add_argument('--labeled-intra-lambda', type=float, default=0.001)
    parser.add_argument('--pert-unlabeled-intra-lambda', type=float, default=0.001)
    parser.add_argument('--adv-lambda', type=float, default=1e-1)
    parser.add_argument('--dis-adv-lambda', type=float, default=1e-2)
    parser.add_argument('--warmup-epoch', type=int, default=39)
    parser.add_argument('--start-save-epoch', type=int, default=40)
    parser.add_argument('--log-dir', type=str, default=r'./outputs/logs')
    return parser.parse_args()


def main():
    opt = parse_opt()
    train(opt)


if __name__ == '__main__':
    main()
