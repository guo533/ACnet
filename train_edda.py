#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

cv2.setNumThreads(0)

import os

import time
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
import torch.optim as optim
from loguru import logger
from tensorboardX import SummaryWriter


import base.base_utils as utils
import base.losses as losses
from base.base_dataset import FileLoaderExt
from model.hover_net import HoVerNetExt
from sl.models.edda import EDDA, EDDA_Stage1


def get_file_list(opt):
    train_filenames = os.listdir(opt.dataset_dir)
    train_file_list = [os.path.join(opt.dataset_dir, filename) for filename in train_filenames]
    train_file_list.sort()
    return train_file_list


def train1(opt):
    cur_time = utils.cur_time_str()

    checkpoint_dir = os.path.join(opt.log_dir, cur_time, 'checkpoints')
    record_dir = os.path.join(opt.log_dir, cur_time, 'records')

    utils.mkdir(opt.log_dir)

    utils.remove_useless_logs(opt.log_dir)

    utils.mkdir(checkpoint_dir)
    utils.mkdir(record_dir)

    net = EDDA_Stage1(input_ch=3, num_types=opt.num_types, pretrained_encoder_path=opt.pretrained_backbone_path).to(utils.device())
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)
    training_file_list = get_file_list(opt)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 25, verbose=True)
    train_input_dataset = FileLoaderExt(training_file_list, with_type=opt.num_types is not None, input_shape=(opt.patch_size, opt.patch_size),
                                        mask_shape=(opt.mask_size, opt.mask_size), mode='train', gen_hv=True)

    train_dataloader = DataLoader(
        train_input_dataset,
        num_workers=opt.workers,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True
    )

    scaler = amp.GradScaler()
    tfwriter = SummaryWriter(log_dir=record_dir)
    output_weight_path = ''
    for epoch in range(50):
        epoch_start_time = time.time()
        batch_loss_record = np.array([], dtype=np.float32)

        for idx, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()
            imgs = batch_data['img'].to(utils.device()).type(torch.float32)
            true_np = batch_data["np_map"].to(utils.device()).type(torch.int64)
            true_tp = batch_data['tp_map'].to(utils.device()).type(torch.int64)
            true_hv = batch_data["hv_map"].to(utils.device()).type(torch.float32)
            imgs = imgs.permute(0, 3, 1, 2).contiguous()
            imgs /= 255.0
            true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
            true_tp_onehot = F.one_hot(true_tp, num_classes=opt.num_types).type(torch.float32)
            with amp.autocast():
                pred_dict = net(imgs)
                pred_dict = OrderedDict([(k, v.permute(0, 2, 3, 1).contiguous()) for k, v in pred_dict.items()])
                pred_np = F.softmax(pred_dict['np'], dim=-1)
                pred_tp = F.softmax(pred_dict['tp'], dim=-1)
                pred_hv = pred_dict['hv']

                np_loss = losses.xentropy_loss(true_np_onehot, pred_np) + losses.dice_loss(true_np_onehot, pred_np)
                tp_loss = losses.xentropy_loss(true_tp_onehot, pred_tp) + losses.dice_loss(true_tp_onehot, pred_tp)
                hv_loss = losses.mse_loss(true_hv, pred_hv) + losses.msge_loss(true_hv, pred_hv, true_np_onehot[..., 1])
                loss = np_loss + tp_loss + hv_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_loss_record = np.append(batch_loss_record, loss.item())

        avg_epoch_loss = np.mean(batch_loss_record)

        tfwriter.add_scalar('learning_rate', scheduler.get_last_lr(), epoch)
        scheduler.step()

        tfwriter.flush()
        tfwriter.add_scalar('loss', avg_epoch_loss, epoch)
        tfwriter.flush()

        cost_time = time.time() - epoch_start_time
        logger.info(f'epoch: {epoch}\tloss: {avg_epoch_loss: 0.6}\tcost: {cost_time: 0.6}')
        if epoch == 49:
            checkpoint_name = f'weight1_{epoch}.tar'
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save({
                'net_state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint_path)
            logger.info(f'save checkpoint at {checkpoint_path}')
            output_weight_path = checkpoint_path
    tfwriter.close()
    return output_weight_path

def train2(opt):
    cur_time = utils.cur_time_str()

    checkpoint_dir = os.path.join(opt.log_dir, cur_time, 'checkpoints')
    record_dir = os.path.join(opt.log_dir, cur_time, 'records')

    utils.mkdir(opt.log_dir)

    utils.remove_useless_logs(opt.log_dir)

    utils.mkdir(checkpoint_dir)
    utils.mkdir(record_dir)

    net = EDDA(input_ch=3, num_types=opt.num_types, pretrained_encoder_path=opt.pretrained_backbone_path).to(utils.device())
    net.train()

    if opt.pretrained_checkpoint_path:
        net_state_dict = torch.load(opt.pretrained_checkpoint_path, map_location=utils.device())['net_state_dict']
        load_feedback = net.load_state_dict(net_state_dict, strict=False)
    else:
        logger.info('Net is random initialized')

    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)
    training_file_list = get_file_list(opt)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 25, verbose=True)
    train_input_dataset = FileLoaderExt(training_file_list, with_type=opt.num_types is not None, input_shape=(opt.patch_size, opt.patch_size),
                                        mask_shape=(opt.mask_size, opt.mask_size), mode='train', gen_hv=True)

    train_dataloader = DataLoader(
        train_input_dataset,
        num_workers=opt.workers,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True
    )

    scaler = amp.GradScaler()
    tfwriter = SummaryWriter(log_dir=record_dir)

    for epoch in range(opt.epochs):
        epoch_start_time = time.time()
        batch_loss_record = np.array([], dtype=np.float32)

        for idx, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()
            imgs = batch_data['img'].to(utils.device()).type(torch.float32)
            true_np = batch_data["np_map"].to(utils.device()).type(torch.int64)
            true_tp = batch_data['tp_map'].to(utils.device()).type(torch.int64)
            true_hv = batch_data["hv_map"].to(utils.device()).type(torch.float32)
            imgs = imgs.permute(0, 3, 1, 2).contiguous()
            imgs /= 255.0
            true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
            true_tp_onehot = F.one_hot(true_tp, num_classes=opt.num_types).type(torch.float32)
            with amp.autocast():
                pred_dict = net(imgs)
                pred_dict = OrderedDict([(k, v.permute(0, 2, 3, 1).contiguous()) for k, v in pred_dict.items()])
                pred_np = F.softmax(pred_dict['np'], dim=-1)
                pred_tp = F.softmax(pred_dict['tp'], dim=-1)
                pred_hv = pred_dict['hv']

                np_loss = losses.xentropy_loss(true_np_onehot, pred_np) + losses.dice_loss(true_np_onehot, pred_np)
                tp_loss = losses.xentropy_loss(true_tp_onehot, pred_tp) + losses.dice_loss(true_tp_onehot, pred_tp)
                hv_loss = losses.mse_loss(true_hv, pred_hv) + losses.msge_loss(true_hv, pred_hv, true_np_onehot[..., 1])
                loss = np_loss + tp_loss + 2 * hv_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_loss_record = np.append(batch_loss_record, loss.item())

        avg_epoch_loss = np.mean(batch_loss_record)

        tfwriter.add_scalar('learning_rate', scheduler.get_last_lr(), epoch)
        scheduler.step()

        tfwriter.flush()
        tfwriter.add_scalar('loss', avg_epoch_loss, epoch)
        tfwriter.flush()

        cost_time = time.time() - epoch_start_time
        logger.info(f'epoch: {epoch}\tloss: {avg_epoch_loss: 0.6}\tcost: {cost_time: 0.6}')
        if epoch >= opt.start_save_epoch or epoch == 39:
            if epoch == 39:
                checkpoint_name = f'weight2_pretrained.tar'
            else:
                checkpoint_name = f'weight2_{epoch}.tar'
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save({
                'net_state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint_path)
            logger.info(f'save checkpoint at {checkpoint_path}')
    tfwriter.close()


def parse_pannuke_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default=r'')
    parser.add_argument('--pretrained-backbone-path', type=str, default=r'pretrained/resnet50-0676ba61.pth')
    parser.add_argument('--pretrained-checkpoint-path', type=str, default=r"")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--mask-size', type=int, default=256)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--num-types', type=int, default=6)
    parser.add_argument('--use-affine', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('--start-save-epoch', type=int, default=45)
    parser.add_argument('--log-dir', type=str, default=r'./logs')
    return parser.parse_args()


def parse_lizard_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default=r'')
    parser.add_argument('--pretrained-backbone-path', type=str, default=r'pretrained/resnet50-0676ba61.pth')
    parser.add_argument('--pretrained-checkpoint-path', type=str, default=r"")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--mask-size', type=int, default=256)
    parser.add_argument('--num-types', type=int, default=7)
    parser.add_argument('--use-affine', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('--start-save-epoch', type=int, default=45)
    parser.add_argument('--log-dir', type=str, default=r'./logs')
    return parser.parse_args()

def parse_lusc_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default=r'')
    parser.add_argument('--pretrained-backbone-path', type=str, default=r'pretrained/resnet50-0676ba61.pth')
    parser.add_argument('--pretrained-checkpoint-path', type=str, default=r"")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--mask-size', type=int, default=256)
    parser.add_argument('--num-types', type=int, default=3)
    parser.add_argument('--use-affine', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('--start-save-epoch', type=int, default=45)
    parser.add_argument('--log-dir', type=str, default=r'./logs')
    return parser.parse_args()


def main():
    # dataset_name should be one of pannuke, lizard and lusc
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
    output_weight_path = train1(opt)
    opt.pretrained_checkpoint_path = output_weight_path
    train2(opt)


if __name__ == '__main__':
    main()
