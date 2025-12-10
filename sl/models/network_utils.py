import os
from collections import OrderedDict

import mmcv
import mmcv.ops as ops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet
from loguru import logger
from base.base_model_utils import DenseBlockExt
import segmentation_models_pytorch.base.modules as md


class DownDim(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownDim, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(self.in_channels, self.out_channels, bias=False)

    def forward(self, features):
        """
        Parameters
        ----------
        features: torch.FloatTensor
            the dimension of features is [B, C, H, W]
        """
        decomposed_features = []
        for feature in features:
            x = torch.permute(feature, (1, 2, 0)).contiguous()
            reduced_x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
            reduced_x = self.linear(reduced_x)
            x = torch.reshape(reduced_x, (x.shape[0], x.shape[1], self.out_channels))
            x = torch.permute(x, (2, 0, 1)).contiguous()
            decomposed_features.append(torch.unsqueeze(x, dim=0))
        return torch.cat(decomposed_features, dim=0)


class DAModule(nn.Module):

    def __init__(self, in_channels, reduced_channels):
        super(DAModule, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True))
        self.deform_conv = nn.Sequential(ops.DeformConv2dPack(in_channels, 128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True))
        reduced_channels = 128 * 2 + 3 - 10
        self.tucker_layer = DownDim(128 * 2 + 3, reduced_channels)
        channels = reduced_channels + in_channels
        self.out = nn.Conv2d(channels, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)

    def forward(self, x, imgs):
        x1 = self.conv(x)
        x2 = self.deform_conv(x)
        imgs = F.interpolate(imgs, scale_factor=(0.5))
        features = self.tucker_layer(torch.cat((x1, imgs, x2), dim=1))
        features = torch.cat((x, features), dim=1)
        out = self.out(features)
        return out
    

def create_edda_np_decoder_branch_ext(ksize=3, middle_channels=64):
    pad = ksize // 2
    module_list = [
        nn.Conv2d(1024, 256, ksize, stride=1, padding=pad, bias=False),
        DenseBlockExt(256, [1, ksize], [128, 32], 8, split=4),
        nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
    ]
    u3 = nn.Sequential(*module_list)

    module_list = [
        nn.Conv2d(512, 128, ksize, stride=1, padding=pad, bias=False),
        DenseBlockExt(128, [1, ksize], [128, 32], 4, split=4),
        nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False)
    ]
    u2 = nn.Sequential(*module_list)

    module_list = [
        nn.Conv2d(256, middle_channels, ksize, stride=1, padding=pad, bias=False)
    ]
    u1 = nn.Sequential(*module_list)

    module_list = [
        nn.BatchNorm2d(middle_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 2, 1, stride=1, padding=0, bias=True)
    ]
    u0 = nn.Sequential(*module_list)

    decoder = nn.Sequential(
        OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)])
    )
    return decoder


def create_edda_hv_decoder_branch_ext(ksize=3, middle_channels=64):
    pad = ksize // 2
    module_list = [
        nn.Conv2d(1024, 256, ksize, stride=1, padding=pad, bias=False),
        DenseBlockExt(256, [1, ksize], [128, 32], 8, split=4),
        nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
    ]
    u3 = nn.Sequential(*module_list)

    module_list = [
        nn.Conv2d(512, 128, ksize, stride=1, padding=pad, bias=False),
        DenseBlockExt(128, [1, ksize], [128, 32], 4, split=4),
        nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False)
    ]
    u2 = nn.Sequential(*module_list)

    module_list = [
        nn.Conv2d(256, middle_channels, ksize, stride=1, padding=pad, bias=False)
    ]
    u1 = nn.Sequential(*module_list)

    module_list = [
        nn.BatchNorm2d(middle_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 2, 1, stride=1, padding=0, bias=True)
    ]
    u0 = nn.Sequential(*module_list)

    decoder = nn.Sequential(
        OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)])
    )
    return decoder


def create_edda_tp_decoder_branch_ext(out_ch=2, ksize=3):
    pad = ksize // 2
    module_list = [
        nn.Conv2d(1024, 256, ksize, stride=1, padding=pad, bias=False),
        DenseBlockExt(256, [1, ksize], [128, 32], 8, split=4),
        nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
    ]
    u3 = nn.Sequential(*module_list)

    module_list = [
        DenseBlockExt(128, [1, ksize], [128, 32], 4, split=4),
        nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
    ]
    u2 = nn.Sequential(*module_list)

    module_list = [
        nn.Conv2d(256, 64, ksize, stride=1, padding=pad, bias=False)
    ]
    u1 = nn.Sequential(*module_list)

    module_list = [
        nn.BatchNorm2d(64, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True)
    ]
    u0 = nn.Sequential(*module_list)

    decoder = nn.Sequential(
        OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)])
    )
    return decoder


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super(CenterBlock, self).__init__(conv1, conv2)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super(DecoderBlock, self).__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class UNetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        xs = []
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            xs.append(x)
        return x, xs
