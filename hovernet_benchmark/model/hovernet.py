from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import hovernet_benchmark.model.net_utils as net_utils
from hovernet_benchmark.model.net_utils import (Net, ResidualBlock, TFSamepaddingLayer, UpSample2x)
import hovernet_benchmark.utils as utils


class HoVerNet(Net):

    def __init__(self, input_ch=3, num_types=None, freeze=False, mode='original'):
        super(HoVerNet, self).__init__()
        self.mode = mode
        self.freeze = freeze
        self.num_types = num_types
        self.output_ch = 3 if num_types is None else 4

        assert mode == 'original' or mode == 'fast', \
            'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode

        module_list = [
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == 'fast':  # prepend the padding for `fast` mode
            module_list = [("pad", TFSamepaddingLayer(ksize=7, stride=1))] + module_list

        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1)
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2)
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2)
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2)

        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        ksize = 5 if mode == 'original' else 3

        if num_types is None:
            decoder = OrderedDict(
                [
                    ("np", net_utils.create_decoder_branch(out_ch=2, ksize=ksize)),
                    ("hv", net_utils.create_decoder_branch(out_ch=2, ksize=ksize))
                ]
            )
        else:
            decoder = OrderedDict(
                [
                    ("np", net_utils.create_decoder_branch(out_ch=2, ksize=ksize)),
                    ("tp", net_utils.create_decoder_branch(out_ch=num_types, ksize=ksize)),
                    ("hv", net_utils.create_decoder_branch(out_ch=2, ksize=ksize))
                ]
            )

        self.decoder = nn.ModuleDict(decoder)

        self.upsample2x = UpSample2x()
        self.weights_init()

    def forward(self, imgs):
        if self.training:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0, self.freeze)
            with torch.set_grad_enabled(not self.freeze):
                if self.freeze:
                    d1 = self.d1(d0)
                    d2 = self.d2(d1)
                    d3 = self.d3(d2)
                else:
                    d1 = checkpoint(self.d1, d0)
                    d2 = checkpoint(self.d2, d1)
                    d3 = checkpoint(self.d3, d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]
        else:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0)
            d1 = self.d1(d0)
            d2 = self.d2(d1)
            d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]

        if self.mode == 'original':
            d[0] = utils.crop_op(d[0], [184, 184])
            d[1] = utils.crop_op(d[1], [72, 72])
        else:
            d[0] = utils.crop_op(d[0], [92, 92])
            d[1] = utils.crop_op(d[1], [36, 36])

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = checkpoint(self.upsample2x, d[-1]) + d[-2]
            u3 = checkpoint(branch_desc[0], u3)

            u2 = checkpoint(self.upsample2x, u3) + d[-3]

            u2 = checkpoint(branch_desc[1], u2)

            u1 = checkpoint(self.upsample2x, u2) + d[-4]
            u1 = checkpoint(branch_desc[2], u1)

            u0 = checkpoint(branch_desc[3], u1)
            out_dict[branch_name] = u0
        return out_dict
