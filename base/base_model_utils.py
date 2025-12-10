import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet
from loguru import logger


class BaseModel(nn.Module):
    """ A base class provides a common weight initialisation scheme."""

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            # ! Fixed the type checking
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            if "norm" in classname.lower():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if "linear" in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return x


class DenseBlockExt(BaseModel):


    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, split=1):
        super(DenseBlockExt, self).__init__()
        assert len(unit_ksize) == len(unit_ch), "Unbalance Unit Info"

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch

        unit_in_ch = in_ch
        pad_vals = [v // 2 for v in unit_ksize]
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            self.units.append(
                nn.Sequential(
                    nn.BatchNorm2d(unit_in_ch, eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        unit_in_ch, unit_ch[0], unit_ksize[0],
                        stride=1, padding=pad_vals[0], bias=False,
                    ),
                    nn.BatchNorm2d(unit_ch[0], eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        unit_ch[0], unit_ch[1], unit_ksize[1],
                        stride=1, padding=pad_vals[1], bias=False,
                        groups=split,
                    ),
                )
            )
            unit_in_ch += unit_ch[1]

        self.blk_bna = nn.Sequential(
            nn.BatchNorm2d(unit_in_ch, eps=1e-5),
            nn.ReLU(inplace=True)
        )

    def out_ch(self):
        return self.in_ch + self.nr_unit * self.unit_ch[-1]

    def forward(self, prev_feat):
        for idx in range(self.nr_unit):
            new_feat = self.units[idx](prev_feat)
            prev_feat = torch.cat([prev_feat, new_feat], dim=1)
        prev_feat = self.blk_bna(prev_feat)

        return prev_feat

class UpSample2x(nn.Module):
    """Upsample input by a factor of 2.

    Assume input is of NCHW, port FixedUnpooling from TensorPack.
    """

    def __init__(self):
        super(UpSample2x, self).__init__()
        self.register_buffer(
            "unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32"))
        )
        self.unpool_mat.unsqueeze(0)

    def forward(self, x):
        input_shape = list(x.shape)
        x = x.unsqueeze(-1)  # bchwx1
        mat = self.unpool_mat.unsqueeze(0)  # 1xshxsw
        ret = torch.tensordot(x, mat, dims=1)  # bxcxhxwxshxsw
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))
        return ret


class ResNetExt(ResNet):

    def __init__(self, block, layers, in_channels=3, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(ResNetExt, self).__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual,
                                        groups=groups, width_per_group=width_per_group,
                                        replace_stride_with_dilation=replace_stride_with_dilation,
                                        norm_layer=norm_layer)
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=1, padding=3)

    def forward(self, x, freeze):
        if self.training:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            with torch.set_grad_enabled(not freeze):
                if freeze:
                    x1 = x = self.layer1(x)
                    x2 = x = self.layer2(x)
                    x3 = x = self.layer3(x)
                    x4 = x = self.layer4(x)
                else:
                    x1 = x = checkpoint(self.layer1, x)
                    x2 = x = checkpoint(self.layer2, x)
                    x3 = x = checkpoint(self.layer3, x)
                    x4 = x = checkpoint(self.layer4, x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x1 = x = self.layer1(x)
            x2 = x = self.layer2(x)
            x3 = x = self.layer3(x)
            x4 = x = self.layer4(x)
        return x1, x2, x3, x4


def resnet50(num_input_channels, pretrained_path=None):
    model = ResNetExt(ResNetBottleneck, [3, 4, 6, 3], num_input_channels)

    if pretrained_path is None:
        logger.info('resnet50 is random initialized')
        return model

    if os.path.exists(pretrained_path):
        net_state_dict = torch.load(pretrained_path)
        missing_keys, unexpected_keys = model.load_state_dict(net_state_dict, strict=False)
        return model
    logger.error(f'{pretrained_path} is invalid')
    return model


def create_decoder_branch_ext(out_ch=2, ksize=3):
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
        nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
    ]
    u2 = nn.Sequential(*module_list)

    module_list = [
        nn.Conv2d(256, 64, ksize, stride=1, padding=pad, bias=False),
    ]
    u1 = nn.Sequential(*module_list)

    module_list = [
        nn.BatchNorm2d(64, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),
    ]
    u0 = nn.Sequential(*module_list)

    decoder = nn.Sequential(
        OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)])
    )
    return decoder
