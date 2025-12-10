from collections import OrderedDict

import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import hovernet_benchmark.model.net_utils as net_utils
from hovernet_benchmark.model.net_utils import UpSample2x


class HoVerNetExt(nn.Module):

    def __init__(self, input_ch=3, num_types=None, freeze=False, pretrained_encoder_path=None):
        super(HoVerNetExt, self).__init__()
        self.input_ch = input_ch
        self.num_types = num_types
        self.freeze = freeze
        self.backbone = net_utils.resnet50(self.input_ch, pretrained_path=pretrained_encoder_path)
        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)
        if num_types is None:
            decoder = OrderedDict(
                [
                    ("np", net_utils.create_decoder_branch_ext(out_ch=2, ksize=3)),
                    ("hv", net_utils.create_decoder_branch_ext(out_ch=2, ksize=3))
                ]
            )
        else:
            decoder = OrderedDict(
                [
                    ("np", net_utils.create_decoder_branch_ext(out_ch=2, ksize=3)),
                    ("tp", net_utils.create_decoder_branch_ext(out_ch=num_types, ksize=3)),
                    ("hv", net_utils.create_decoder_branch_ext(out_ch=2, ksize=3))
                ]
            )
        self.decoder = nn.ModuleDict(decoder)
        self.upsample2x = UpSample2x()

    def forward(self, imgs):
        d0, d1, d2, d3 = self.backbone(imgs, self.freeze)
        if self.freeze:
            d3 = self.conv_bot(d3)
        else:
            d3 = checkpoint(self.conv_bot, d3)
        d = [d0, d1, d2, d3]

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
