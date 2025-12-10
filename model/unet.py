from collections import OrderedDict

import torch
import torch.nn as nn
import segmentation_models_pytorch.encoders as encoders
from model.model_utils import UNetDecoder


class UNet(nn.Module):

    def __init__(self, in_channels=3, num_types=6, returns_decoder_output=False):
        super(UNet, self).__init__()
        self.returns_decoder_output = returns_decoder_output
        self.encoder = encoders.get_encoder('resnet50', in_channels=in_channels, depth=5, weights='imagenet')
        self.decoder_channels = (1024, 512, 256, 128, 64)
        self.decoder = UNetDecoder(encoder_channels=self.encoder.out_channels,
                                   decoder_channels=self.decoder_channels,
                                   use_batchnorm=True,
                                   center=False, attention_type=None)
        self.out = nn.Sequential(nn.Conv2d(self.decoder_channels[-1], num_types, kernel_size=(3, 3), padding=1))

    def forward(self, x):
        features = self.encoder(x)
        decoder_output, _ = self.decoder(*features)
        pred_dict = OrderedDict()
        if self.returns_decoder_output:
            pred_dict['decoder_output'] = decoder_output
        pred_dict['type_map'] = self.out(decoder_output)
        return pred_dict
