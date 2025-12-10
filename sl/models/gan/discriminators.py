import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import sl.models.gan.net_utils as net_utils
from sl.gradient_reversal import GradientReversal


class Discriminator(nn.Module):
    def __init__(self, num_types=6, channels=(64, 128, 256, 512)):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
                                nn.Conv2d(num_types, channels[0], kernel_size=4, stride=2, padding=1, bias=False),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = nn.Sequential(
                                  nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(channels[1]),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),

                                  nn.Conv2d(channels[1], channels[2], kernel_size=4, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(channels[2]),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),

                                  nn.Conv2d(channels[2], channels[3], kernel_size=4, stride=2, padding=1,
                                            bias=False),
                                  nn.BatchNorm2d(channels[3]),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(channels[3], 1, kernel_size=4, stride=1, padding=0,
                                            bias=False))
        # net_utils.weights_init(self)

    def forward(self, x):
        x = self.conv1(x)
        x = checkpoint(self.conv2, x)
        return x


class DiscriminatorUDA(nn.Module):
    def __init__(self, num_types=6, channels=(64, 128, 256, 512)):
        super(DiscriminatorUDA, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(num_types, channels[0], kernel_size=4, stride=2, padding=1, bias=False),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = nn.Sequential(
                                  nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(channels[1]),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),

                                  nn.Conv2d(channels[1], channels[2], kernel_size=4, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(channels[2]),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    GradientReversal(alpha=1.0),
                                  nn.Conv2d(channels[2], channels[3], kernel_size=4, stride=2, padding=1,
                                            bias=False),
                                  nn.BatchNorm2d(channels[3]),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(channels[3], 1, kernel_size=4, stride=1, padding=0,
                                            bias=False))
        # net_utils.weights_init(self)

    def forward(self, x):
        x = self.conv1(x)
        x = checkpoint(self.conv2, x)
        return x