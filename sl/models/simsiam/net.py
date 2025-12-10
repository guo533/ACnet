import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, in_dim=3, dim=1000, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        base_encoder = models.__dict__['resnet50']
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(pretrained=True, progress=True, num_classes=dim, zero_init_residual=True)
        self.encoder.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC
        #
        return p1, p2, z1.detach(), z2.detach()

def main():
    # net = SimSiam(in_dim=9)
    # net.train()
    #
    # x1 = torch.randn((8, 9, 256, 256))
    # x2 = torch.randn((8, 9, 256, 256))
    # p1, p2, z1, z2 = net(x1, x2)
    #
    # loss = simsiam_loss(p1, z2, p2, z1)
    pass


if __name__ == '__main__':
    main()
