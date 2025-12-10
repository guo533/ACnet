import torch
import torch.nn.functional as F
import sl.features as features


####
def xentropy_loss(true, pred, reduction='mean', epsilon=1e-7):
    """Cross entropy loss. Assumes NHWC!

    Args:
        true: ground truth array
        pred: prediction array

    Returns:
        cross entropy loss
    """
    # scale preds so that the class probs of each sample sum to 1
    pred = pred / torch.sum(pred, -1, keepdim=True)
    # manual computation of crossentropy
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
    loss = loss.mean() if reduction == "mean" else loss.sum()
    return loss


####
def dice_loss(true, pred, smooth=1e-3):
    """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
    inse = torch.sum(pred * true, (0, 1, 2))
    l = torch.sum(pred, (0, 1, 2))
    r = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss


####
def mse_loss(true, pred):
    """Calculate mean squared error loss.

    Args:
        true: ground truth of combined horizontal
              and vertical maps
        pred: prediction of combined horizontal
              and vertical maps

    Returns:
        loss: mean squared error

    """
    loss = pred - true
    loss = (loss * loss).mean()
    return loss


def simsiam_loss(p1, z2, p2, z1):
    def _D(p, z):
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()
    return _D(p1, z2) / 2 + _D(p2, z1) / 2


def cross_domain_class_level_regularization_loss(true_decoder_outputs, true_type_maps, pred_decoder_outputs, pred_type_maps):
    """Calculate cross domain class level regularization loss
    Parameters
    ----------
    true_decoder_outputs (torch.Tensor): its dim is [B, H, W, C] and it is the feature maps extracted from labeled images
    true_type_maps (torch.Tensor): its dim is [B, H, W, num_classes] and it is the type maps of labeled images
    pred_decoder_outputs (torch.Tensor): its dim is [B, H, W, C] and it is the feature maps extracted from unlabeled images
    pred_type_maps (torch.Tensor): its dim is [B, H, W, num_classes] and it is the type maps of unlabeled images

    Returns
    -------
    loss: cross domain class level regularization loss
    """
    true_prototypes = features.extract_prototypes(true_decoder_outputs, true_type_maps)
    pred_prototypes = features.extract_prototypes(pred_decoder_outputs, pred_type_maps)
    count = true_prototypes.shape[0] if true_prototypes.shape[0] < pred_prototypes.shape[0] else pred_prototypes.shape[0]
    loss = mse_loss(true_prototypes[:count], pred_prototypes[:count])
    return loss
