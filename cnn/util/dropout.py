import torch
import torch.nn.functional as F


def dropout_2d(x, probs, training=False):
    """
    Arguments:
        x (Tensor): size (B, C, H, W)
        probs (Tensor): size (B,)
    """
    if not training:
        return x
    if isinstance(probs, float):
        return F.dropout2d(x, probs, training)
    probs = probs.unsqueeze(1).repeat(1, x.size(1)).detach()
    mask = (1 - probs).bernoulli().div_(1 - probs)
    mask = mask.unsqueeze(2).unsqueeze(2)
    return x * mask


def dropout(x, probs, training=False):
    """
    Arguments:
        x (Tensor): whose first dimension has size B
        probs (Tensor): size (B,)
    """
    if not training:
        return x
    if isinstance(probs, float):
        return F.dropout(x, probs, training)
    x_size = x.size()
    x = x.view(x.size(0), -1)
    probs = probs.unsqueeze(1).repeat(1, x.size(1)).detach()
    mask = (1 - probs).bernoulli().div_(1 - probs)
    return (x * mask).view(x_size)
