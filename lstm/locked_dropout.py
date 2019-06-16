import ipdb

import torch
import torch.nn as nn


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout):
        if not self.training:
            return x

        if isinstance(dropout, torch.Tensor):
            m = x.data.new(1, x.size(1), x.size(2))
            expanded_dropout_probs = (1 - dropout.unsqueeze(0).expand_as(m)).detach()
            mask = expanded_dropout_probs.bernoulli() / expanded_dropout_probs
        else:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
            mask = m / (1 - dropout)

        mask = mask.expand_as(x)
        return mask * x
