import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperLinear(nn.Module):

    def __init__(self, in_features, out_features, num_hparams, bias=True):
        super(HyperLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_hparams = num_hparams

        self.elem_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.hnet_weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.elem_bias = nn.Parameter(torch.Tensor(out_features))
            self.hnet_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('elem_bias', None)
            self.register_parameter('hnet_bias', None)

        self.htensor_to_scalars = nn.Linear(num_hparams, 2*out_features, bias=False)
        self.elem_scalar = nn.Parameter(torch.ones(1))
        self.init_params()

    def forward(self, input, htensor):
        """
        Arguments:
            input (tensor): size should be (B, D)
            htensor (tensor): size should be (B, num_hparams)
        """

        output = F.linear(input, self.elem_weight, self.elem_bias)
        output *= self.elem_scalar

        if htensor is not None:
            hnet_scalars = self.htensor_to_scalars(htensor)
            hnet_wscalars = hnet_scalars[:, :self.out_features]
            hnet_bscalars = hnet_scalars[:, self.out_features:]
            hnet_out = hnet_wscalars * F.linear(input, self.hnet_weight)

            if self.hnet_bias is not None:
                hnet_out += hnet_bscalars * self.hnet_bias

            output += hnet_out

        return output

    def init_params(self):
        # Initialize elementary parameters.
        stdv = 1. / math.sqrt(self.in_features) 
        self.elem_weight.data.uniform_(-stdv, stdv)
        self.hnet_weight.data.uniform_(-stdv, stdv)
        if self.elem_bias is not None:
            self.elem_bias.data.uniform_(-stdv, stdv)
            self.hnet_bias.data.uniform_(-stdv, stdv)

        # Intialize hypernet parameters.
        self.htensor_to_scalars.weight.data.normal_(std=0.01)
