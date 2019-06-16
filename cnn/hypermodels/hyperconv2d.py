import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, num_hparams,
        stride=1, bias=True):
        super(HyperConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_hparams = num_hparams
        self.stride = stride

        self.elem_weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size))
        self.hnet_weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.elem_bias = nn.Parameter(torch.Tensor(out_channels))
            self.hnet_bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('elem_bias', None)
            self.register_parameter('hnet_bias', None)

        self.htensor_to_scalars = nn.Linear(
            self.num_hparams, self.out_channels*2, bias=False)
        self.elem_scalar = nn.Parameter(torch.ones(1))

        self.init_params()

    def forward(self, input, htensor):
        """
        Arguments:
            input (tensor): size should be (B, C, H, W)
            htensor (tensor): size should be (B, D)
        """
        output = F.conv2d(input, self.elem_weight, self.elem_bias, padding=self.padding, 
            stride=self.stride)
        output *= self.elem_scalar
        if htensor is not None:
            hnet_scalars = self.htensor_to_scalars(htensor)
            hnet_wscalars = hnet_scalars[:, :self.out_channels].unsqueeze(2).unsqueeze(2)
            hnet_bscalars = hnet_scalars[:, self.out_channels:]

            hnet_out = F.conv2d(input, self.hnet_weight, padding=self.padding, 
                stride=self.stride)
            hnet_out *= hnet_wscalars
            if self.hnet_bias is not None:
                hnet_out += (hnet_bscalars * self.hnet_bias).unsqueeze(2).unsqueeze(2)
            output += hnet_out

        return output

    def init_params(self):
        # Initialize elementary parameters.
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.elem_weight.data.uniform_(-stdv, stdv)
        self.hnet_weight.data.uniform_(-stdv, stdv)
        if self.elem_bias is not None:
            self.elem_bias.data.uniform_(-stdv, stdv)
            self.hnet_bias.data.uniform_(-stdv, stdv)

        # Intialize hypernet parameters.
        self.htensor_to_scalars.weight.data.normal_(std=0.01)