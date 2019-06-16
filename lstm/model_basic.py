import sys
import ipdb

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from weight_drop import WeightDrop
from locked_dropout import LockedDropout
from embed_regularize import embedded_dropout


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropouto=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type == 'LSTM':
            self.rnns = [DropconnectCell(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), wdrop=wdrop) for l in range(nlayers)]
            if wdrop:
                print("Using weight drop {}".format(wdrop))

        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights
        if tie_weights:
            print("Tie weights")
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.ntoken = ntoken
        self.nlayers = nlayers
        self.dropouto = dropouto
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.wdrop = wdrop
        self.tie_weights = tie_weights

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output

            rnn.h2h.mask_weights(self.wdrop)
            rnn.i2h.mask_weights(0)

            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropouto)
        outputs.append(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()),
                    Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()))
                    for l in range(self.nlayers)]


class DropconnectLinear(nn.Module):
    def __init__(self, input_dim, output_dim, wdrop=0):
        super(DropconnectLinear, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.wdrop = wdrop

        self.elem_weight_raw = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.elem_bias = nn.Parameter(torch.Tensor(output_dim))

        self.init_params()

    def mask_weights(self, wdrop):
        if self.training:
            m = self.elem_weight_raw.data.new(self.elem_weight_raw.size()).bernoulli_(1 - wdrop)
            mask = Variable(m, requires_grad=False) / (1 - wdrop)
        else:
            m = self.elem_weight_raw.data.new(self.elem_weight_raw.size()).fill_(1)  # All 1's (nothing dropped) at test-time
            mask = Variable(m, requires_grad=False)

        self.elem_weight = self.elem_weight_raw * mask

    def forward(self, input):
        return F.linear(input, self.elem_weight, self.elem_bias)

    def init_params(self):
        # Initialize elementary parameters.
        n = self.input_dim
        stdv = 1. / math.sqrt(n)
        self.elem_weight_raw.data.uniform_(-stdv, stdv)
        self.elem_bias.data.uniform_(-stdv, stdv)


class DropconnectCell(nn.Module):
    """Cell for dropconnect RNN."""

    def __init__(self, ninp, nhid, wdrop=0):
        super(DropconnectCell, self).__init__()

        self.ninp = ninp
        self.nhid = nhid
        self.wdrop = wdrop

        self.i2h = DropconnectLinear(ninp, 4*nhid, wdrop=0)
        self.h2h = DropconnectLinear(nhid, 4*nhid, wdrop=wdrop)

    def forward(self, input, hidden):

        hidden_list = []

        nhid = self.nhid

        h, cell = hidden

        # Loop over the indexes in the sequence --> process each index in parallel across items in the batch
        for i in range(len(input)):

            h = h.squeeze()
            cell = cell.squeeze()

            x = input[i]

            x_components = self.i2h(x)
            h_components = self.h2h(h)

            preactivations = x_components + h_components

            gates_together = torch.sigmoid(preactivations[:, 0:3*nhid])
            forget_gate = gates_together[:, 0:nhid]
            input_gate = gates_together[:, nhid:2*nhid]
            output_gate = gates_together[:, 2*nhid:3*nhid]
            new_cell = torch.tanh(preactivations[:, 3*nhid:4*nhid])

            cell = forget_gate * cell + input_gate * new_cell
            h = output_gate * torch.tanh(cell)

            hidden_list.append(h)

        hidden_stacked = torch.stack(hidden_list)

        return hidden_stacked, (h.unsqueeze(0), cell.unsqueeze(0))
