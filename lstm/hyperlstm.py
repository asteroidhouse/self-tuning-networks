import ipdb
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from weight_drop import WeightDrop
from locked_dropout import LockedDropout
from embed_regularize import embedded_dropout

import utils


class HyperLSTM(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayers, dropouto=0, dropouth=0, dropouti=0, dropoute=0, wdrop=0,
                 tie_weights=True, num_hparams=1, device='cuda:0'):
        super(HyperLSTM, self).__init__()
        self.lockdrop = LockedDropout()

        self.encoder = HyperEmbedding(ntoken, ninp, num_hparams)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            print("Tie weights")
            self.decoder.weight = self.encoder.elem_embedding.weight

        self.rnns = [HyperLSTMCell(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                                   num_hparams=num_hparams, wdrop=wdrop,) for l in range(nlayers)]
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)

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

    def forward(self, input, hidden, hnet_tensor, hparam_tensor, hdict, return_h=False):
        if 'wdrop' in hdict:
            wdrop = hparam_tensor[:, hdict['wdrop'].index][0]
        else:
            wdrop = self.wdrop

        if 'dropoute' in hdict:
            embed_dropout = hparam_tensor[:, hdict['dropoute'].index][0]
        else:
            embed_dropout = self.dropoute

        if 'dropouti' in hdict:
            input_dropout = hparam_tensor[:, hdict['dropouti'].index].unsqueeze(1)
        else:
            input_dropout = self.dropouti

        if 'dropouto' in hdict:
            output_dropout = hparam_tensor[:, hdict['dropouto'].index].unsqueeze(1)
        else:
            output_dropout = self.dropouto

        if 'dropouth' in hdict:
            hidden_dropout = hparam_tensor[:, hdict['dropouth'].index].unsqueeze(1)
        else:
            hidden_dropout = self.dropouth


        emb = self.encoder(input, hnet_tensor, dropoute=embed_dropout if self.training else 0)
        emb = self.lockdrop(emb, input_dropout)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []

        for l, rnn in enumerate(self.rnns):
            current_input = raw_output

            rnn.h2h.mask_weights(wdrop)
            rnn.i2h.mask_weights(0)

            raw_output, new_h = rnn(raw_output, hidden[l], hnet_tensor)
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, hidden_dropout)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, output_dropout)
        outputs.append(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))

        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                 weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                 for l in range(self.nlayers)]


class HyperEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_size, num_hparams):
        super(HyperEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.num_hparams = num_hparams

        self.elem_embedding = nn.Embedding(num_embeddings, embedding_size)
        self.hnet_embedding = nn.Embedding(num_embeddings, embedding_size)

        self.hnet_tensor_to_scalars = nn.Linear(self.num_hparams, embedding_size, bias=False)
        self.init_params()

    def forward(self, input, hnet_tensor, dropoute=0):
        if dropoute:
            mask = self.elem_embedding.weight.data.new().resize_((self.elem_embedding.weight.size(0), 1)).bernoulli_(1 - dropoute).detach().expand_as(self.elem_embedding.weight) / (1 - dropoute)
            masked_elem_embed_weight = mask * self.elem_embedding.weight
            masked_hnet_embed_weight = mask * self.hnet_embedding.weight
        else:
            masked_elem_embed_weight = self.elem_embedding.weight
            masked_hnet_embed_weight = self.hnet_embedding.weight

        padding_idx = self.elem_embedding.padding_idx
        if padding_idx is None:
            padding_idx = -1

        output = F.embedding(input, masked_elem_embed_weight,
                             padding_idx, self.elem_embedding.max_norm, self.elem_embedding.norm_type,
                             self.elem_embedding.scale_grad_by_freq, self.elem_embedding.sparse)

        if hnet_tensor is not None:
            hnet_scalars = self.hnet_tensor_to_scalars(hnet_tensor)
            padding_idx = self.hnet_embedding.padding_idx
            if padding_idx is None:
                padding_idx = -1

            hnet_out = F.embedding(input, masked_hnet_embed_weight,
                                   padding_idx, self.hnet_embedding.max_norm, self.hnet_embedding.norm_type,
                                   self.hnet_embedding.scale_grad_by_freq, self.hnet_embedding.sparse)

            hnet_out *= hnet_scalars
            output += hnet_out

        return output

    def init_params(self):
        initrange = 0.1
        self.elem_embedding.weight.data.uniform_(-initrange, initrange)
        self.hnet_embedding.weight.data.uniform_(-initrange, initrange)
        self.hnet_tensor_to_scalars.weight.data.normal_(std=0.01)  # Intialize hypernet parameters.


class HyperLinear(nn.Module):
    def __init__(self, input_dim, output_dim, num_hparams, bias=True):
        super(HyperLinear, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hparams = num_hparams
        self.num_scalars = output_dim
        self.bias = bias

        # "raw" because we allow for weight dropping to set the "actual" self.elem_weight
        self.elem_weight_raw = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.hnet_weight_raw = nn.Parameter(torch.Tensor(output_dim, input_dim))

        if bias:
            self.elem_bias = nn.Parameter(torch.Tensor(output_dim))
            self.hnet_bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('elem_bias', None)
            self.register_parameter('hnet_bias', None)

        self.hnet_tensor_to_scalars = nn.Linear(self.num_hparams, self.num_scalars*2, bias=False)
        self.init_params()

    def mask_weights(self, wdrop):
        if self.training:
            m = self.elem_weight_raw.data.new(self.elem_weight_raw.size()).bernoulli_(1 - wdrop)
            mask = m.detach() / (1 - wdrop)
        else:
            mask = self.elem_weight_raw.data.new(self.elem_weight_raw.size()).fill_(1)  # All 1's (nothing dropped) at test-time

        self.elem_weight = self.elem_weight_raw * mask
        self.hnet_weight = self.hnet_weight_raw * mask

    def forward(self, input, hnet_tensor):
        output = F.linear(input, self.elem_weight, self.elem_bias)

        if hnet_tensor is not None:
            hnet_scalars = self.hnet_tensor_to_scalars(hnet_tensor)
            hnet_wscalars = hnet_scalars[:, :self.num_scalars]
            hnet_bscalars = hnet_scalars[:, self.num_scalars:]
            hnet_out = hnet_wscalars * F.linear(input, self.hnet_weight)

            if self.hnet_bias is not None:
                hnet_out += hnet_bscalars * self.hnet_bias

            output += hnet_out

        return output

    def init_params(self):
        # Initialize elementary parameters.
        n = self.input_dim
        stdv = 1. / math.sqrt(n)
        self.elem_weight_raw.data.uniform_(-stdv, stdv)
        self.hnet_weight_raw.data.uniform_(-stdv, stdv)
        if self.elem_bias is not None:
            self.elem_bias.data.uniform_(-stdv, stdv)
            self.hnet_bias.data.uniform_(-stdv, stdv)

        # Intialize hypernet parameters.
        self.hnet_tensor_to_scalars.weight.data.normal_(std=0.01)


class HyperLSTMCell(nn.Module):
    """Cell for dropconnect RNN."""

    def __init__(self, ninp, nhid, num_hparams, wdrop=0):
        super(HyperLSTMCell, self).__init__()

        self.ninp = ninp
        self.nhid = nhid
        self.wdrop = wdrop
        self.num_hparams = num_hparams

        self.i2h = HyperLinear(ninp, 4*nhid, num_hparams)
        self.h2h = HyperLinear(nhid, 4*nhid, num_hparams)

    def forward(self, input, hidden, hparams):

        hidden_list = []
        nhid = self.nhid
        h, cell = hidden

        h = h.squeeze(0)  # Only squeeze 0th dim. From (1, 1, 650) to (1, 650)
        cell = cell.squeeze(0)

        # Loop over the indexes in the sequence --> process each index in parallel across items in the batch
        for i in range(len(input)):
            x = input[i]

            x_components = self.i2h(x, hparams)
            h_components = self.h2h(h, hparams)

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
