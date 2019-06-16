import sys
import ipdb
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, '..')
from stn_utils.hyperparameter import logit, s_logit, s_sigmoid, inv_softplus, robustify, project, gaussian_cdf, HyperparameterInfo


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, use_device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    data = data.to(use_device)
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False, flatten_targets=True):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    if flatten_targets:
        target = Variable(source[i+1:i+1+seq_len].view(-1))
    else:
        target = Variable(source[i+1:i+1+seq_len])
    return data, target


def create_hparams(args, use_device):
    """
    Arguments:
        args: the arguments supplied by the user to the main script
        device: device we are training on

    Returns:
        htensor: unconstrained reparametrization of the starting hyperparameters
        hscale: unconstrained reparametrization of the perturbation distribution's scale
        hdict: dictionary mapping hyperparameter names to info about the hyperparameter
    """
    hdict = OrderedDict()
    htensor_list = []
    hscale_list = []

    drop_fcn = {'sigmoid': torch.sigmoid, 'none': lambda x: x}[args.drop_transform]
    alpha_beta_fcn = {'softplus': F.softplus, 'none': lambda x: x}[args.alpha_beta_transform]

    if args.tune_alpha:
        htensor_list.append(inv_softplus(torch.tensor(args.alpha)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['alpha'] = HyperparameterInfo(index=len(hdict), range=(0.,float('inf')), hnet_fcn=alpha_beta_fcn, minibatch=False)

    if args.tune_beta:
        htensor_list.append(inv_softplus(torch.tensor(args.beta)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['beta'] = HyperparameterInfo(index=len(hdict), range=(0.,float('inf')), hnet_fcn=alpha_beta_fcn, minibatch=False)

    if args.tune_dropouto:
        htensor_list.append(logit(torch.tensor(args.dropouto)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['dropouto'] = HyperparameterInfo(index=len(hdict), range=(0.,1.), hnet_fcn=drop_fcn)

    if args.tune_dropouti:
        htensor_list.append(logit(torch.tensor(args.dropouti)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['dropouti'] = HyperparameterInfo(index=len(hdict), range=(0.,1.), hnet_fcn=drop_fcn)

    if args.tune_dropouth:
        htensor_list.append(logit(torch.tensor(args.dropouth)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['dropouth'] = HyperparameterInfo(index=len(hdict), range=(0.,1.), hnet_fcn=drop_fcn)

    if args.tune_dropoute:
        htensor_list.append(logit(torch.tensor(args.dropoute)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['dropoute'] = HyperparameterInfo(index=len(hdict), range=(0.,1.), hnet_fcn=drop_fcn, minibatch=False)

    if args.tune_wdrop:
        htensor_list.append(logit(torch.tensor(args.wdrop)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['wdrop'] = HyperparameterInfo(index=len(hdict), range=(0.,1.), hnet_fcn=drop_fcn, minibatch=False)

    htensor = nn.Parameter(torch.stack(htensor_list).to(use_device))
    hscale = nn.Parameter(torch.stack(hscale_list).to(use_device))
    return htensor, hscale, hdict
