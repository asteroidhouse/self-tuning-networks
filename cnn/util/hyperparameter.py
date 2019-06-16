import sys
sys.path.insert(0, '..')
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from stn_utils.hyperparameter import logit, inv_softplus, s_sigmoid, s_logit, HyperparameterInfo


###############################################################################
# Miscellaneous functions
###############################################################################

def create_hparams(args, cnn_class, device):
    """
    Arguments:
        args: the arguments supplied by the user to the main script
        cnn_class: the convolutional net class
        device: device we are training on

    Returns:
        htensor: unconstrained reparametrization of the starting hyperparameters
        hscale: unconstrained reparametrization of the perturbation distribution's scale
        hdict: dictionary mapping hyperparameter names to info about the hyperparameter
    """
    hdict = OrderedDict()
    htensor_list = []
    hscale_list = []

    drop_max = 0.9 # improves stability
    if args.tune_dropout:
        htensor_list.append(s_logit(torch.tensor(args.start_drop), min=0., max=drop_max))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['dropout'] = HyperparameterInfo(index=len(hdict), range=(0.,drop_max), hnet_fcn=lambda x: x)

    if args.tune_dropoutl:
        for i in range(cnn_class.num_drops):
            htensor_list.append(s_logit(torch.tensor(args.start_drop), min=0., max=drop_max))
            hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
            hdict['dropout' + str(i)] = HyperparameterInfo(index=len(hdict), range=(0.,drop_max), hnet_fcn=lambda x: x)

    if args.tune_hue or args.tune_jitters:
        htensor_list.append(logit(torch.tensor(2*args.start_hue)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['hue'] = HyperparameterInfo(index=len(hdict), range=(0.,0.5), hnet_fcn=lambda x: x)

    if args.tune_contrast or args.tune_jitters:
        htensor_list.append(logit(torch.tensor(args.start_contrast)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['contrast'] = HyperparameterInfo(index=len(hdict), range=(0.,1.), hnet_fcn=lambda x: x)

    if args.tune_sat or args.tune_jitters:
        htensor_list.append(logit(torch.tensor(args.start_sat)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['sat'] = HyperparameterInfo(index=len(hdict), range=(0.,1.), hnet_fcn=lambda x: x)

    if args.tune_bright or args.tune_jitters:
        htensor_list.append(logit(torch.tensor(args.start_bright)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['bright'] = HyperparameterInfo(index=len(hdict), range=(0.,1.), hnet_fcn=lambda x: x)

    if args.tune_indropout:
        htensor_list.append(logit(torch.tensor(args.start_indrop)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['indropout'] = HyperparameterInfo(index=len(hdict), range=(0.,1.), hnet_fcn=lambda x: x)

    if args.tune_inscale:
        htensor_list.append(s_logit(torch.tensor(args.start_inscale), min=0, max=0.3))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['inscale'] = HyperparameterInfo(index=len(hdict),range=(0.,0.3), hnet_fcn=lambda x: x)

    if args.tune_cutlength:
        # Search over patch length of {0, 1, ..., 24}
        htensor_list.append(s_logit(torch.tensor(args.start_cutlength), min=0., max=25.))
        hscale_list.append(inv_softplus(torch.tensor(args.cutlength_scale)))
        hdict['cutlength'] = HyperparameterInfo(index=len(hdict), discrete=True, range=(0.,24.), hnet_fcn=lambda x: x)

    if args.tune_cutholes:
        htensor_list.append(s_logit(torch.tensor(args.start_cutholes), min=0., max=4.))
        hscale_list.append(inv_softplus(torch.tensor(args.cutholes_scale)))
        hdict['cutholes'] = HyperparameterInfo(index=len(hdict), discrete=True, range=(0.,4.), hnet_fcn=lambda x: x)

    if args.tune_fcdropout:
        for i in range(cnn_class.num_fcdrops):
            htensor_list.append(s_logit(torch.tensor(args.start_fcdrop), min=0., max=drop_max))
            hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
            hdict['fcdropout' + str(i)] = HyperparameterInfo(index=len(hdict), range=(0.,drop_max), hnet_fcn=lambda x: x)

    htensor = nn.Parameter(torch.stack(htensor_list).to(device))
    hscale = nn.Parameter(torch.stack(hscale_list).to(device))
    return htensor, hscale, hdict
