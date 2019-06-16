import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperconv2d import HyperConv2d
from hyperlinear import HyperLinear
from util.dropout import dropout, dropout_2d

class AlexNet(nn.Module):
    num_drops = 5
    num_layers = 5
    num_fcdrops = 2

    def __init__(self, args, num_classes, num_hparams):
        super(AlexNet, self).__init__()
        self.dropout = args.start_drop
        self.num_hparams = num_hparams
        self.filters = [3, 64, 192, 384, 256, 256]

        strides = [2, 1, 1, 1, 1]
        self.convs = nn.ModuleList([HyperConv2d(self.filters[i], self.filters[i+1], 
            stride=strides[i], kernel_size=3, padding=1, num_hparams=num_hparams)
            for i in range(5)])

        self.last_dim = self.filters[-1]*2*2
        self.fc1 = HyperLinear(self.last_dim, 4096, num_hparams)
        self.fc2 = HyperLinear(4096, 4096, num_hparams)
        self.fc3 = HyperLinear(4096, num_classes, num_hparams)

    def forward(self, x, hnet_tensor, hparam_tensor, hdict):
        for layer, hconv in enumerate(self.convs):
            x = F.relu(hconv(x, hnet_tensor))
            if layer in [0, 1, 4]:
                x = F.max_pool2d(x, kernel_size=2)

            drop_probs = self.get_drop_probs(hparam_tensor, hdict, layer)
            x = dropout_2d(x, drop_probs, training=self.training) 

        # Set-up before running through fully-connected layers.
        x = x.view(x.size(0), -1)
        fc_probs = self.get_fcdrop_probs(hparam_tensor, hdict)
        x = dropout(x, fc_probs[0], training=self.training)    
        x = F.relu(self.fc1(x, hnet_tensor))
        x = dropout(x, fc_probs[1], training=self.training)
        x = F.relu(self.fc2(x, hnet_tensor))
        x = self.fc3(x, hnet_tensor)
        return x

    def get_drop_probs(self, hparam_tensor, hdict, layer):
        if 'dropout' in hdict:
            drop_idx = hdict['dropout'].index
        elif 'dropout' + str(layer) in hdict:
            drop_idx = hdict['dropout' + str(layer)].index
        else:
            return 0.
        return hparam_tensor[:, drop_idx]

    def get_fcdrop_probs(self, hparam_tensor, hdict):
        if 'fcdropout0' not in hdict:
            return (0., 0.)
        fcdrop0_idx = hdict['fcdropout0'].index
        fcdrop1_idx = hdict['fcdropout1'].index
        return (hparam_tensor[:,fcdrop0_idx], hparam_tensor[:, fcdrop1_idx])