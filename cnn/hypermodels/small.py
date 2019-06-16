import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperconv2d import HyperConv2d
from util.dropout import dropout_2d

class SmallCNN(nn.Module):
    num_drops = 3
    num_layers = 3

    def __init__(self, args, num_classes, num_hparams):
        super(SmallCNN, self).__init__()
        self.dropout = args.start_drop
        self.num_hparams = num_hparams
        self.filters = [3, 10, 20, 40]

        # Intialize convolutional filters and last fully connected layer. 
        self.convs = nn.ModuleList([
            HyperConv2d(self.filters[i], self.filters[i+1], kernel_size=3, padding=1, 
                num_hparams=self.num_hparams) for i in range(3)])

        self.last_dim = self.filters[-1]*4*4
        self.fc = nn.Linear(self.last_dim, num_classes)

    def forward(self, x, hnet_tensor, hparam_tensor, hdict):
        for layer, hconv in enumerate(self.convs):
            x = hconv(x, hnet_tensor)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2)
            drop_probs = self.get_drop_probs(hparam_tensor, hdict, layer)
            x = dropout_2d(x, drop_probs, training=self.training)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_drop_probs(self, hparam_tensor, hdict, layer):
        if 'dropout' in hdict:
            drop_idx = hdict['dropout'].index
        elif 'dropout' + str(layer) in hdict:
            drop_idx = hdict['dropout' + str(layer)].index
        else:
            return 0.
        return hparam_tensor[:, drop_idx]