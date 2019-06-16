import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):

    def __init__(self, num_classes, dropRates):
        super(SmallCNN, self).__init__()
        self.dropRates = dropRates
        channels = [3, 10, 20, 40, 60]
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(channels[i], channels[i+1], 3, padding=1) 
            for i in range(3)])
        self.last_dim = 40*4*4
        self.fc = nn.Linear(self.last_dim, num_classes)
     
    def forward(self, x):
        for i, layer in enumerate(self.conv_layers):
            x = F.relu(layer(x))
            x = F.dropout2d(x, p=self.dropRates[i], training=self.training)
            x = F.max_pool2d(x, 2)
        x = x.view(-1, self.last_dim)
        x = self.fc(x)
        return x