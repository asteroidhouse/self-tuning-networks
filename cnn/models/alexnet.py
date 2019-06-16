import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):

    def __init__(self, num_classes, dropRates, fc_dropRates, filters):
        super(AlexNet, self).__init__()

        self.dropRates = dropRates
        self.fc_dropRates = fc_dropRates
        
        channels = [3] + filters
        strides = [2, 1, 1, 1, 1]

        self.convs = []
        for in_, out, stride in zip(channels[:-1], channels[1:], strides):
            self.convs.append(nn.Conv2d(in_, out, stride=stride, kernel_size=3, padding=1))
        self.convs = nn.ModuleList(self.convs)

        self.fc1 = nn.Linear(filters[-1]*2*2, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x))
            if i in [0, 1, 4]:
                x = F.max_pool2d(x, kernel_size=2)
            x = F.dropout2d(x, p=self.dropRates[i], training=self.training)

        x = x.view(x.size(0), -1)

        x = F.dropout(x, p=self.fc_dropRates[0], training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.fc_dropRates[1], training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x