'''
All convolutional Network
https://arxiv.org/pdf/1412.6806.pdf
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ACN(nn.Module):
    def __init__(self, cifar100=False):
        super(ACN, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(192, 192, 3)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 =  nn.Conv2d(192, 100 if cifar100 else 10, 1)
    
    def __str__(self):
        return "ACN"

    def forward(self, x):
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.dropout(F.relu(self.conv3(x)), training=self.training)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.dropout(F.relu(self.conv6(x)), training=self.training)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = torch.squeeze(torch.mean(torch.mean(x, 2), 3))
        return x
