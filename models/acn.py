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
        
    def forward(self, x):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x))
        #print (x.size())
        x = F.elu(self.conv2(x))
        #print (x.size())
        x = F.elu(self.conv3(x))
        x = F.dropout(x, p=0.5, training=self.training)
        #print (x.size())
        x = F.elu(self.conv4(x))
        #print (x.size())
        x = F.elu(self.conv5(x))
        #print (x.size())
        x = F.elu(self.conv6(x))
        x = F.dropout(x, p=0.5, training=self.training)
        #print (x.size())
        x = F.elu(self.conv7(x))
        #print (x.size())
        x = F.elu(self.conv8(x))
        #print (x.size())
        x = F.elu(self.conv9(x))
        #print (x.size())
        x = torch.squeeze(torch.mean(torch.mean(x, 2), 3))
        #print (x.size())
        #print (x[0])
        return x
