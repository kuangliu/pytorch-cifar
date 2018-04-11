from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #print m.weight.data.size()
        m.weight.data.normal_(0.0, 0.02)
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or  classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _DCGAND(nn.Module):
    def __init__(self, ngpu=0, ndf=64, nc=3):
        super(_DCGAND, self).__init__()
        self.ngpu = ngpu
        self.layer1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pred = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            x = self.layer1(input)
            x = self.layer2(x)
            x = self.layer3(x)
            output = self.pred(x)
        return output.view(-1, 1).squeeze(1), x

class _DCGANConf(nn.Module):
    def __init__(self, ngpu=0, ndf=64, nc=3, num_classes=21):
        super(_DCGANConf, self).__init__()
        self.dcgan = _DCGAND(0, 64, 3)
        self.conf = nn.Linear(512*4*4, num_classes)

    def forward(self, input):
        _, y = self.dcgan(input)
        y = y.view(y.size(0), -1)
        conf = self.conf(y)
        return conf

def build_dcganconf(ngpu=0, ndf=64, nc=3, num_classes=21):
    net =  _DCGANConf(ngpu=ngpu, ndf=ndf, nc=nc, num_classes=num_classes)
    net.apply(weights_init)
    return net

    