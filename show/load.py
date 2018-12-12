# -*- coding: utf-8 -*-

import torch
import torch.backends.cudnn as cudnn

import sys
sys.path.append("..")

from models import *


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module # that I actually define.
    def forward(self, x):
        return self.module(x)


def loadnet(index):
    if index == 1:
        print('ckpt1: 用shufflenet v2训练的, 20个epoch, 80%, 主要用来试验一下代码之类的')
        net = ShuffleNetV2(1)
        fname = '../saved_ckpt/ckpt1'
    elif index == 2:
        print('ckpt2: 用shufflenet v2训练了两百个epoch, 80%, 貌似极限就是这样了')
        net = ShuffleNetV2(1)
        fname = '../saved_ckpt/ckpt2'
    elif index == 3:
        print('ckpt3: 用DenseNet训练到三十个epoch左右开始卡住了, 87%')
        net = DenseNet121()
        fname = '../saved_ckpt/ckpt3'
    elif index == 4:
        print('ckpt4: DenseNet121, 88%')
        net = DenseNet121()
        fname = '../saved_ckpt/ckpt4'
    else:
        print('Invalid index')
        return

    if torch.cuda.is_available() == 'cuda':
        checkpoint = torch.load(fname)
        net = torch.nn.DataParallel(net)
        net.load_state_dict(checkpoint['net'])
        cudnn.benchmark = True
    else:
        checkpoint = torch.load(fname, map_location='cpu')
        net = WrappedModel(net)
        net.load_state_dict(checkpoint['net'])
    acc = checkpoint['acc']

    return [net, acc]


if __name__ == '__main__':
    [net, acc] = loadnet(1)
    print ("Accuracy: %f" % acc)
    print (net)
