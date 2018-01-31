import argparse
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Caluculate Fisher')
parser.add_argument('--net', type=str, help='net path', required=True)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = torch.load(args.net)
net = net.cpu()

if use_cuda:
    net.cuda()
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

net.train()
sum_flat_fisher = 0
for batch_idx, (inputs, targets) in enumerate(trainloader):
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs), Variable(targets)
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    grads = torch.autograd.grad(loss, net.parameters())
    grads = [g.contiguous() for g in grads]
    flat_grad = torch.nn.utils.parameters_to_vector(grads)
    sum_flat_fisher += flat_grad.data**2

mean_flat_fisher = sum_flat_fisher / (batch_idx + 1)

torch.save(mean_flat_fisher, 'flat_fisher.pt')


