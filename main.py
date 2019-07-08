'''Single node, multi-GPUs training.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import argparse
import torchvision
import torchvision.transforms as transforms

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Distributed Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--local_rank', default=0, type=int, help='current process rank')
parser.add_argument('--world_size', default=1, type=int, help='total number of ranks')
args = parser.parse_args()


# Init process group
print("Initialize Process Group...")
dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456',
                        rank=args.local_rank, world_size=args.world_size)
torch.cuda.set_device(args.local_rank)

# Init Model
print("Initialize Model...")
net = EfficientNetB0().cuda()
net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
net = torch.nn.parallel.DistributedDataParallel(
    net, device_ids=[args.local_rank], output_device=args.local_rank)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), 1e-3, momentum=0.9, weight_decay=1e-4)

# Data
print('==> Preparing data..')
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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=(trainsampler is None), num_workers=2, pin_memory=True, sampler=trainsampler)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=False)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


for epoch in range(200):
    trainsampler.set_epoch(epoch)
    train(epoch)
    test(epoch)
