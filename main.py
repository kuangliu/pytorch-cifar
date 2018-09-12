'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


# Training
def train(args, net, trainloader, device, optimizer, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    all_losses = []
    loss_reduction = None

    num_backprop = 0

    for batch_idx, (data, targets) in enumerate(trainloader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = net(data)

        if args.selective_backprop:
            loss = nn.CrossEntropyLoss(reduce=False)(output, targets)
            k = min(len(loss), args.top_k)
            values, indices = torch.topk(loss, k)

            if len(all_losses) == len(loss):
                all_losses_tensor = torch.cat(all_losses)
                loss_reduction = torch.mean(all_losses_tensor)
                loss_reduction.backward()
                optimizer.step()

                train_loss += loss_reduction.item()
                num_backprop += 1
                all_losses = []

            else:
                all_losses.append(values)
        else:
            loss_reduction = nn.CrossEntropyLoss(reduce=True)(output, targets)
            loss_reduction.backward()
            optimizer.step()
            train_loss += loss_reduction.item()
            num_backprop += 1

        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0 and loss_reduction is not None:
            print('train_debug,{},{},{:.6f},{:.6f},{},{:.6f}'.format(
                        epoch,
                        num_backprop,
                        loss_reduction.item(),
                        train_loss / float(num_backprop),
                        time.time(),
                        100.*correct/total))

def test(args, net, testloader, device, epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = F.nll_loss(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(testloader.dataset)
    print('test_debug,{},{:.6f},{:.6f},{}'.format(
                epoch,
                test_loss,
                100.*correct/total,
                time.time()))

    # Save checkpoint.
    '''
    global best_acc
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    '''


def main():

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--top-k', type=int, default=1, metavar='N',
                        help='how many images to backprop per batch')
    parser.add_argument('--selective-backprop', type=bool, default=False, metavar='N',
                        help='whether or not to use selective-backprop')
    parser.add_argument('--net', default="resnet", metavar='N',
                        help='which network architecture to train')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    if args.net == "resnet":
        net = ResNet18()
    elif args.net == "vgg":
        net = VGG('VGG19')
    elif args.net == "preact_resnet":
        net = PreActResNet18()
    elif args.net == "googlenet":
        net = GoogLeNet()
    elif args.net == "densenet":
        net = DenseNet121()
    elif args.net == "resnext":
        net = ResNeXt29_2x64d()
    elif args.net == "mobilenet":
        net = MobileNet()
    elif args.net == "mobilenet_v2":
        net = MobileNetV2()
    elif args.net == "dpn":
        net = DPN92()
    elif args.net == "shufflenet":
        net = ShuffleNetG2()
    elif args.net == "senet":
        net = SENet18()
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, start_epoch+500):
        train(args, net, trainloader, device, optimizer, epoch)
        test(args, net, testloader, device, epoch)

if __name__ == '__main__':
    main()
