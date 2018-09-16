'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import pickle
import numpy as np
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
def train(args,
          net,
          trainloader,
          device,
          optimizer,
          epoch,
          total_num_images_backpropped,
          images_hist):

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    losses_pool = []
    data_pool = []
    targets_pool = []
    ids_pool = []
    num_backprop = 0
    loss_reduction = None

    for batch_idx, (data, targets, image_id) in enumerate(trainloader):

        data, targets = data.to(device), targets.to(device)

        if args.selective_backprop:

            output = net(data)
            loss = nn.CrossEntropyLoss(reduce=True)(output, targets)
            losses_pool.append(loss.item())
            data_pool.append(data)
            targets_pool.append(targets)
            ids_pool.append(image_id.item())

            if len(losses_pool) == args.pool_size:
            # Choose frames from pool to backprop
                indices = np.array(losses_pool).argsort()[-args.top_k:]
                chosen_data = [data_pool[i] for i in indices]
                chosen_targets = [targets_pool[i] for i in indices]
                chosen_ids = [ids_pool[i] for i in indices]

                data_batch = torch.stack(chosen_data, dim=1)[0]
                targets_batch = torch.cat(chosen_targets)
                output_batch = net(data_batch) # redundant

                for chosen_id in chosen_ids:
                    if chosen_id not in images_hist.keys():
                        images_hist[chosen_id] = 0
                    images_hist[chosen_id] += 1

                # Note: This will only work for batch size of 1
                loss_reduction = nn.CrossEntropyLoss(reduce=True)(output_batch, targets_batch)
                optimizer.zero_grad()
                loss_reduction.backward()
                optimizer.step()
                train_loss += loss_reduction.item()
                num_backprop += args.top_k

                losses_pool = []
                data_pool = []
                targets_pool = []
                ids_pool = []

                output = output_batch
                targets = targets_batch

        else:
            output = net(data)
            loss_reduction = nn.CrossEntropyLoss(reduce=True)(output, targets)
            optimizer.zero_grad()
            loss_reduction.backward()
            optimizer.step()
            train_loss += loss_reduction.item()
            num_backprop += args.batch_size

        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0 and loss_reduction is not None:
            print('train_debug,{},{},{:.6f},{:.6f},{},{:.6f}'.format(
                        epoch,
                        total_num_images_backpropped + num_backprop,
                        loss_reduction.item(),
                        train_loss / float(num_backprop),
                        time.time(),
                        100.*correct/total))
    return num_backprop

def test(args, net, testloader, device, epoch, total_num_images_backpropped):
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
    print('test_debug,{},{},{:.6f},{:.6f},{}'.format(
                epoch,
                total_num_images_backpropped,
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
    parser.add_argument('--decay', default=5e-4, type=float, help='decay')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--top-k', type=int, default=8, metavar='N',
                        help='how many images to backprop per batch')
    parser.add_argument('--pool-size', type=int, default=16, metavar='N',
                        help='how many images to backprop per batch')
    parser.add_argument('--selective-backprop', type=bool, default=False, metavar='N',
                        help='whether or not to use selective-backprop')
    parser.add_argument('--net', default="resnet", metavar='N',
                        help='which network architecture to train')
    parser.add_argument('--pickle-file', default="/tmp/image_id_hist.pickle",
                        help='image id histogram pickle')

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
    trainset = [t + (i,) for i, t in enumerate(trainset)]
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

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

    images_hist = {}
    total_num_images_backpropped = 0
    for epoch in range(start_epoch, start_epoch+500):
        test(args, net, testloader, device, epoch, total_num_images_backpropped)
        num_images_backpropped = train(args,
                                       net,
                                       trainloader,
                                       device,
                                       optimizer,
                                       epoch,
                                       total_num_images_backpropped,
                                       images_hist)
        total_num_images_backpropped += num_images_backpropped
        with open(args.pickle_file, "wb") as handle:
            pickle.dump(images_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
