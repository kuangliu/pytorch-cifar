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

import time

# Training
# write data for the epoch to file
def train(epoch, file):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    log = ''
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        log = progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # write the log to a file
    file.write(log)

# write data for the epoch to file
def test(epoch, file):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    log = ''
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            log = progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # write the log to a file
        file.write(log)

    # Save checkpoint.
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
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


# model name, dataset, batchsize train, batchsize test, epoch size
with open('info.txt') as f:
    configs = []
    for line in f:
        configs.append(line.strip().split())

    for config in configs:
        modelname = config[0]
        dataset = config[1]
        batchsize_train = int(config[2])
        batchsize_test = int(config[3])
        epoch_size = int(config[4])

        parser = argparse.ArgumentParser(description='PyTorch ' + dataset + ' Training')
        parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
        parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
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

        if (dataset == 'CIFAR10'):
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize_train, shuffle=True, num_workers=2)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize_test, shuffle=False, num_workers=2)
        # similarly for other datasets as well


        # does this also change?
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # fill in the dictionary with all the model names
        models = {'EfficientNetB0': EfficientNetB0, 'MobileNet': MobileNet, 'MobileNetV2': MobileNetV2, 'ShuffleNetV2': ShuffleNetV2, 'DPN92': DPN92, 'SENet18': SENet18}
        # Model
        print('==> Building model..')
        # net = VGG('VGG19')
        # net = ResNet18()
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        # net = ShuffleNetV2(1)
        
        if (modelname == 'ShuffleNetV2'):
          net = models[modelname](1)
        else:
          net = models[modelname]()

        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            try:
                checkpoint = torch.load('./checkpoint/ckpt.pth')
                net.load_state_dict(checkpoint['net'])
                best_acc = checkpoint['acc']
                start_epoch = checkpoint['epoch']
            except:
                pass

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        # open file handle
        filename = modelname + '_' + dataset + '_' + str(epoch_size) + '.txt'
        file = open(filename, 'a')
        
        # measure time
        start_time = time.time()
        for epoch in range(start_epoch, start_epoch+epoch_size):
            train(epoch, file)
            test(epoch, file)
        file.write('\nTraining + Testing time = ' + str((time.time() - start_time)/60) + ' minutes')

        file.close()
