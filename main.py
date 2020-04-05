import torch
import torch.distributed as dist
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

from torch.multiprocessing import Process

from random import Random

import time

modelname = ''
dataset = ''
batchsize_train = ''
batchsize_test = ''
epoch_size = ''

best_acc = 0

""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(batch_size):
    
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    size = dist.get_world_size()
    bsz = batch_size / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(trainset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=batch_size,
                                         shuffle=True)
    return train_set, bsz


""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

# Training
# write data for the epoch to file
def train(epoch, file, dataset, device, model, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    log = ''
    for batch_idx, (inputs, targets) in enumerate(dataset):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        #average_gradients(model)
        
        log = progress_bar(batch_idx, len(dataset), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # write the log to a file
    file.write(log)


# write data for the epoch to file
def test(epoch, file, dataset, device, model, criterion):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    log = ''
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataset):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #average_gradients(model)
            log = progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # write the log to a file
        file.write(log)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


""" Distributed Synchronous SGD Example """
def run(rank, size):
    parser = argparse.ArgumentParser(description='PyTorch ' + dataset + ' Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_set, bsz = partition_dataset(batchsize_train)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize_train, shuffle=False, num_workers=2)
    
    # does this also change?
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # fill in the dictionary with all the model names
    models = {'EfficientNetB0': EfficientNetB0, 'MobileNet': MobileNet, 'MobileNetV2': MobileNetV2, 'ShuffleNetV2': ShuffleNetV2, 'DPN92': DPN92, 'SENet18': SENet18, 'ResNeXt29_2x64d': ResNeXt29_2x64d}
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
    
    # get the number of gpus
    print('Number of GPUs:' + str(torch.cuda.device_count()))
    
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
    filename = modelname + '_' + dataset + '_' + str(epoch_size) + '_' + str(rank) + '.txt'
    file = open(filename, 'a')
    
    # measure time
    start_time = time.time()
    for epoch in range(start_epoch, start_epoch+epoch_size):
        train(epoch, file, train_set, device, net, optimizer, criterion)
        test(epoch, file, testloader, device, net, criterion)
    file.write('\nTraining + Testing time = ' + str((time.time() - start_time)/60) + ' minutes')

    file.close()


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    if rank == 0:
      os.environ['MASTER_ADDR'] = '10.125.100.2'
      os.environ['MASTER_PORT'] = '29500'
    else:
      os.environ['MASTER_ADDR'] = '10.125.100.5'
      os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


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
        models = {
          'EfficientNetB0': EfficientNetB0, 
          'MobileNet': MobileNet, 
          'MobileNetV2': MobileNetV2, 
          'ShuffleNetV2': ShuffleNetV2, 
          'DPN92': DPN92, 
          'SENet18': SENet18
        }

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
        
        size = 2
        processes = []
        
        for rank in range(size):
            p = Process(target=init_process, args=(rank, size, run))
            p.start()
            processes.append(p)
    
        for p in processes:
            p.join()
