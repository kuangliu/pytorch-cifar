'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import torch.nn.utils.prune as prune
from prune_params import get_prune_params, print_sparsity
import os
import argparse

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

parser.add_argument('--prune_one_shot', '-pos', action='store_true',
                    help='resume from checkpoint with one shot pruning')

parser.add_argument('--prune_iterative', '-pit', action='store_true',
                    help='resume from checkpoint with iterative pruning')

parser.add_argument('--prune_amount', '-pr', action='store_true',
                    help='resume from checkpoint with one shot pruning')
parser.add_argument('-pa', default=0, type=float, help='pruning amount')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
pos_best_acc = 0  # best accuracy for one shot pruned model

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
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=256, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=256, shuffle=False, num_workers=2)

model_save_path = './checkpoint/ckpt.pth'
prune_amount = 0

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    # pos_best_acc = checkpoint['pos_best_acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

if args.prune_one_shot:
    print('Perform one shot pruning and retraining')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    # pos_best_acc = checkpoint['pos_best_acc']
    start_epoch = checkpoint['epoch']


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    global pos_best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    if args.prune_iterative:
        acc = 100. * correct / total
        if acc > pos_best_acc:
            # Remove pruning before saving
            prune_params = get_prune_params(net)
            for prune_param in prune_params:
                prune.remove(prune_param[0], 'weight')

            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'pos_best_acc': pos_best_acc,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_prune_iterative_' + str(int(100 * prune_amount)) + '.pth')
            pos_best_acc = acc
            print_sparsity(net)

            # apply pruning masks back before continuing (this will be the same since model is already pruned)
            prune.global_unstructured(get_prune_params(net), pruning_method=prune.L1Unstructured,
                                      importance_scores=None, amount=prune_amount)

    elif args.prune_one_shot:

        acc = 100. * correct / total
        if acc > pos_best_acc:
            # Remove pruning before saving
            prune_params = get_prune_params(net)
            for prune_param in prune_params:
                prune.remove(prune_param[0], 'weight')

            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'pos_best_acc': pos_best_acc,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_prune_one_shot_' + str(int(100 * prune_amount)) + '.pth')
            pos_best_acc = acc
            print_sparsity(net)

            # apply pruning masks back before continuing (this will be the same since model is already pruned)
            prune.global_unstructured(get_prune_params(net), pruning_method=prune.L1Unstructured,
                                      importance_scores=None, amount=prune_amount)

    else:
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'pos_best_acc': pos_best_acc
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc


if __name__ == '__main__':
    # num_epoch_train, num_epoch_one_shot, num_epoch_iterative = (200, 100, 25)
    num_epoch_train, num_epoch_one_shot, num_epoch_iterative = (4, 4, 4)
    # Iterative pruning
    if args.prune_iterative:
        total_prune_amount = args.pa

        num_pruning_iter = 4
        # increase the pruning amount over num_pruning_iter iterations
        for prune_x in range(num_pruning_iter):
            prune_amount = (prune_x + 1) * total_prune_amount / num_pruning_iter
            parameters_to_prune = get_prune_params(net)
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, importance_scores=None,
                                      amount=prune_amount)
            for epoch in range(start_epoch, start_epoch + num_epoch_iterative):
                train(epoch)
                test(epoch)
                scheduler.step()

    # One shot pruning and retraining
    elif args.prune_one_shot:
        prune_amount = args.pa
        parameters_to_prune = get_prune_params(net)
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, importance_scores=None,
                                  amount=prune_amount)
        for epoch in range(start_epoch, start_epoch + num_epoch_one_shot):
            train(epoch)
            test(epoch)
            scheduler.step()

    # No pruning
    else:
        for epoch in range(start_epoch, start_epoch + num_epoch_train):
            train(epoch)
            test(epoch)
            scheduler.step()
