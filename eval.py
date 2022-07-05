'''Output Model probs and labels'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import os.path

import os
import argparse

from models import *
from utils import progress_bar

from cifar import CIFAR10_C 

# import wandb
# wandb.init(project="cifar-test")


parser = argparse.ArgumentParser(description='Model Evaluation')
parser.add_argument('--checkpoint', default="./checkpoint/ckpt.pth", type=str)
parser.add_argument('--model', default="cf10-densenet", type=str)
parser.add_argument('--dataset', default="cf10c-gaussian_noise", type=str)
parser.add_argument('--data_fp', default="/nlp/scr/jiayili/data/CIFAR-10-C/gaussian_noise.npy", type=str)
parser.add_argument('--labels_fp', default="/nlp/scr/jiayili/data/CIFAR-10-C/labels.npy", type=str)

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy of the model on its source domain
epoch = 0  # epoch from which the best model is taken

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# testset = torchvision.datasets.CIFAR10(
#     root='/jagupard11/scr0/jiayili', train=False, download=True, transform=transform_test)

rt = args.data_fp
labels_rt = args.labels_fp

testset = CIFAR10_C(root = rt, labels_root = labels_rt, transform = transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if True:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    epoch = checkpoint['epoch']
    print("Finish Loading model {} with accuracy {}, achieved from epoch {}".format(args.model, best_acc, epoch))

criterion = nn.CrossEntropyLoss()

# Evaluate the model on the dataset

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    labels_fp = "./results/{}_labels.csv".format(args.dataset)
    preds_fp = "./results/{}_{}_preds.csv".format(args.model, args.dataset)
    labels_exist = os.path.exists(labels_fp)
    preds_exist = os.path.exists(preds_fp)
    if labels_exist: print("{} already exists".format(labels_fp))
    if preds_exist: print("{} already exists".format(preds_fp))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if not labels_exist:
                t_np = targets.cpu().numpy()
                t_df = pd.DataFrame(t_np)
                t_df.to_csv(labels_fp, mode='a', index=False, header=False)
            if not preds_exist:
                o_np = outputs.cpu().numpy()
                o_df = pd.DataFrame(o_np)
                o_df.to_csv(preds_fp, mode='a', index=False, header=False)

            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print("Accuracy of {} on {} is {}".format(args.model, args.dataset, 100.*correct/total))

test()
