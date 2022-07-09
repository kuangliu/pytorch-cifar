'''Active learning with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import numpy as np

from torch.utils.data import DataLoader, Dataset, Subset
from wilds import get_dataset
from wilds.datasets.wilds_dataset import WILDSSubset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

from al_utils import query_the_oracle

import os
import argparse

from models import *
from utils import progress_bar

import wandb
#wandb.init(project="al")


parser = argparse.ArgumentParser(description='Active Learning Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
#"/self/scr-sync/nlp/waterbirds"
parser.add_argument('--root_dir', default="/self/scr-sync/nlp/waterbirds", type=str)
parser.add_argument('--checkpoint', default="./checkpoint/al_waterbirds.pth", type=str)
parser.add_argument('--save_name', default="./checkpoint/al_waterbirds.pth", type=str)
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="waterbirds", download=True, root_dir = args.root_dir)

# Get the training and validation set
train_data = dataset.get_subset(
    "train",
    transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    ),
)
eval_data = dataset.get_subset(
    "val",
    transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    ),
)

print('Train set size: ', len(train_data))
print('Eval set size: ', len(eval_data))

# We assume that in the beginning, the entire train set is unlabeled
unlabeled_mask = np.ones(len(train_data))

# Prepare the standard data loader
train_loader = get_train_loader("standard", train_data, batch_size=8)
eval_loader = get_eval_loader("standard", eval_data, batch_size=8)

# Model
print('==> Building model..')
# Number of classes in the classification problem
num_classes = 2
net = torchvision.models.resnet50(num_classes = 2)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    query_start_epoch = checkpoint['query_start_epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# wandb.config = {
#   "learning_rate": args.lr,
#   "epochs": 100,
#   "batch_size": args.batch_size
# }

ram = 0

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, metadata) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        #wandb.log({"loss": loss.item()})
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, query_start_epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, metadata) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(eval_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            #wandb.log({"test accuracy": 100.*correct/total})

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'query_start_epoch': query_start_epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, args.save_name)
        best_acc = acc

    return 100.*correct/total


# for epoch in range(start_epoch, start_epoch+200):
#     train(epoch)
#     test(epoch)
#     scheduler.step()

# Uncertainty sampling parameters
seed_size = 40
num_queries = 10
query_size = 30
query_strategy = 'least_confidence' # 'least_confidence', 'margin', 'random'

# Label the initial subset
query_the_oracle(unlabeled_mask, net, device, train_data, query_size=seed_size, 
                query_strategy='random', pool_size=0, batch_size=args.batch_size)
# print(unlabeled_mask.size)
# print(np.nonzero(unlabeled_mask == 0)[0][:10])

# Pre-train on the initial subset
epoch = 0
query_start_epoch = np.zeros(num_queries + 1) # store the start epoch index for each query; the first query is the initial seed set with start epoch 0

labeled_idx = np.where(unlabeled_mask == 0)[0]
train_loader = get_train_loader("standard", WILDSSubset(train_data, labeled_idx, transform=None), 
                                batch_size=args.batch_size, num_workers=2)
previous_test_acc = 0
current_test_acc = 1
while current_test_acc > previous_test_acc:
    previous_test_acc = current_test_acc
    train_loss = train(epoch)
    current_test_acc = test(epoch, query_start_epoch)
    #scheduler.step()
    epoch += 1

# Start the query loop 
for query in range(num_queries):
    print(query_start_epoch)
    query_start_epoch[query + 1] = epoch

    # Query the oracle for more labels
    query_the_oracle(unlabeled_mask, net, device, train_data, query_size=query_size, 
                    query_strategy=query_strategy, pool_size=0, batch_size=args.batch_size)
    
    # Train the model on the data that has been labeled so far:
    labeled_idx = np.where(unlabeled_mask == 0)[0]
    train_loader = get_train_loader("standard", WILDSSubset(train_data, labeled_idx, transform=None), 
                                    batch_size=args.batch_size, num_workers=2)
    previous_test_acc = 0
    current_test_acc = 1
    while current_test_acc > previous_test_acc:
        previous_test_acc = current_test_acc
        train_loss = train(epoch)
        current_test_acc = test(epoch, query_start_epoch)
        #scheduler.step()
        epoch += 1