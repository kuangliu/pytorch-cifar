'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchinfo import summary

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--net', default='SimpleDLA')
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--prune', action='store_true')
parser.add_argument('--pruning_rate', type=float, default=0.30)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--select_device', type=str, default='gpu', help='gpu | cpu')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() and args.select_device == 'gpu' else 'cpu'
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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.net == 'VGG19': net = VGG('VGG19')
elif args.net == 'ResNet18': net = ResNet18()
elif args.net == 'PreActResNet18': net = PreActResNet18()
elif args.net == 'GoogLeNet': net = GoogLeNet()
elif args.net == 'DenseNet121': net = DenseNet121()
elif args.net == 'ResNeXt29_2x64d': net = ResNeXt29_2x64d()
elif args.net == 'MobileNet': net = MobileNet()
elif args.net == 'MobileNetV2': net = MobileNetV2()
elif args.net == 'DPN92': net = DPN92()
elif args.net == 'ShuffleNetG2': net = ShuffleNetG2()
elif args.net == 'SENet18': net = SENet18()
elif args.net == 'ShuffleNetV2': net = ShuffleNetV2(1)
elif args.net == 'EfficientNetB0': net = EfficientNetB0()
elif args.net == 'RegNetX_200MF': net = RegNetX_200MF()
elif args.net == 'SimpleDLA': net = SimpleDLA()

# Borrow sparsity() and prune() from
# https://github.com/ultralytics/yolov5/blob/a2a1ed201d150343a4f9912d644be2b210206984/utils/torch_utils.py#L174
def sparsity(model):
    # Return global model sparsity
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a

def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
            print(' %.3g global sparsity' % sparsity(model))


def count_layer_params(model, layer_name=nn.Conv2d):
    print('\n\n layer_name: ', layer_name)
    total_params = 0
    total_traina_params = 0
    n_layers = 0
    for name, m in model.named_modules():
        if isinstance(m, layer_name):
            # print('\nm:', m)
            # print('\ndir(m): ', dir(m))

            for name, parameter in m.named_parameters():
                params = parameter.numel()
                total_params += params
                if not parameter.requires_grad: continue
                n_layers += 1
                total_traina_params += params
    print('\n\nlayer_name: {}, total_params: {}, total_traina_params: {}, n_layers: {}'.\
        format(layer_name, total_params, total_traina_params, n_layers))
    # time.sleep(100)

net = net.to(device)
if device == 'cuda':
    if args.train: net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    print('\n\ndevice: ', device)
    checkpoint = torch.load('./checkpoint/{}_ckpt.pth'.format(args.net), map_location=device)
    net.load_state_dict(checkpoint['net'], strict=False)
    print('\n model weights loaded!')
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


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
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    if args.prune:
        prune(net, args.pruning_rate)
    input_size = (1, 3, 32, 32)
    summary(net, input_size)
    count_layer_params(net)

    # print('------------')
    # print(net)

    def extract_features_labels(my_net, my_data):
        intermediate_outputs = []

        def hook(module, input, output):
            for output_each in output:
                intermediate_outputs.append(output_each.detach().cpu().tolist()) # intermediate outputs

        # print('np.shape(my_data): ', np.shape(my_data))
        # print('\n net: ', net)
        # print('\n dir(my_net): ', dir(my_net))
        # print('\n dir(my_net.children()): ', dir(my_net.children()))
        my_net.layer4[1].shortcut.register_forward_hook(hook) # intermediate outputs from the third last fc layer

        features = []
        labels = []
        # print('\n np.shape(my_data): ', np.shape(my_data))
        '''
        e.g.
        np.shape(my_data):  torch.Size([1, 3, 32, 32])
        '''
        # e.g. torch.Size([1, 3, 32, 32])
        # for step, (x, y) in enumerate(my_data):
        #     batch_x = Variable(x)
        #     batch_y = Variable(y)
        #     output = my_net(batch_x)
        #     labels.extend(batch_y.numpy().tolist())

        output = my_net(my_data)
        features = torch.from_numpy(np.array(intermediate_outputs))
        # print('\n np.shape(output): ', np.shape(output))
        # print('\n np.shape(np.array(features)): ', np.shape(np.array(features)))
        '''
        e.g.
        np.shape(output):  torch.Size([1, 10])
        np.shape(np.array(features)):  (1, 512, 4, 4)
        '''
        # features = torch.flatten(features)
        # print('\n after flattened - np.shape(features): ', np.shape(features))
        # return np.array(features)
        # print('\n np.shape(features): ', np.shape(features))
        return features

    def vis_compute_sim(net, input_0, input_1):
        feats_0 = extract_features_labels(net, input_0)
        feats_1 = extract_features_labels(net, input_1)
        input_0 = (input_0 / 2 + 0.5) # * 255
        input_1 = (input_1 / 2 + 0.5) # * 255

        plt.imshow(np.transpose(torch.squeeze(input_0).cpu(), (1, 2, 0))); plt.show()
        plt.imshow(np.transpose(torch.squeeze(input_1).cpu(), (1, 2, 0))); plt.show()
        # sim_score = np.cos(np.array(feats_0).flatten(), np.array(feats_1).flatten())
        print('\n np.shape(feats_0): ', np.shape(feats_0))
        # torch.Size([1, 512, 4, 4])
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        sim_score = cos(feats_0, feats_1)
        print('\n np.shape(sim_score): ', np.shape(sim_score))
        sim_score = sim_score[0][0][0] # torch.mean(sim_score)
        return sim_score

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # print('device: ', device)
            # print('\n np.shape(inputs): ', np.shape(inputs))
            # e.g. torch.Size([5, 3, 32, 32])
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # print('targets: ', targets)
            # print('\n np.shape(targets): ', np.shape(targets))
            # print('\n np.shape(inputs): ', np.shape(inputs))
            # e.g.
            # np.shape(targets):  torch.Size([5])
            # np.shape(inputs):  torch.Size([5, 3, 32, 32]) # [batch_size, channels, , ]

            for i in range(args.test_batch_size):
                for j in range(args.test_batch_size):
                    input_0, input_1 = torch.unsqueeze(inputs[i], dim=0), torch.unsqueeze(inputs[j], dim=0)
                    sim_score = vis_compute_sim(net, input_0, input_1)
                    print('\n =====================')
                    print('\n i: ', i, ', j: ', j)
                    print('\n targets[i]: ', targets[i], ', targets[j]: ', targets[j])
                    print('\n sim_score: ', sim_score)
                    print('\n ---------------------')

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

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
        torch.save(state, './checkpoint/{}_ckpt.pth'.format(args.net))
        best_acc = acc

print('\n\nargs.train: ', args.train, ', args.test:', args.test)
for epoch in range(args.epochs):
    if args.train: train(epoch)
    if args.test:
        test(epoch)
        if not args.train: break
    scheduler.step()
