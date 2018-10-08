'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import cPickle as pickle
import pprint as pp
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

best_acc = 0

def get_stat(data):
    # TODO: Add num backpropped
    stat = {}
    stat["average"] = np.average(data)
    stat["p50"] = np.percentile(data, 50)
    stat["p75"] = np.percentile(data, 75)
    stat["p90"] = np.percentile(data, 90)
    stat["max"] = max(data)
    stat["min"] = min(data)
    return stat

class State:

    def __init__(self, num_images, pickle_dir, pickle_prefix):
        self.num_images_backpropped = 0
        self.num_images_skipped = 0
        self.num_images_seen = 0
        self.sum_sp = 0
        self.pickle_dir = pickle_dir
        self.pickle_prefix = pickle_prefix

        self.init_images_hist(num_images)
        self.init_batch_stats()

    def init_images_hist(self, num_images):
        # Store frequency of each image getting backpropped
        keys = range(num_images)
        self.images_hist = dict(zip(keys, [0] * len(keys)))
        image_id_pickle_dir = os.path.join(self.pickle_dir, "image_id_hist")
        self.image_id_pickle_file = os.path.join(image_id_pickle_dir,
                                                 "{}_images_hist.pickle".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(image_id_pickle_dir):
            os.mkdir(image_id_pickle_dir)

    def init_batch_stats(self):
        self.batch_stats = []

        # Make batch stats pickle path
        batch_stats_pickle_dir = os.path.join(self.pickle_dir, "batch_stats")
        if not os.path.exists(batch_stats_pickle_dir):
            os.mkdir(batch_stats_pickle_dir)
        self.batch_stats_pickle_file = os.path.join(batch_stats_pickle_dir,
                                                    "{}_batch_stats.pickle".format(self.pickle_prefix))

    def update_images_hist(self, image_ids):
        for chosen_id in image_ids:
            self.images_hist[chosen_id] += 1

    def update_batch_stats(self, pool_losses=None,
                                 chosen_losses=None,
                                 pool_sps=None,
                                 chosen_sps=None):
        '''
        batch_stats = [{'chosen_losses': {stat},
                       'pool_losses': {stat}}]
        '''
        snapshot = {}
        snapshot["num_backpropped"] = self.num_images_backpropped
        snapshot["num_skipped"] = self.num_images_skipped
        if chosen_losses:
            snapshot["chosen_losses"] = get_stat(chosen_losses)
        if pool_losses:
            snapshot["pool_losses"] = get_stat(pool_losses)
        if chosen_sps:
            snapshot["chosen_sps"] = get_stat(chosen_sps)
        if pool_sps:
            snapshot["pool_sps"] = get_stat(pool_sps)
        self.batch_stats.append(snapshot)

    def write_summaries(self):
        with open(self.image_id_pickle_file, "wb") as handle:
            print(self.image_id_pickle_file)
            pickle.dump(self.images_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open(self.batch_stats_pickle_file, "wb") as handle:
        #     print(self.batch_stats_pickle_file)
        #     pickle.dump(self.batch_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def update_sum_sp(self, sp):
        self.num_images_seen += 1
        self.sum_sp += sp

    @property
    def average_sp(self):
        if self.num_images_seen == 0:
            return 1
        return self.sum_sp / float(self.num_images_seen)


# Training
def train_topk(args,
               net,
               trainloader,
               device,
               optimizer,
               epoch,
               state):

    print('\nEpoch: %d in train_topk' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    losses_pool = []
    data_pool = []
    targets_pool = []
    ids_pool = []
    num_backprop = 0
    num_skipped = 0
    loss_reduction = None

    for batch_idx, (data, targets, image_id) in enumerate(trainloader):

        data, targets = data.to(device), targets.to(device)

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
            chosen_losses = [losses_pool[i] for i in indices]
            state.num_images_skipped += len(data_pool) - len(chosen_data)

            data_batch = torch.stack(chosen_data, dim=1)[0]
            targets_batch = torch.cat(chosen_targets)
            output_batch = net(data_batch) # redundant

            # Update stats
            state.update_images_hist(chosen_ids)
            state.update_batch_stats(pool_losses = losses_pool, 
                                     chosen_losses = chosen_losses,
                                     pool_sps = [],
                                     chosen_sps = [])

            # Note: This will only work for batch size of 1
            loss_reduction = nn.CrossEntropyLoss(reduce=True)(output_batch, targets_batch)

            optimizer.zero_grad()
            loss_reduction.backward()
            optimizer.step()
            train_loss += loss_reduction.item()
            num_backprop += len(chosen_data)
            state.num_images_backpropped += len(chosen_data)

            losses_pool = []
            data_pool = []
            targets_pool = []
            ids_pool = []

            output = output_batch
            targets = targets_batch

        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % args.log_interval == 0 and loss_reduction is not None:
            print('train_debug,{},{},{},{:.6f},{:.6f},{},{:.6f}'.format(
                        epoch,
                        state.num_images_backpropped,
                        state.num_images_skipped,
                        loss_reduction.item(),
                        train_loss / float(num_backprop),
                        time.time(),
                        100.*correct/total))

        # Stop epoch rightaway if we've exceeded maximum number of epochs
        if args.max_num_backprops:
            if args.max_num_backprops <= state.num_images_backpropped:
                return num_backprop

    return

# Training
def train_baseline(args,
               net,
               trainloader,
               device,
               optimizer,
               epoch,
               state):

    print('\nEpoch: %d in train_baseline' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    losses_pool = []
    num_backprop = 0

    for batch_idx, (data, targets, image_id) in enumerate(trainloader):

        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(data)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        optimizer.step()

        losses_pool.append(loss.item())
        num_backprop += len(data)
        state.num_images_backpropped += len(data)


        # Update stats
        state.update_images_hist([t.item() for t in image_id])
        state.update_batch_stats(pool_losses = losses_pool, 
                                 chosen_losses = losses_pool,
                                 pool_sps = [],
                                 chosen_sps = [])

        losses_pool = []

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % args.log_interval == 0:
            print('train_debug,{},{},{},{:.6f},{:.6f},{},{:.6f}'.format(
                        epoch,
                        state.num_images_backpropped,
                        state.num_images_skipped,
                        loss.item(),
                        train_loss / float(num_backprop),
                        time.time(),
                        100.*correct/total))

        # Stop epoch rightaway if we've exceeded maximum number of epochs
        if args.max_num_backprops:
            if args.max_num_backprops <= state.num_images_backpropped:
                return num_backprop

    return


# Training
def train_sampling(args,
                   net,
                   trainloader,
                   device,
                   optimizer,
                   epoch,
                   state):

    print('\nEpoch: %d in train_sampling' % epoch)
    net.train()
    cumulative_loss = 0
    cumulative_selected_loss = 0
    correct = 0
    total = 0

    losses_pool = []
    data_pool = []
    targets_pool = []
    ids_pool = []
    sps_pool = []

    chosen_data = []
    chosen_losses = []
    chosen_targets = []
    chosen_ids = []
    chosen_sps = []

    num_backprop = 0
    num_skipped = 0

    select_probs = -1

    for batch_idx, (data, targets, image_ids) in enumerate(trainloader):

        data, targets = data.to(device), targets.to(device)

        outputs = net(data)
        losses = nn.CrossEntropyLoss(reduce=False)(outputs, targets)

        # Prepare output for L2 distance
        softmax_outputs = nn.Softmax()(outputs)

        for loss, softmax_output, target, datum, image_id in zip(losses, softmax_outputs, targets, data, image_ids):

            #print("Softmax Output: ", softmax_output.data)

            # Prepare target for L2 distance
            target_vector = np.zeros(len(softmax_output.data))
            target_vector[target.item()] = 1
            target_tensor = torch.Tensor(target_vector)
            #print("Target: ", target_tensor)

            l2_dist = torch.dist(target_tensor.to(device), softmax_output)
            #print("L2 Dist: ", l2_dist.item())

            l2_dist *= l2_dist
            #print("L2 Dist Squared: ", l2_dist.item())

            # Translate l2_dist to new range
            old_max = .81
            old_min = args.sampling_min
            new_max = 1
            new_min = args.sampling_min
            old_range = (old_max - old_min)  
            new_range = (new_max - new_min) 
            l2_dist = (((l2_dist - old_min) * new_range) / old_range) + new_min
            #print("Translated l2_dist: ", l2_dist)

            # Clamp l2_dist into a probability
            select_probs = torch.clamp(l2_dist, min=args.sampling_min, max=1)
            #print("Chosen Probs: ", select_probs.item())

            cumulative_loss += loss.item()
            losses_pool.append(loss.item())
            sps_pool.append(select_probs.item())
            state.update_sum_sp(select_probs.item())

            draw = np.random.uniform(0, 1)

            if draw < select_probs.item():

                # Add to chosen data
                chosen_losses.append(loss.item())
                chosen_data.append(datum)
                chosen_targets.append(target)
                chosen_ids.append(image_id.item())
                chosen_sps.append(select_probs.item())

                if len(chosen_losses) == args.batch_size:

                    # Make new batch of selected examples
                    data_batch = torch.stack(chosen_data)
                    targets_batch = torch.stack(chosen_targets)

                    # Run forward pass
                    output_batch = net(data_batch) # redundant
                    loss_batch = nn.CrossEntropyLoss(reduce=False)(output_batch, targets_batch)

                    # Scale each loss by image-specific select probs
                    chosen_sps_tensor = torch.tensor(chosen_sps, dtype=torch.float)
                    loss_batch = torch.mul(loss_batch, chosen_sps_tensor.to(device))

                    # Reduce loss
                    loss_batch = loss_batch.mean()

                    # Scale loss by average select probs
                    loss_batch.data *= state.average_sp

                    optimizer.zero_grad()
                    loss_batch.backward()
                    optimizer.step()

                    # Bookkeeping
                    cumulative_selected_loss += loss_batch.item()
                    num_backprop += args.batch_size
                    state.num_images_backpropped += args.batch_size
                    state.update_images_hist(chosen_ids)
                    state.update_batch_stats(pool_losses = losses_pool, 
                                             chosen_losses = chosen_losses,
                                             pool_sps = sps_pool,
                                             chosen_sps = chosen_sps)

                    # Reset  pools
                    losses_pool = []
                    sps_pool = []
                    chosen_data = []
                    chosen_losses = []
                    chosen_targets = []
                    chosen_ids = []
                    chosen_sps = []

            else:
                state.num_images_skipped += 1
                num_skipped += 1
                print("Skipping image with sp {}".format(select_probs))

            data_pool.append(data.data[0])
            targets_pool.append(targets)
            ids_pool.append(image_id.item())
            sps_pool.append(select_probs.item())

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % args.log_interval == 0 and num_backprop > 0:
            print('train_debug,{},{},{},{:.6f},{:.6f},{},{:.6f}'.format(
                        epoch,
                        state.num_images_backpropped,
                        state.num_images_skipped,
                        cumulative_loss / float(num_backprop + num_skipped),
                        cumulative_selected_loss / float(num_backprop),
                        time.time(),
                        100.*correct/total))

        # Stop epoch rightaway if we've exceeded maximum number of epochs
        if args.max_num_backprops:
            if args.max_num_backprops <=  + num_backprop:
                return num_backprop

    return

def test(args,
         net,
         testloader,
         device,
         epoch,
         state):

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
    print('test_debug,{},{},{},{:.6f},{:.6f},{}'.format(
                epoch,
                state.num_images_backpropped,
                state.num_images_skipped,
                test_loss,
                100.*correct/total,
                time.time()))


    # Save checkpoint.
    global best_acc
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        net_state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        checkpoint_dir = os.path.join(args.pickle_dir, "checkpoint")
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, args.pickle_prefix + "_ckpt.t7")
        print("Saving checkpoint at {}".format(checkpoint_file))
        torch.save(net_state, checkpoint_file)
        best_acc = acc


def main():

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--decay', default=5e-4, type=float, help='decay')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--net', default="resnet", metavar='N',
                        help='which network architecture to train')

    parser.add_argument('--sb-strategy', default="topk", metavar='N',
                        help='Selective backprop strategy among {topk, sampling}')
    parser.add_argument('--sampling-min', type=float, default=0.05,
                        help='Minimum sampling rate for sampling strategy')
    parser.add_argument('--top-k', type=int, default=8, metavar='N',
                        help='how many images to backprop per batch')
    parser.add_argument('--pool-size', type=int, default=16, metavar='N',
                        help='how many images to backprop per batch')
    parser.add_argument('--pickle-dir', default="/tmp/",
                        help='directory for pickles')
    parser.add_argument('--pickle-prefix', default="stats",
                        help='file prefix for pickles')
    parser.add_argument('--max-num-backprops', type=int, default=None, metavar='N',
                        help='how many images to backprop total')

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

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    #optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    ## Selective backprop setup ##

    assert(args.pool_size >= args.top_k)

    # Partition training set to get more test datapoints
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    trainset = [t + (i,) for i, t in enumerate(trainset)]
    chunk_size = args.pool_size * 100
    partitions = [trainset[i:i + chunk_size] for i in xrange(0, len(trainset), chunk_size)]

    state = State(len(trainset), args.pickle_dir, args.pickle_prefix)

    for epoch in range(start_epoch, start_epoch+500):
        for partition in partitions:
            trainloader = torch.utils.data.DataLoader(partition, batch_size=args.batch_size, shuffle=True, num_workers=2)
            test(args, net, testloader, device, epoch, state)

            # Stop training rightaway if we've exceeded maximum number of epochs
            if args.max_num_backprops:
                if args.max_num_backprops <= state.num_images_backpropped:
                    return

            if args.sb_strategy == "topk":
                trainer = train_topk
            elif args.sb_strategy == "sampling":
                trainer = train_sampling
            elif args.sb_strategy == "baseline":
                trainer = train_baseline
            else:
                print("Unknown selective backprop strategy {}".format(args.sb_strategy))
                exit()

            trainer(args,
                    net,
                    trainloader,
                    device,
                    optimizer,
                    epoch,
                    state)

            # Write out summary statistics
            state.write_summaries()


if __name__ == '__main__':
    main()
