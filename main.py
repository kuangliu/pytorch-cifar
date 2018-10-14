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
        self.init_target_confidences()

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

    def init_target_confidences(self):
        self.target_confidences = {}

        target_confidences_pickle_dir = os.path.join(self.pickle_dir, "target_confidences")
        self.target_confidences_pickle_file = os.path.join(target_confidences_pickle_dir,
                                                           "{}_target_confidences.pickle".format(self.pickle_prefix))

        # Make images hist pickle path
        if not os.path.exists(target_confidences_pickle_dir):
            os.mkdir(target_confidences_pickle_dir)

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

    def update_target_confidences(self, epoch, confidences):
        if epoch not in self.target_confidences.keys():
            self.target_confidences[epoch] = {"confidences": []}
        self.target_confidences[epoch]["confidences"] += confidences
        self.target_confidences[epoch]["num_backpropped"] = self.num_images_backpropped

    def write_summaries(self):
        #with open(self.image_id_pickle_file, "wb") as handle:
        #    print(self.image_id_pickle_file)
        #    pickle.dump(self.images_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.target_confidences_pickle_file, "wb") as handle:
            print(self.target_confidences_pickle_file)
            pickle.dump(self.target_confidences, handle, protocol=pickle.HIGHEST_PROTOCOL)

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


class ImageIdHistLogger(object):

    def __init__(self, pickle_dir, pickle_prefix, num_images):
        self.current_epoch = 0
        self.pickle_dir = pickle_dir
        self.pickle_prefix = pickle_prefix
        self.init_images_hist(num_images)

    def next_epoch(self):
        self.write()
        self.current_epoch += 1

    def init_images_hist(self, num_images):
        # Store frequency of each image getting backpropped
        keys = range(num_images)
        self.images_hist = dict(zip(keys, [0] * len(keys)))
        image_id_pickle_dir = os.path.join(self.pickle_dir, "image_id_hist")
        self.image_id_pickle_file = os.path.join(image_id_pickle_dir,
                                                 "{}_images_hist".format(self.pickle_prefix))
        # Make images hist pickle path
        if not os.path.exists(image_id_pickle_dir):
            os.mkdir(image_id_pickle_dir)

    def update_images_hist(self, image_ids):
        for chosen_id in image_ids:
            self.images_hist[chosen_id] += 1

    def handle_backward_batch(self, batch):
        ids = [example.image_id.item() for example in batch if example.select]
        self.update_images_hist(ids)

    def write(self):
        latest_file = "{}.pickle".format(self.image_id_pickle_file)
        with open(latest_file, "wb") as handle:
            print(latest_file)
            pickle.dump(self.images_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

        epoch_file = "{}.epoch_{}.pickle".format(self.image_id_pickle_file,
                                                 self.current_epoch)
        if self.current_epoch % 10 == 0:
            with open(epoch_file, "wb") as handle:
                print(epoch_file)
                pickle.dump(self.images_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Logger(object):

    def __init__(self, log_interval=1):
        self.current_epoch = 0
        self.current_batch = 0
        self.log_interval = log_interval

        self.global_num_backpropped = 0
        self.global_num_skipped = 0

        self.partition_loss = 0
        self.partition_backpropped_loss = 0
        self.partition_num_backpropped = 0
        self.partition_num_skipped = 0
        self.partition_num_correct = 0

    def next_epoch(self):
        self.current_epoch += 1

    @property
    def partition_seen(self):
        return self.partition_num_backpropped + self.partition_num_skipped

    @property
    def average_partition_loss(self):
        return self.partition_loss / float(self.partition_seen)

    @property
    def average_partition_backpropped_loss(self):
        return self.partition_backpropped_loss / float(self.partition_num_backpropped)

    @property
    def partition_accuracy(self):
        return 100. * self.partition_num_correct / self.partition_seen

    @property
    def train_debug(self):
        return 'train_debug,{},{},{},{:.6f},{:.6f},{},{:.6f}'.format(
            self.current_epoch,
            self.global_num_backpropped,
            self.global_num_skipped,
            self.average_partition_loss,
            self.average_partition_backpropped_loss,
            time.time(),
            self.partition_accuracy)

    def next_partition(self):
        self.partition_loss = 0
        self.partition_backpropped_loss = 0
        self.partition_num_backpropped = 0
        self.partition_num_correct = 0

    def handle_forward_batch(self, batch):
        # Populate batch_stats
        self.partition_loss += sum([example.loss for example in batch])

    def handle_backward_batch(self, batch):

        self.current_batch += 1

        num_backpropped = sum([int(example.select) for example in batch])
        num_skipped = sum([int(not example.select) for example in batch])
        self.global_num_backpropped += num_backpropped
        self.global_num_skipped += num_skipped

        self.partition_num_backpropped += num_backpropped
        self.partition_num_skipped += num_skipped
        self.partition_backpropped_loss += sum([example.backpropped_loss
                                                for example in batch
                                                if example.backpropped_loss])
        self.partition_num_correct += sum([int(example.is_correct) for example in batch])

        self.write()

    def write(self):
        if self.current_batch % self.log_interval == 0:
            print(self.train_debug)


class Example(object):
    # TODO: Add ExampleCollection class
    def __init__(self,
                 loss=None,
                 softmax_output=None,
                 target=None,
                 datum=None,
                 image_id=None,
                 select_probability=None):
        self.loss = loss.detach()
        self.softmax_output = softmax_output.detach()
        self.target = target.detach()
        self.datum = datum.detach()
        self.image_id = image_id.detach()
        self.select_probability = select_probability
        self.backpropped_loss = None   # Populated after backprop

    @property
    def predicted(self):
        _, predicted = self.softmax_output.max(0)
        return predicted

    @property
    def is_correct(self):
        return self.predicted.eq(self.target)


class Backpropper(object):

    def __init__(self, device, net, optimizer, recenter=False):
        self.optimizer = optimizer
        self.net = net
        self.device = device
        self.recenter = recenter
        self.sum_select_probabilities = 0
        self.total_num_examples = 0

    def update_sum_probabilities(self, batch):
        probabilities = [example.select_probability for example in batch]
        self.sum_select_probabilities += sum(probabilities)
        self.total_num_examples += len(probabilities)

    def get_chosen_data_tensor(self, batch):
        chosen_data = [example.datum for example in batch if example.select]
        return torch.stack(chosen_data)

    def get_chosen_targets_tensor(self, batch):
        chosen_targets = [example.target for example in batch if example.select]
        return torch.stack(chosen_targets)

    def get_chosen_probabilities_tensor(self, batch):
        probabilities = [example.select_probability for example in batch if example.select]
        return torch.tensor(probabilities, dtype=torch.float)

    @property
    def average_select_probability(self):
        return float(self.sum_select_probabilities) / self.total_num_examples

    def backward_pass(self, batch):

        self.update_sum_probabilities(batch)

        data = self.get_chosen_data_tensor(batch)
        targets = self.get_chosen_targets_tensor(batch)
        probabilities = self.get_chosen_probabilities_tensor(batch)

        # Run forward pass
        # Necessary if the network has been updated between last forward pass
        outputs = self.net(data) 
        losses = nn.CrossEntropyLoss(reduce=False)(outputs, targets)

        # Scale each loss by image-specific select probs
        losses = torch.div(losses, probabilities.to(self.device))

        # Scale loss by average select probs
        if self.recenter:
            losses = torch.mul(losses, self.average_select_probability)

        # Add for logging selected loss
        for example, loss in zip(batch, losses):
            example.backpropped_loss = loss

        # Reduce loss
        loss = losses.mean()

        # Run backwards pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return batch


class Trainer(object):
    def __init__(self, device, net, selector, backpropper, batch_size, max_num_backprops=float('inf')):
        self.device = device
        self.net = net
        self.selector = selector
        self.backpropper = backpropper
        self.batch_size = batch_size
        self.backprop_queue = []
        self.forward_pass_handlers = []
        self.backward_pass_handlers = []
        self.global_num_backpropped = 0
        self.max_num_backprops = 0
        self.on_backward_pass(self.update_num_backpropped)

    def update_num_backpropped(self, batch):
        self.global_num_backpropped += sum([1 for e in batch if e.select])

    def on_forward_pass(self, handler):
        self.forward_pass_handlers.append(handler)

    def on_backward_pass(self, handler):
        self.backward_pass_handlers.append(handler)

    def emit_forward_pass(self, batch):
        for handler in self.forward_pass_handlers:
            handler(batch)

    def emit_backward_pass(self, batch):
        for handler in self.backward_pass_handlers:
            handler(batch)

    def stopped(self):
        return self.global_num_backpropped >= self.max_num_backprops

    def train(self, trainloader):
        self.net.train()
        for batch in trainloader:
            if self.stopped: break
            self.train_batch(batch)

    def train_batch(self, batch):
        forward_pass_batch = self.forward_pass(*batch)
        annotated_forward_batch = self.selector.mark(forward_pass_batch)
        self.emit_forward_pass(annotated_forward_batch)
        self.backprop_queue += annotated_forward_batch
        backprop_batch = self.get_batch()
        if backprop_batch:
            annotated_backward_batch = self.backpropper.backward_pass(backprop_batch)
            self.emit_backward_pass(annotated_backward_batch)

    def forward_pass(self, data, targets, image_ids):
        data, targets = data.to(self.device), targets.to(self.device)
        outputs = self.net(data)
        losses = nn.CrossEntropyLoss(reduce=False)(outputs, targets)

        # Prepare output for L2 distance
        softmax_outputs = nn.Softmax()(outputs)

        examples = zip(losses, softmax_outputs, targets, data, image_ids)
        return [Example(*example) for example in examples]

    def get_batch(self):
        num_images_to_backprop = 0
        for index, example in enumerate(self.backprop_queue):
            num_images_to_backprop += int(example.select)
            if num_images_to_backprop == self.batch_size:
                # Note: includes item that should and shouldn't be backpropped
                backprop_batch = self.backprop_queue[:index+1]
                self.backprop_queue = self.backprop_queue[index+1:]
                return backprop_batch
        return None


class Selector(object):
    def __init__(self, batch_size, probability_calcultor):
        self.get_select_probability = probability_calcultor.get_probability

    def select(self, example):
        select_probability = example.select_probability
        draw = np.random.uniform(0, 1)
        return draw < select_probability.item()

    def mark(self, forward_pass_batch):
        for example in forward_pass_batch:
            example.select_probability = self.get_select_probability(
                    example.target,
                    example.softmax_output)
            example.select = self.select(example)
        return forward_pass_batch


class SelectProbabiltyCalculator(object):
    def __init__(self, sampling_min, num_classes, device, square=False, translate=False):
        self.sampling_min = sampling_min
        self.num_classes = num_classes
        self.device = device
        self.square = square
        self.translate = translate
        self.old_max = .9
        if self.square:
            self.old_max *= self.old_max

    def get_probability(self, target, softmax_output):
        target_tensor = self.hot_encode_scalar(target)
        l2_dist = torch.dist(target_tensor.to(self.device), softmax_output)
        if self.square:
            l2_dist *= l2_dist
        if self.translate:
            l2_dist = self.translate_probability(l2_dist)
        return torch.clamp(l2_dist, min=self.sampling_min, max=1)

    def hot_encode_scalar(self, target):
        target_vector = np.zeros(self.num_classes)
        target_vector[target.item()] = 1
        target_tensor = torch.Tensor(target_vector)
        return target_tensor

    def translate_probability(self, l2_dist):
        new_max = 1
        old_range = (self.old_max - self.sampling_min)  
        new_range = (new_max - self.sampling_min) 
        l2_dist = (((l2_dist - self.sampling_min) * new_range) / old_range) + self.sampling_min
        return l2_dist


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

            softmax_outputs = nn.Softmax()(outputs)
            targets_array = targets.cpu().numpy()
            outputs_array = softmax_outputs.cpu().numpy()
            confidences = [o[t] for t, o in zip(targets_array, outputs_array)]
            state.update_target_confidences(epoch, confidences[:10])

    test_loss /= len(testloader.dataset)
    print('test_debug,{},{},{},{:.6f},{:.6f},{}'.format(
                epoch,
                state.num_images_backpropped,
                state.num_images_skipped,
                test_loss,
                100.*correct/total,
                time.time()))

    # Save checkpoint.
    if epoch == 65:
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
    parser.add_argument('--sb-start-epoch', type=int, default=0,
                        help='epoch to start selective backprop')
    parser.add_argument('--pickle-dir', default="/tmp/",
                        help='directory for pickles')
    parser.add_argument('--pickle-prefix', default="stats",
                        help='file prefix for pickles')
    parser.add_argument('--max-num-backprops', type=int, default=None, metavar='N',
                        help='how many images to backprop total')

    parser.add_argument('--sampling-strategy', default="recenter", metavar='N',
                        help='Selective backprop sampling strategy among {recenter, translate, nosquare, square}')
    parser.add_argument('--sampling-min', type=float, default=0.05,
                        help='Minimum sampling rate for sampling strategy')

    parser.add_argument('--top-k', type=int, default=8, metavar='N',
                        help='how many images to backprop per batch')
    parser.add_argument('--pool-size', type=int, default=16, metavar='N',
                        help='how many images to backprop per batch')


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

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint_dir = os.path.join(args.pickle_dir, "checkpoint")
        checkpoint_file = os.path.join(checkpoint_dir, args.pickle_prefix + "_ckpt.t7")
        assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
        print("Loading checkpoint at {}".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

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


    ## Setup Trainer ##
    square = args.sampling_strategy in ["square", "translate", "recenter"]
    translate = args.sampling_strategy in ["translate", "recenter"]
    recenter = args.sampling_strategy == "recenter"

    probability_calculator = SelectProbabiltyCalculator(args.sampling_min,
                                                        len(classes),
                                                        device,
                                                        square=square,
                                                        translate=translate)
    selector = Selector(args.batch_size, probability_calculator)
    backpropper = Backpropper(device, net, optimizer, recenter=recenter)
    trainer = Trainer(device,
                      net,
                      selector,
                      backpropper,
                      args.batch_size,
                      max_num_backprops=args.max_num_backprops)
    logger = Logger(log_interval = args.log_interval)
    image_id_hist_logger = ImageIdHistLogger(args.pickle_dir,
                                             args.pickle_prefix,
                                             len(trainset))
    trainer.on_forward_pass(logger.handle_forward_batch)
    trainer.on_backward_pass(logger.handle_backward_batch)
    trainer.on_backward_pass(image_id_hist_logger.handle_backward_batch)
    stopped = False

    for epoch in range(start_epoch, start_epoch+500):

        if stopped: break

        for partition in partitions[0:1]:
            trainloader = torch.utils.data.DataLoader(partition, batch_size=args.batch_size, shuffle=True, num_workers=2)
            test(args, net, testloader, device, epoch, state)

            # Stop training rightaway if we've exceeded maximum number of epochs
            if args.max_num_backprops:
                if args.max_num_backprops <= state.num_images_backpropped:
                    return

            if args.sb_strategy == "topk" and epoch >= args.sb_start_epoch:
                # TODO: Use Trainer
                old_trainer = train_topk
            elif args.sb_strategy == "sampling" and epoch >= args.sb_start_epoch:
                trainer.train(trainloader)
                logger.next_partition()
                if trainer.stopped():
                    stopped = True
                    break
                continue
            elif args.sb_strategy == "baseline" or epoch < args.sb_start_epoch:
                # TODO: Use Trainer
                old_trainer = train_baseline
            else:
                print("Unknown selective backprop strategy {}".format(args.sb_strategy))
                exit()

            old_trainer(args,
                    net,
                    trainloader,
                    device,
                    optimizer,
                    epoch,
                    state)
        logger.next_epoch()
        image_id_hist_logger.next_epoch()

        # Write out summary statistics
        state.write_summaries()


if __name__ == '__main__':
    main()
