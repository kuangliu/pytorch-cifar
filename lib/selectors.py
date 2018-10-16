import numpy as np
import torch
import torch.nn as nn

class PrimedSelector(object):
    def __init__(self, initial, final, initial_epochs):
        self.epoch = 0
        self.initial = initial
        self.final = final
        self.initial_epochs = initial_epochs

    def next_epoch(self):
        self.epoch += 1

    def get_selector(self):
        return self.initial if self.epoch < self.initial_epochs else self.final

    def select(self, *args, **kwargs):
        return self.get_selector().select(*args, **kwargs)

    def mark(self, *args, **kwargs):
        return self.get_selector().mark(*args, **kwargs)


class SamplingSelector(object):
    def __init__(self, probability_calcultor):
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


class DeterministicSamplingSelector(object):
    def __init__(self, probability_calcultor, initial_sum=0):
        self.global_select_sums = {}
        self.get_select_probability = probability_calcultor.get_probability
        self.initial_sum = initial_sum

    def increase_select_sum(self, example):
        select_probability = example.select_probability
        image_id = example.image_id.item()
        if image_id not in self.global_select_sums.keys():
            self.global_select_sums[image_id] = self.initial_sum
        self.global_select_sums[image_id] += select_probability.item()

    def decrease_select_sum(self, example):
        image_id = example.image_id.item()
        self.global_select_sums[image_id] -= 1
        assert(self.global_select_sums[image_id] >= 0)

    def select(self, example):
        image_id = example.image_id.item()
        return self.global_select_sums[image_id] >= 1

    def mark(self, forward_pass_batch):
        for example in forward_pass_batch:
            example.select_probability = self.get_select_probability(
                    example.target,
                    example.softmax_output)
            self.increase_select_sum(example)
            example.select = self.select(example)
            if example.select:
                self.decrease_select_sum(example)
        return forward_pass_batch


class BaselineSelector(object):

    def select(self, example):
        return True

    def mark(self, forward_pass_batch):
        for example in forward_pass_batch:
            example.select_probability = torch.tensor([[1]])
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


