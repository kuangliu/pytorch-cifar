import torch
import torch.nn as nn

class PrimedBackpropper(object):
    def __init__(self, initial, final, initial_epochs):
        self.epoch = 0
        self.initial = initial
        self.final = final
        self.initial_epochs = initial_epochs

    def next_epoch(self):
        self.epoch += 1

    def get_backpropper(self):
        return self.initial if self.epoch < self.initial_epochs else self.final

    def backward_pass(self, *args, **kwargs):
        return self.get_backpropper().backward_pass(*args, **kwargs)


class BaselineBackpropper(object):

    def __init__(self, device, net, optimizer):
        self.optimizer = optimizer
        self.net = net
        self.device = device
        self.sum_select_probabilities = 0
        self.total_num_examples = 0

    def _get_chosen_data_tensor(self, batch):
        chosen_data = [example.datum for example in batch if example.select]
        return torch.stack(chosen_data)

    def _get_chosen_targets_tensor(self, batch):
        chosen_targets = [example.target for example in batch if example.select]
        return torch.stack(chosen_targets)

    def _get_chosen_probabilities_tensor(self, batch):
        probabilities = [example.select_probability for example in batch if example.select]
        return torch.tensor(probabilities, dtype=torch.float)

    def backward_pass(self, batch):

        data = self._get_chosen_data_tensor(batch)
        targets = self._get_chosen_targets_tensor(batch)
        probabilities = self._get_chosen_probabilities_tensor(batch)

        # Run forward pass
        # Necessary if the network has been updated between last forward pass
        outputs = self.net(data) 
        losses = nn.CrossEntropyLoss(reduce=False)(outputs, targets)

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


class SamplingBackpropper(object):

    def __init__(self, device, net, optimizer, recenter=False):
        self.optimizer = optimizer
        self.net = net
        self.device = device
        self.recenter = recenter
        self.sum_select_probabilities = 0
        self.total_num_examples = 0

    def update_sum_probabilities(self, batch):
        probabilities = [example.select_probability.item() for example in batch]
        self.sum_select_probabilities += sum(probabilities)
        self.total_num_examples += len(probabilities)

    def _get_chosen_data_tensor(self, batch):
        chosen_data = [example.datum for example in batch if example.select]
        return torch.stack(chosen_data)

    def _get_chosen_targets_tensor(self, batch):
        chosen_targets = [example.target for example in batch if example.select]
        return torch.stack(chosen_targets)

    def _get_chosen_probabilities_tensor(self, batch):
        probabilities = [example.select_probability for example in batch if example.select]
        return torch.tensor(probabilities, dtype=torch.float)

    @property
    def average_select_probability(self):
        return float(self.sum_select_probabilities) / self.total_num_examples

    def backward_pass(self, batch):

        self.update_sum_probabilities(batch)

        data = self._get_chosen_data_tensor(batch)
        targets = self._get_chosen_targets_tensor(batch)
        probabilities = self._get_chosen_probabilities_tensor(batch)

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


