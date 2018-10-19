import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class CIFAR10:
    def __init__(self, model, test_batch_size, partition_size):

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.model = model

        # Testing set
        transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

        # Training set
        transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
        self.trainset = [t + (i,) for i, t in enumerate(self.trainset)]
        self.partitions = [self.trainset[i:i + partition_size] for i in xrange(0, len(self.trainset), partition_size)]

        self.num_training_images = len(self.trainset)

class MNIST:
    def __init__(self, device, test_batch_size, partition_size):

        self.model = MNISTNet().to(device)

        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

        # Testing set
        self.testloader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('../data', train=False, download=True,
                           transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size=test_batch_size, shuffle=False, num_workers=2)

        # Training set

        self.trainset = torchvision.datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        self.trainset = [t + (i,) for i, t in enumerate(self.trainset)]
        self.partitions = [self.trainset[i:i + partition_size] for i in xrange(0, len(self.trainset), partition_size)]

        self.num_training_images = len(self.trainset)

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
