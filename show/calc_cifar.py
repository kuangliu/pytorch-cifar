# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from load import loadnet


pic_num = 5
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    # 原因: 如 https://blog.csdn.net/xiemanR/article/details/71700531
    # test loader涉及多线程操作, 在windows环境下需要用__name__ == '__main__'包装
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=pic_num, shuffle=True, num_workers=2)

    net, _ = loadnet(1)

    net.eval()  # 变为测试模式, 对dropout和batch normalization有影响
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():  # 运算不需要进行求导, 提高性能
        (inputs, targets) = list(testloader)[0]
        for i in range(pic_num):
            print(classes[targets[i]])  # 显示label
            img = inputs[i].numpy()  # FloatTensor转为ndarray
            img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
            plt.imshow(img)
            plt.show()