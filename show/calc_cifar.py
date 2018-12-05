# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from load import loadnet


pic_num = 5
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
std = (0.2023, 0.1994, 0.2010)
mean = (0.4914, 0.4822, 0.4465)

if __name__ == '__main__':
    # 原因: 如 https://blog.csdn.net/xiemanR/article/details/71700531
    # test loader涉及多线程操作, 在windows环境下需要用__name__ == '__main__'包装
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=pic_num, shuffle=True, num_workers=2)

    net, _ = loadnet(1)

    net.eval()  # 变为测试模式, 对dropout和batch normalization有影响
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    to_pil_image = transforms.ToPILImage()
    with torch.no_grad():  # 运算不需要进行求导, 提高性能
        (inputs, targets) = list(testloader)[0]
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        for i in range(pic_num):
            print("正确类别: %s, 计算类别: %s" % (
                classes[targets[i]], classes[predicted[i]]))  # 显示label
            img = inputs[i].new(*inputs[i].size())
            img[0, :, :] = inputs[i][0, :, :] * std[0] + mean[0]
            img[1, :, :] = inputs[i][1, :, :] * std[1] + mean[1]
            img[2, :, :] = inputs[i][2, :, :] * std[2] + mean[2]
            img = to_pil_image(img)
            plt.imshow(img)
            plt.show()