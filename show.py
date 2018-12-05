# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms

from utils import progress_bar
from load import loadnet


if __name__ == '__main__':
    # 原因: 如 https://blog.csdn.net/xiemanR/article/details/71700531
    # test loader涉及多线程操作, 在windows环境下需要用__name__ == '__main__'包装
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    [net, acc] = loadnet(1)
    print ('Expected accuracy: %f%%' % acc)

    net.eval()  # 变为测试模式, 对dropout和batch normalization有影响
    correct = 0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():  # 运算不需要进行求导, 提高性能
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            print (inputs.size())
            break
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(batch_idx, len(testloader))
            # progress_bar(batch_idx, len(testloader),
            #     'Acc: %.3f%% (%d/%d)' % (correct / total * 100, correct, total))
    print ('Caculated accuracy: %f%%' % (float(correct) / total))
