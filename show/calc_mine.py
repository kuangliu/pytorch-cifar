# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms

import tkinter.filedialog
import matplotlib.pyplot as plt
from PIL import Image

from load import loadnet


pic_num = 5
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
std = (0.2023, 0.1994, 0.2010)
mean = (0.4914, 0.4822, 0.4465)

fname = tkinter.filedialog.askopenfilename()

if __name__ == '__main__':
    # 原因: 如 https://blog.csdn.net/xiemanR/article/details/71700531
    # test loader涉及多线程操作, 在windows环境下需要用__name__ == '__main__'包装
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    img = Image.open(fname).convert('RGB')
    plt.imshow(img)
    plt.show()
    img = img.resize((32, 32), Image.ANTIALIAS)
    plt.imshow(img)
    plt.show()
    inputs = transform(img).reshape(1, 3, 32, 32)

    net, _ = loadnet(1)

    net.eval()  # 变为测试模式, 对dropout和batch normalization有影响
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        net.cuda()
    to_pil_image = transforms.ToPILImage()
    with torch.no_grad():  # 运算不需要进行求导, 提高性能
        inputs = inputs.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        print("计算类别: %s" % (classes[predicted[0]]))  # 显示label