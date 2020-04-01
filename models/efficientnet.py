'''EfficientNet in PyTorch.

Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".

Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * x.sigmoid()


class Block(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self, in_planes, out_planes, kernel_size, stride, expand_ratio=1, se_ratio=0., drop_rate=0.):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate

        # Expansion
        planes = expand_ratio * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Depthwise conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size,
                               stride=stride, padding=(kernel_size-1)//2, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # SE layers
        se_planes = max(1, int(planes * se_ratio))
        self.se1 = nn.Conv2d(planes, se_planes, kernel_size=1)
        self.se2 = nn.Conv2d(se_planes, planes, kernel_size=1)

        # Output
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )


    def forward(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        # Squeeze-Excitation
        w = F.avg_pool2d(out, out.size(2))
        w = swish(self.se1(w))
        w = self.se2(w).sigmoid()
        out = out * w
        # Output
        out = self.bn3(self.conv3(out))
        if self.drop_rate > 0:
            out = F.dropout2d(out, self.drop_rate)
        shortcut = self.shortcut(x) if self.stride == 1 else out
        out = out + shortcut
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(cfg[-1][1], num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, kernel_size, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, kernel_size, stride, expansion, se_ratio=0.25, drop_rate=0))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def EfficientNetB0():
    # (expansion, out_planes, num_blocks, kernel_size, stride)
    cfg = [(1,  16, 1, 3, 1),
           (6,  24, 2, 3, 2),
           (6,  40, 2, 5, 2),
           (6,  80, 3, 3, 2),
           (6, 112, 3, 5, 1),
           (6, 192, 4, 5, 2),
           (6, 320, 1, 3, 1)]
    return EfficientNet(cfg)


def test():
    net = EfficientNetB0()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.shape)


if __name__ == '__main__':
    test()
