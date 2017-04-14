# This implementation is based on the DenseNet implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import math
import torch
from torch import nn
from torchvision.models.resnet import conv3x3


class BasicBlockWithDeathRate(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, death_rate=0.,
                 downsample=None):
        super(BasicBlockWithDeathRate, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.death_rate = death_rate

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            x = self.downsample(x)
        # TODO: fix the bug of original Stochatic depth
        if not self.training or torch.rand(1)[0] >= self.death_rate:
            residual = self.conv1(residual)
            residual = self.bn1(residual)
            residual = self.relu1(residual)
            residual = self.conv2(residual)
            residual = self.bn2(residual)
            if self.training:
                residual /= (1. - self.death_rate)
            x = x + residual
            x = self.relu2(x)

        return x


# class DropLayer(nn.Module):
#     '''Drop the layer with probability p.
#     It can be used for stochasitc depth'''

#     def __init__(self, layer, death_rate=0.5):
#         super(DropLayer, self).__init__()
#         self.layer = layer
#         self.death_rate = death_rate

#     def forward(self, x):
#         print(self.layer)
#         if not self.training or torch.rand(1)[0] >= self.death_rate:
#             print('pass')
#             return self.layer(x)
#         else:
#             print('stop')
#             return x.div_(1 - self.death_rate)

#     def __str__(self):
#         return 'DropLayer(death_rate={})'.format(self.death_rate)


class DownsampleB(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)


class ResNetCifar(nn.Module):
    '''Small ResNet for CIFAR & SVHN
    death_rates: death_rates of each block except for the first and
                 the last block
    '''

    def __init__(self, depth, death_rates=None, block=BasicBlockWithDeathRate,
                 num_classes=10):
        assert (depth - 2) % 6 == 0, 'depth should be one of 6N+2'
        super(ResNetCifar, self).__init__()
        n = (depth - 2) // 6
        assert death_rates is None or len(death_rates) == 3 * n
        if death_rates is None:
            death_rates = [0.] * (3 * n)
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, death_rates[:n])
        self.layer2 = self._make_layer(block, 32, death_rates[n:2 * n],
                                       stride=2)
        self.layer3 = self._make_layer(block, 64, death_rates[2 * n:],
                                       stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, death_rates, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleB(self.inplanes, planes * block.expansion,
                                     stride)
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(planes * block.expansion),
            # )

        layers = [block(self.inplanes, planes, stride, downsample=downsample,
                        death_rate=death_rates[0])]
        self.inplanes = planes * block.expansion
        for death_rate in death_rates[1:]:
            layers.append(block(self.inplanes, planes, death_rate=death_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def createModel(depth, data, num_classes, death_mode='none', death_rate=0.5,
                **kwargs):
    assert (depth - 2) % 6 == 0, 'depth should be one of 6N+2'
    print('Create ResNet-{:d} for {}'.format(depth, data))
    nblocks = (depth - 2) // 2
    if death_mode == 'uniform':
        death_rates = [death_rate] * nblocks
    elif death_mode == 'linear':
        death_rates = [float(i + 1) * death_rate / float(nblocks)
                       for i in range(nblocks)]
    else:
        death_rates = None
    return ResNetCifar(depth, death_rates, BasicBlockWithDeathRate,
                       num_classes)