'''
ResNet-based model to map an image from pixel space to a features space.
Need to be pretrained on the dataset.

if isometric_map = True, there is an extra step (elf.classifier_1 = nn.Linear(512, 32*32*3)) to increase the dimension of the feature map from 512 to 32*32*3. This selection is for desity-ratio estimation in feature space.

codes are based on
@article{
zhang2018mixup,
title={mixup: Beyond Empirical Risk Minimization},
author={Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz},
journal={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=r1Ddp1-Rb},
}
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, isometric_map = True, num_classes=10, nc=3, ngpu = 1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.isometric_map = isometric_map
        self.ngpu = ngpu

        self.main = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False),  # h=h
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_layer(block, 64, num_blocks[0], stride=1),  # h=h
            self._make_layer(block, 128, num_blocks[1], stride=2), 
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block, 512, num_blocks[3], stride=2),
            nn.AvgPool2d(kernel_size=4)
        )
        self.classifier = nn.Linear(512*block.expansion, num_classes)
        self.classifier_1 = nn.Sequential(
                nn.Linear(512*block.expansion, 32*32*3),
                #nn.ReLU()
                #nn.Tanh() #exist in the first version
                )
        self.classifier_2 = nn.Linear(32*32*3, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        if x.is_cuda and self.ngpu > 1:
            features = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
            features = features.view(features.size(0), -1)
            if self.isometric_map:
                features = nn.parallel.data_parallel(self.classifier_1, features, range(self.ngpu))
                out = nn.parallel.data_parallel(self.classifier_2, features, range(self.ngpu))
            else:
                out = nn.parallel.data_parallel(self.classifier, features, range(self.ngpu))
        else:
            features = self.main(x)
            features = features.view(features.size(0), -1)
            if self.isometric_map:
                features = self.classifier_1(features)
                out = self.classifier_2(features)
            else:
                out = self.classifier(features)
        return out, features


def ResNet18(isometric_map = False, num_classes=10, ngpu = 1):
    return ResNet(BasicBlock, [2,2,2,2], isometric_map, num_classes=num_classes, ngpu = ngpu)

def ResNet34(isometric_map = False, num_classes=10, ngpu = 1):
    return ResNet(BasicBlock, [3,4,6,3], isometric_map, num_classes=num_classes, ngpu = ngpu)

def ResNet50(isometric_map = False, num_classes=10, ngpu = 1):
    return ResNet(Bottleneck, [3,4,6,3], isometric_map, num_classes=num_classes, ngpu = ngpu)

def ResNet101(isometric_map = False, num_classes=10, ngpu = 1):
    return ResNet(Bottleneck, [3,4,23,3], isometric_map, num_classes=num_classes, ngpu = ngpu)

def ResNet152(isometric_map = False, num_classes=10, ngpu = 1):
    return ResNet(Bottleneck, [3,8,36,3], isometric_map, num_classes=num_classes, ngpu = ngpu)


if __name__ == "__main__":
    net = ResNet50(isometric_map = True, num_classes=10, ngpu = 2).cuda()
    x = torch.randn(256,3,32,32).cuda()
    out, features = net(x)
    print(out.size())
    print(features.size())
