'''
ResNet-based density ratio model for estimation in pixel space

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

class DR_ResNet(nn.Module):
    def __init__(self, block, num_blocks, ngpu=2, p_dropout=0.4, fc_layers=False, final_ActFn="Softplus"):
        super(DR_ResNet, self).__init__()
        self.in_planes = 64
        self.ngpu = ngpu
        self.p_dropout = p_dropout

        convs = [nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU()]
        convs += self._make_layer(block, 64, num_blocks[0], stride=1)
        convs += self._make_layer(block, 128, num_blocks[1], stride=2)
        convs += self._make_layer(block, 256, num_blocks[2], stride=2)
        convs += self._make_layer(block, 512, num_blocks[3], stride=2)
        convs += [nn.AvgPool2d(kernel_size=4)]
        self.convs = nn.Sequential(*convs)

        if fc_layers:
            linear_layers = [
                    nn.Linear(512*block.expansion, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(self.p_dropout),

                    nn.Linear(256,128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(self.p_dropout),

                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(self.p_dropout),

                    nn.Linear(64, 1)
            ]

        else:
            linear_layers = [
                    nn.Linear(512*block.expansion, 1)
            ]

        if final_ActFn == "ReLU":
            linear_layers += [nn.ReLU()]
        elif final_ActFn == "Softplus":
            linear_layers += [nn.Softplus()]
        self.linear = nn.Sequential(*linear_layers)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return layers

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            features = nn.parallel.data_parallel(self.convs, x, range(self.ngpu))
            features = features.view(features.size(0), -1)
            out = nn.parallel.data_parallel(self.linear, features, range(self.ngpu))
        else:
            features = self.convs(x)
            features = features.view(features.size(0), -1)
            out = self.linear(features)
        return out


def DR_ResNet18(ngpu=1, p_dropout=0.4, fc_layers=False, final_ActFn="Softplus"):
    return DR_ResNet(BasicBlock, [2,2,2,2], ngpu=ngpu, p_dropout=p_dropout, fc_layers=fc_layers, final_ActFn=final_ActFn)

def DR_ResNet34(ngpu=1, p_dropout=0.4, fc_layers=False, final_ActFn="Softplus"):
    return DR_ResNet(BasicBlock, [3,4,6,3], ngpu=ngpu, p_dropout=p_dropout, fc_layers=fc_layers, final_ActFn=final_ActFn)

def DR_ResNet50(ngpu=1, p_dropout=0.4, fc_layers=False, final_ActFn="Softplus"):
    return DR_ResNet(Bottleneck, [3,4,6,3], ngpu=ngpu, p_dropout=p_dropout, fc_layers=fc_layers, final_ActFn=final_ActFn)

def DR_ResNet101(ngpu=1, p_dropout=0.4, fc_layers=False, final_ActFn="Softplus"):
    return DR_ResNet(Bottleneck, [3,4,23,3], ngpu=ngpu, p_dropout=p_dropout, fc_layers=fc_layers, final_ActFn=final_ActFn)

def DR_ResNet152(ngpu=1, p_dropout=0.4, fc_layers=False, final_ActFn="Softplus"):
    return DR_ResNet(Bottleneck, [3,8,36,3], ngpu=ngpu, p_dropout=p_dropout, fc_layers=fc_layers, final_ActFn=final_ActFn)


if __name__=="__main__":
   net = DR_ResNet50(ngpu=2, p_dropout=0.4, final_ActFn="Softplus").cuda()
   x = torch.randn(256,3,32,32).cuda()
   out = net(x)
   print(out.size())
