'''
VGG-based density ratio model for estimation in pixel space

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

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

p_dropout = 0.4

class DR_VGG(nn.Module):
    def __init__(self, vgg_name, fc_layers=False, NGPU=2, final_ActFn="Softplus"):
        super(DR_VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.NGPU = NGPU
        if fc_layers:
            linear_layers = [
                    #l1
                    nn.Linear(512, 256),
                    nn.GroupNorm(4, 256),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    #l2
                    nn.Linear(256,128),
                    nn.GroupNorm(4, 128),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    #l3
                    nn.Linear(128, 64),
                    nn.GroupNorm(4, 64),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    #l4
                    nn.Linear(64, 1),
                    ]
        else:
            linear_layers = [
                    nn.Linear(512, 1)]
        if final_ActFn == "ReLU":
            linear_layers += [nn.ReLU()]
        elif final_ActFn == "Softplus":
            linear_layers += [nn.Softplus()]
        else:
            raise Exception("The final activation function must be either ReLU or Softplus!!!")
        self.linear = nn.Sequential(*linear_layers)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)] #h/2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), #h
                           #nn.BatchNorm2d(x),
                           nn.GroupNorm(4, x),
                           nn.ReLU(inplace=True)]
                in_channels = x
#        layers += [nn.AvgPool2d(kernel_size=1, stride=1)] #h
        layers += [nn.AdaptiveAvgPool2d((1,1))] #h
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.parallel.data_parallel(self.features, x, range(self.NGPU))
        out = out.view(out.size(0), -1)
        out = nn.parallel.data_parallel(self.linear, out, range(self.NGPU))
        return out

#net = DR_VGG('VGG11').cuda()
#x = torch.randn(4,3,32,32).cuda()
#print(net(Variable(x)).size())
