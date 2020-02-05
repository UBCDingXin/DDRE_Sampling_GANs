'''
https://github.com/christiancosgrove/pytorch-spectral-normalization-gan

chainer: https://github.com/pfnet-research/sngan_projection
'''

# ResNet generator and discriminator
import torch
from torch import nn
import torch.nn.functional as F

# from spectral_normalization import SpectralNorm
import numpy as np
from torch.nn.utils import spectral_norm


# device_ids=[1,0]

channels = 3

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0) #h=h
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        self.bypass = nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.bypass_conv,
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        if stride != 1:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        else:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            spectral_norm(self.conv1),
            nn.ReLU(),
            spectral_norm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

GEN_SIZE=64
DISC_SIZE=64

class SNGAN_Generator(nn.Module):
    def __init__(self, z_dim, ngpu = 1):
        super(SNGAN_Generator, self).__init__()
        self.z_dim = z_dim
        self.ngpu = ngpu

        self.dense = nn.Linear(self.z_dim, 4 * 4 * (GEN_SIZE*16))
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResBlockGenerator((GEN_SIZE*16), (GEN_SIZE*8)), #4--->8
            ResBlockGenerator((GEN_SIZE*8), (GEN_SIZE*4)), #8--->16
            ResBlockGenerator((GEN_SIZE*4), GEN_SIZE*2), #16--->32
            ResBlockGenerator((GEN_SIZE*2), GEN_SIZE), #32--->64
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        if z.is_cuda and self.ngpu > 1:
            out = nn.parallel.data_parallel(self.dense, z, range(self.ngpu))
            out = out.view(-1, (GEN_SIZE*16), 4, 4)
            out = nn.parallel.data_parallel(self.model, out, range(self.ngpu))
        else:
            out = self.dense(z)
            out = out.view(-1, (GEN_SIZE*16), 4, 4)
            out = self.model(out)
        return out


class SNGAN_Discriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(SNGAN_Discriminator, self).__init__()
        self.ngpu = ngpu

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2), #64--->32
                ResBlockDiscriminator(DISC_SIZE  , DISC_SIZE*2, stride=2), #32--->16
                ResBlockDiscriminator(DISC_SIZE*2, DISC_SIZE*4, stride=2), #16--->8
                ResBlockDiscriminator(DISC_SIZE*4, DISC_SIZE*8, stride=2), #8--->4
                ResBlockDiscriminator(DISC_SIZE*8, DISC_SIZE*16, stride=1), #4--->4; 1024x4x4
                nn.ReLU(),
                #nn.AvgPool2d(2), #6--->3
            )
        self.fc = nn.Linear(DISC_SIZE*16, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = spectral_norm(self.fc)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            features = nn.parallel.data_parallel(self.model, x, range(self.ngpu))
            features = torch.sum(features, dim=(2, 3)) # Global pooling
            out = nn.parallel.data_parallel(self.fc, features, range(self.ngpu))
        else:
            features = self.model(x)
            features = torch.sum(features, dim=(2, 3)) # Global pooling
            out = self.fc(features)
        return out, features

class SNGAN_Aux_Classifier(nn.Module):
    """Discriminator, Auxiliary Classifier."""
    def __init__(self, ngpu = 1):
        super(SNGAN_Aux_Classifier, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        if x.is_cuda and self.ngpu>1:
            out = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            out = nn.main(x)
        return out




if __name__=="__main__":
    #test
    NGPU = 2

    netG = SNGAN_Generator(z_dim=128, ngpu = NGPU).cuda()
    netD = SNGAN_Discriminator(ngpu = NGPU).cuda()

    z = torch.randn(5, 128).cuda()
    x = netG(z)
    print(x.size())


    o,f = netD(x)
    print(o.size())
    print(f.size())
