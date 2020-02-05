'''

based on the DCGAN structure of MH-GAN: https://github.com/uber-research/metropolis-hastings-gans/blob/master/mhgan/contrib/dcgan/dcgan.py

'''
from itertools import count
# We could use sum([...], []) instead of concat to avoid np import here
from numpy import concatenate as concat
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils


class cnn_generator(nn.Module):
    def __init__(self, ngpu=1, nz=128, ngf=64, nc=3):
        super(cnn_generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)
        return output


class cnn_discriminator(nn.Module):
    def __init__(self, use_sigmoid = True, ngpu=1, ndf=64, nc=3):
        super(cnn_discriminator, self).__init__()
        self.ngpu = ngpu

        self.main = [
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        ]
        if use_sigmoid:
            self.main += [nn.Sigmoid()]

        self.main = nn.Sequential(*self.main)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)





if __name__=="__main__":
    #test
    ngpu=1

    netG = cnn_generator(ngpu=ngpu, nz=128, ngf=64, nc=3).cuda()
    netD = cnn_discriminator(use_sigmoid = True, ngpu=ngpu, nc=3, ndf=64).cuda()

    z = torch.randn(5, 128,1,1).cuda()
    x = netG(z)
    o = netD(x)
    print(x.size())
    print(o.size())
