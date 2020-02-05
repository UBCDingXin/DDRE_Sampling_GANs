"""
codes for MMD-GAN

based on "MMD-GAN: Towards Deeper Understanding of Moment Matching Network"

based on: https://github.com/OctoberChang/MMD-GAN

"""

import torch
import torch.nn as nn



###########################################################################
# base_module
###########################################################################

# input: batch_size * nc * 64 * 64
# output: batch_size * k * 1 * 1
class MMDGAN_Encoder(nn.Module):
    def __init__(self, isize, nc, k=100, ndf=64, ngpu = 2):
        super(MMDGAN_Encoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        
        self.ngpu = ngpu
        
        # input is nc x isize x isize
        main = [nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)]
        
        csize, cndf = isize / 2, ndf

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main += [nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False),
                     nn.BatchNorm2d(out_feat),
                     nn.LeakyReLU(0.2, inplace=True)
                    ]
            cndf = cndf * 2
            csize = csize / 2
        
        main += [nn.Conv2d(cndf, k, 4, 1, 0, bias=False)]
        
        self.main = nn.Sequential(*main)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# input: batch_size * k * 1 * 1
# output: batch_size * nc * image_size * image_size
class MMDGAN_Decoder(nn.Module):
    def __init__(self, isize, nc, k=100, ngf=64, ngpu = 2):
        super(MMDGAN_Decoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        
        self.ngpu = ngpu
        
        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2
        
        main = [nn.ConvTranspose2d(k, cngf, 4, 1, 0, bias=False),
                nn.BatchNorm2d(cngf),
                nn.ReLU(True)]
        
        csize = 4
        while csize < isize // 2:
            main += [nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False),
                     nn.BatchNorm2d(cngf // 2),
                     nn.ReLU(True)]
            cngf = cngf // 2
            csize = csize * 2
        
        main += [nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False),
                 nn.Tanh()]

        self.main = nn.Sequential(*main)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


def MMDGAN_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)


###########################################################################
# Generator and Discriminator
###########################################################################
# NetG is a decoder
# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
class MMDGAN_G(nn.Module):
    def __init__(self, decoder):
        super(MMDGAN_G, self).__init__()
        self.decoder = decoder

    def forward(self, input):
        output = self.decoder(input)
        return output

# NetD is an encoder + decoder
# input: batch_size * nc * image_size * image_size
# f_enc_X: batch_size * k * 1 * 1
# f_dec_X: batch_size * nc * image_size * image_size
class MMDGAN_D(nn.Module):
    def __init__(self, encoder, decoder):
        super(MMDGAN_D, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        f_enc_X = self.encoder(input)
        f_dec_X = self.decoder(f_enc_X)

        f_enc_X = f_enc_X.view(input.size(0), -1)
        f_dec_X = f_dec_X.view(input.size(0), -1)
        return f_enc_X, f_dec_X


class MMDGAN_ONE_SIDED(nn.Module):
    def __init__(self):
        super(MMDGAN_ONE_SIDED, self).__init__()

        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean()
        return output


