import torch
import torch.nn as nn

#########################################################
# genearator
class generator(nn.Module):
    def __init__(self, ngpu=1, nz=2, out_dim=2, high_dim = False):
        super(generator, self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        self.out_dim = out_dim

        if high_dim:
            self.linear = nn.Sequential(
                    nn.Linear(nz, 128), #layer 1
                    nn.BatchNorm1d(128),
                    nn.ReLU(True),
                    nn.Linear(128, 256), #layer 2
                    nn.BatchNorm1d(256),
                    nn.ReLU(True),
                    nn.Linear(256, 512), #layer 3
                    nn.BatchNorm1d(512),
                    nn.ReLU(True),
                    nn.Linear(512,256), #layer 4
                    nn.BatchNorm1d(256),
                    nn.ReLU(True),
                    nn.Linear(256,128), #layer 5
                    nn.BatchNorm1d(128),
                    nn.ReLU(True),
                    nn.Linear(128, self.out_dim), #layer 6
            )


        else:
            inner_dim = 100
            self.linear = nn.Sequential(
                    nn.Linear(nz, inner_dim), #layer 1
                    nn.ReLU(True),
                    nn.Linear(inner_dim, inner_dim), #layer 2
                    nn.ReLU(True),
                    nn.Linear(inner_dim, inner_dim), #layer 3
                    nn.ReLU(True),
                    nn.Linear(inner_dim, self.out_dim), #layer 4
            )

    def forward(self, input):
        input = input.view(-1, self.nz)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.linear, input, range(self.ngpu))
        else:
            output = self.linear(input)
        return output

#########################################################
# discriminator
class discriminator(nn.Module):
    def __init__(self, ngpu=1, input_dim = 2):
        super(discriminator, self).__init__()
        self.ngpu = ngpu
        self.input_dim = input_dim

        inner_dim = 100
        self.main = nn.Sequential(
            nn.Linear(input_dim, inner_dim), #Layer 1
            nn.ReLU(True),
            nn.Linear(inner_dim, inner_dim), #layer 2
            nn.ReLU(True),
            nn.Linear(inner_dim, inner_dim), #layer 3
            nn.ReLU(True),
            nn.Linear(inner_dim, 1), #layer 4
            nn.Sigmoid()
        )


    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1)


if __name__=="__main__":
    #test
    ngpu=1

    netG = generator(ngpu=ngpu, nz=2, out_dim=2).cuda()
    netD = discriminator(ngpu=ngpu, input_dim = 2).cuda()

    z = torch.randn(128, 2).cuda()
    x = netG(z)
    o = netD(x)
    print(x.size())
    print(o.size())
