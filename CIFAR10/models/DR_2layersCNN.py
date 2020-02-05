'''

based on "Direct Density Ratio Estimation with Convolutional Neural Networks"

'''
import torch
import torch.nn as nn

IMG_SIZE=32
NC=3

class DR_2layersCNN(nn.Module):
    def __init__(self, ngpu=1):
        super(DR_2layersCNN, self).__init__()
        self.ngpu = ngpu
        self.conv = nn.Sequential(
                nn.Conv2d(NC, 6, kernel_size=9, stride=1, padding=0), #state size: 24
                nn.AvgPool2d(kernel_size=2, stride=2), #state size: 12
                nn.Sigmoid(),
                nn.Conv2d(6, 12, kernel_size=5, stride=2, padding=0), #state size: 4
                nn.AvgPool2d(kernel_size=2, stride=2), #state size: 2
                nn.Sigmoid(),
        )
        linear = [
                    nn.Linear(12*2*2, 1),
                    nn.ReLU()
                ]

        self.linear = nn.Sequential(*linear)

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            out = nn.parallel.data_parallel(self.conv, x, range(self.ngpu))
            out = out.view(out.size(0),-1)
            out = nn.parallel.data_parallel(self.linear, out, range(self.ngpu))
        else:
            out = self.conv(x)
            out = out.view(out.size(0),-1)
            out = self.linear(out)
        return out

if __name__=="__main__":
   net = DR_2layersCNN(ngpu=1).cuda()
   x = torch.randn(256,3,32,32).cuda()
   out = net(x)
   print(out.size())
