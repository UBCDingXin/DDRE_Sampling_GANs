'''

based on "Deep density ratio estimation for change point detection"

'''
import torch
import torch.nn as nn

IMG_SIZE=32
NC=3

class DR_6layersCNN(nn.Module):
    def __init__(self, ngpu=1):
        super(DR_6layersCNN, self).__init__()
        self.ngpu = ngpu
        self.conv = nn.Sequential(
                nn.Conv2d(NC, 60, kernel_size=3, stride=1, padding=0), #state size: (32-2-1)/1+1=30
                nn.MaxPool2d(kernel_size=2, stride=1), #state size: (30-2)/1+1=29
                #nn.ReLU(),
                nn.Conv2d(60, 50, kernel_size=3, stride=1, padding=0), #state size: (29-2-1)/1+1=27
                #nn.ReLU(),
                nn.Conv2d(50, 40, kernel_size=3, stride=1, padding=0), #state size: (27-2-1)/1+1=25
                nn.MaxPool2d(kernel_size=2, stride=1), #state size: (25-2)/1+1=24
                #nn.ReLU(),
                nn.Conv2d(40, 20, kernel_size=3, stride=1, padding=0), #state size: (24-2-1)/1+1=22
                nn.MaxPool2d(kernel_size=2, stride=1), #state size: (22-2)/1+1=21
                #nn.ReLU(),
                nn.Conv2d(20, 10, kernel_size=2, stride=1, padding=0), #state size: (21-1-1)/1+1=20
                #nn.ReLU(),
                nn.Conv2d(10, 5, kernel_size=2, stride=1, padding=0), #state size: (20-1-1)/1+1=19
                #nn.ReLU(),
        )
        linear = [
                    nn.Linear(19*19*5, 250),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(250, 1),
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
   net = DR_6layersCNN(ngpu=2).cuda()
   x = torch.randn(256,3,32,32).cuda()
   out = net(x)
   print(out.size())
