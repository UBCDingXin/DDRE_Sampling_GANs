'''

Density Ration Approximation via Multilayer Perceptron

Multilayer Perceptron : trained to model density ratio in a feature space; based on "Rectified Linear Units Improve Restricted Boltzmann Machines"

Its input is the output of a pretrained Deep CNN, say ResNet-34

'''

import torch
import torch.nn as nn



# cfg = {"CNN5": [2048,1024,512,256,128]}

IMG_SIZE = 64
NC = 3

class DR_CNN(nn.Module):
    def __init__(self, CNN_name, ngpu=1, final_ActFn="ReLU", p_dropout=0.4, init_in_dim = IMG_SIZE*IMG_SIZE*NC):
        super(DR_CNN, self).__init__()
        self.ngpu = ngpu
        self.init_in_dim = init_in_dim
        self.p_dropout=p_dropout

        self.fc1 = nn.Sequential(
                    nn.Linear(self.init_in_dim, 4096),
                    nn.GroupNorm(8, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.p_dropout)
                )

        if CNN_name == "CNN5":
            self.convs = nn.Sequential(
                        # conv1
                        nn.Conv2d(2, 4, kernel_size=4, stride=2, padding=1), #h=h/2; 4*16*16=1024
                        nn.GroupNorm(2, 4),
                        nn.ReLU(),
                        # conv2
                        nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1), #h=h/2; 8*8*8=512
                        nn.GroupNorm(2, 8),
                        nn.ReLU(),
                        # conv3
                        nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1), #h=h/2; 16*4*4=256
                        nn.GroupNorm(2, 16),
                        nn.ReLU(),
                        # conv4
                        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), #h=h/2; 32*2*2=128
                        nn.GroupNorm(2, 32),
                        nn.ReLU(),
                    )


        self.fc2 = [nn.Linear(128, 1)]
        if final_ActFn == "ReLU":
            self.fc2 += [nn.ReLU()]
        elif final_ActFn == "Softplus":
            self.fc2 += [nn.Softplus()]
        self.fc2 = nn.Sequential(*self.fc2)

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            out = nn.parallel.data_parallel(self.fc1, x, range(self.ngpu))
            out = out.view(out.size(0),2,32,32) # 32*32*2=2048
            out = nn.parallel.data_parallel(self.convs, out, range(self.ngpu))
            out = out.view(out.size(0),-1)
            out = nn.parallel.data_parallel(self.fc2, out, range(self.ngpu))
        else:
            out = self.fc1(x)
            out = out.view(out.size(0),2,32,32) # 32*32*2=2048
            out = self.convs(out)
            out = out.view(out.size(0),-1)
            out = self.fc2(out)
        return out


if __name__ == "__main__":
    init_in_dim = 2
    net = DR_CNN('CNN5', init_in_dim = init_in_dim).cuda()
    x = torch.randn((55,init_in_dim)).cuda()
    out = net(x)
    print(out.size())
