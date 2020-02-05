'''

Density Ration Approximation via Multilayer Perceptron

Multilayer Perceptron : trained to model density ratio in a feature space; based on "Rectified Linear Units Improve Restricted Boltzmann Machines"

Its input is the output of a pretrained Deep CNN, say ResNet-34

'''

import torch
import torch.nn as nn

cfg = {"MLP5": [2048,1024,512,256,128]}
# cfg = {"MLP5": [500,500,500,500,500]}
# cfg = {"MLP5": [128,128,128,128,128]}
# cfg = {"MLP5": [16,32,64,32,16]}

class DR_MLP(nn.Module):
    def __init__(self, MLP_name, init_in_dim = 2, ngpu=1, final_ActFn="ReLU", p_dropout=0.2):
        super(DR_MLP, self).__init__()
        self.ngpu = ngpu
        self.init_in_dim = init_in_dim
        self.p_dropout=p_dropout
        layers = self._make_layers(cfg[MLP_name])
        layers += [nn.Linear(cfg[MLP_name][-1], 1)]
        if final_ActFn == "ReLU":
            layers += [nn.ReLU()]
        elif final_ActFn == "Softplus":
            layers += [nn.Softplus()]

        self.main = nn.Sequential(*layers)

    def _make_layers(self, cfg):
        layers = []
        in_dim = self.init_in_dim #initial input dimension
        for x in cfg:
            layers += [nn.Linear(in_dim, x),
                       # nn.BatchNorm1d(x),
                       nn.GroupNorm(4, x),
                       nn.ReLU(inplace=True),
                       nn.Dropout(self.p_dropout)
                       ]
            in_dim = x
        return layers

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            out = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            out = self.main(x)
        return out


if __name__ == "__main__":
    init_in_dim = 100
    net = DR_MLP('MLP5', init_in_dim = init_in_dim).cuda()
    x = torch.randn((128,init_in_dim)).cuda()
    out = net(x)
    print(out.size())
