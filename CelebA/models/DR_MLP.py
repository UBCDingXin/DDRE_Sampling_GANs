'''

Density Ration Approximation via Multilayer Perceptron

Multilayer Perceptron : trained to model density ratio in a feature space

Its input is the output of a pretrained Deep CNN, say ResNet-34

'''

import torch
import torch.nn as nn

IMG_SIZE = 64
NC = 3


cfg = {"MLP3": [2048,1024,512],
       "MLP5": [2048,1024,512,256,128],
       "MLP7": [2048,2048,1024,1024,512,256,128],
       "MLP9": [2048,2048,1024,1024,512,512,256,256,128],}

# cfg = {"MLP5": [4096,2048,1024,512,256]}


class DR_MLP(nn.Module):
    def __init__(self, MLP_name, ngpu=1, final_ActFn="ReLU", p_dropout=0.4, init_in_dim = IMG_SIZE*IMG_SIZE*NC):
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
                       nn.Dropout(self.p_dropout) # do we really need dropout?
                       ]
            in_dim = x
        return layers

    def forward(self, x):
        out = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        return out


if __name__ == "__main__":
    net = DR_MLP('MLP9').cuda()
    x = torch.randn((5,IMG_SIZE*IMG_SIZE*NC)).cuda()
    out = net(x)
    print(out.size())
