import torch
import torch.nn as nn

cfg = {"MLP5": [2048,1024,512,256,128]}


class PreNetDRE_MLP(nn.Module):
    def __init__(self, MLP_name="MLP5", n_class = 25, init_in_dim = 2, ngpu=1, p_dropout=0.5):
        super(PreNetDRE_MLP, self).__init__()
        self.n_class = n_class
        self.ngpu = ngpu
        self.init_in_dim = init_in_dim
        self.p_dropout=p_dropout
        layers = self._make_layers(cfg[MLP_name])
        layers += [
                    nn.Linear(cfg[MLP_name][-1], init_in_dim),
                    nn.BatchNorm1d(init_in_dim),
                    # nn.ReLU()
                ] #output dim is consistent with the input dim

        self.main = nn.Sequential(*layers)

        self.classifier = nn.Sequential(nn.Linear(init_in_dim, n_class))

    def _make_layers(self, cfg):
        layers = []
        in_dim = self.init_in_dim #initial input dimension
        for x in cfg:
            layers += [nn.Linear(in_dim, x),
                       nn.BatchNorm1d(x),
                       nn.ReLU(inplace=True),
                       nn.Dropout(self.p_dropout) # do we really need dropout?
                       ]
            in_dim = x
        return layers

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            feature = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
            out = nn.parallel.data_parallel(self.classifier, feature, range(self.ngpu))
        else:
            feature = self.main(x)
            out = self.classifier(feature)
        return out, feature


if __name__ == "__main__":
    init_in_dim = 100
    ngpu = 2
    net = PreNetDRE_MLP(init_in_dim = init_in_dim, ngpu = ngpu).cuda()
    x = torch.randn((128,init_in_dim)).cuda()
    out = net(x)
    print(out.size())
