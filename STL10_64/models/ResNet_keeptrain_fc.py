'''

Fully-connected layers on top of a pre-trained ResNet which are used to extract high-level features

'''

import torch
import torch.nn as nn


input_dim_dict = {"ResNet34": 512,
                  "ResNet50": 2048}

NC = 3
IMG_SIZE = 64


class ResNet_keeptrain_fc(nn.Module):
    def __init__(self, ResNet_name, ngpu=1, num_classes=10):
        super(ResNet_keeptrain_fc, self).__init__()
        self.ngpu = ngpu
        input_dim = input_dim_dict[ResNet_name]

        fc_layers_1 = [
                        nn.Linear(input_dim, NC*IMG_SIZE*IMG_SIZE),
                    ]

        fc_layers_2 = [
                        # nn.BatchNorm1d(NC*IMG_SIZE*IMG_SIZE),
                        # nn.ReLU(),
                        nn.Linear(NC*IMG_SIZE*IMG_SIZE, num_classes)
                    ]

        self.fc1 = nn.Sequential(*fc_layers_1)
        self.fc2 = nn.Sequential(*fc_layers_2)

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            feature = nn.parallel.data_parallel(self.fc1, x, range(self.ngpu))
            out = nn.parallel.data_parallel(self.fc2, feature, range(self.ngpu))
        else:
            feature = self.fc1(x)
            out = self.fc2(feature)
        return out, feature
