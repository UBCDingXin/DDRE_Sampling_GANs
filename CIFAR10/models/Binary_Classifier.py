'''

A Convolutional Neural Network which is used to distinguish real and fake images. This classifier can be used for DRE.
The structure is based on "Bias Correction of Learned Generative Models using Likelihood-Free Importance Weighting"

https://github.com/BoyuanJiang/matching-networks-pytorch/blob/master/matching_networks.py

'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

NC=3
IMG_SIZE=32

def convLayer(in_channels, out_channels, keep_prob=0.0):
    """3*3 convolution with padding,ever time call it the output size become half"""
    cnn_seq = [
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.ReLU(True),
                nn.BatchNorm2d(out_channels),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(keep_prob)
            ]
    return cnn_seq

class BiClassifier(nn.Module):
    def __init__(self, ngpu=1, num_channels=NC, image_size=IMG_SIZE, layer_size=64, keep_prob=1.0):
        super(BiClassifier, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:
        """
        self.ngpu=ngpu
        convs = convLayer(num_channels, layer_size, keep_prob)
        convs += convLayer(layer_size, layer_size, keep_prob)
        convs += convLayer(layer_size, layer_size, keep_prob)
        convs += convLayer(layer_size, layer_size, keep_prob)
        self.convs = nn.Sequential(*convs)
        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        outSize = finalSize * finalSize * layer_size

        self.classifier = nn.Sequential(
                nn.Linear(outSize, 1),
                nn.Sigmoid()
            )


    def forward(self, x):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        if x.is_cuda and self.ngpu > 1:
            out = nn.parallel.data_parallel(self.convs, x, range(self.ngpu))
            out = out.view(out.size(0), -1)
            out = nn.parallel.data_parallel(self.classifier, out, range(self.ngpu))
        else:
            out = self.convs(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
        return out


if __name__ == "__main__":
    net = BiClassifier(ngpu=2, num_channels=NC, image_size=IMG_SIZE).cuda()
    img = Variable(torch.randn(25, NC, IMG_SIZE, IMG_SIZE)).cuda()
    out = net(img)
    print(out.size())
