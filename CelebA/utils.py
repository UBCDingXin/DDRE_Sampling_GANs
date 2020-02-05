"""
Some helpful functions

"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
import PIL
from PIL import Image
import random

################################################################################
# Progress Bar
class SimpleProgressBar():
    def __init__(self, width=50):
        self.last_x = -1
        self.width = width

    def update(self, x):
        assert 0 <= x <= 100 # `x`: progress in percent ( between 0 and 100)
        if self.last_x == int(x): return
        self.last_x = int(x)
        pointer = int(self.width * (x / 100.0))
        sys.stdout.write( '\r%d%% [%s]' % (int(x), '#' * pointer + '.' * (self.width - pointer)))
        sys.stdout.flush()
        if x == 100:
            print('')

################################################################################
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

################################################################################
# torch dataset for CelebA
TransHFlip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
class celeba_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, normalize_img = True, random_transform = False, means_imgs = (0.5,0.5,0.5), stds_imgs = (0.5,0.5,0.5)):
        super(celeba_dataset, self).__init__()

        self.images = images
        self.labels = labels
        self.n_images, self.nc, self.nx, self.ny = self.images.shape
        self.normalize_img = normalize_img
        self.random_transform = random_transform
        self.means_imgs = means_imgs
        self.stds_imgs = stds_imgs

    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]

        # random transformation: horizontal flipping
        if self.random_transform:
            image = np.transpose(image, (1, 2, 0)) #now W * H * C
            PIL_im = Image.fromarray(np.uint8(image), mode = 'RGB')
            PIL_im = TransHFlip(PIL_im)
            image = np.array(PIL_im)
            image = np.transpose(image, (2, 0, 1)) #now C * W * H

        # normalization
        if self.normalize_img:
            image = image / 255.0
            for i in range(3):
                image[i,:,:] = (image[i,:,:] - self.means_imgs[i]) / self.stds_imgs[i]

        return (image, label)

    def __len__(self):
        return self.n_images


#---------------------------------------------------------------------------------
class celeba_h5_dataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, normalize_img = True, random_transform = False, means_imgs = (0.5,0.5,0.5), stds_imgs = (0.5,0.5,0.5)):
        super(celeba_h5_dataset, self).__init__()

        self.root = root
        self.train = train
        if train:
            self.num_imgs = len(h5py.File(root, 'r')['labels_train'])
        else:
            self.num_imgs = len(h5py.File(root, 'r')['labels_valid'])

        self.normalize_img = normalize_img
        self.random_transform = random_transform
        self.means_imgs = means_imgs
        self.stds_imgs = stds_imgs

    def __getitem__(self, index):

        with h5py.File(self.root,'r') as f:
            if self.train:
                image = f['images_train'][index]
                label = f['labels_train'][index]
            else:
                image = f['images_valid'][index]
                label = f['labels_valid'][index]

        # random transformation: horizontal flipping
        if self.random_transform:
            image = np.transpose(image, (1, 2, 0)) #now W * H * C
            PIL_im = Image.fromarray(np.uint8(image), mode = 'RGB')
            PIL_im = TransHFlip(PIL_im)
            image = np.array(PIL_im)
            image = np.transpose(image, (2, 0, 1)) #now C * W * H

        # normalization
        if self.normalize_img:
            image = image / 255.0
            for i in range(3):
                image[i,:,:] = (image[i,:,:] - self.means_imgs[i]) / self.stds_imgs[i]

        return (image, label)

    def __len__(self):
        return self.num_imgs

################################################################################
# torch dataset from numpy array
class IMGs_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None):
        super(IMGs_dataset, self).__init__()

        self.images = images
        self.labels = labels
        if labels is not None:
            self.labels = labels
        self.n_images = len(self.images)

    def __getitem__(self, index):

        image = self.images[index]
        if self.labels is not None:
            label = self.labels[index]
            return (image, label)
        else:
            return image

    def __len__(self):
        return self.n_images

################################################################################
# plot loss
def PlotLoss(loss, filename):
    x_axis = np.arange(start = 1, stop = len(loss)+1)
    plt.switch_backend('agg')
    mpl.style.use('seaborn')
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_axis, np.array(loss))
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.legend()
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=3)
    #plt.title('Training Loss')
    plt.savefig(filename)


################################################################################
def SampPreGAN(netG, GAN_Latent_Length = 100, Conditional = False, NFAKE = 10000, BATCH_SIZE = 500, N_CLASS = 8, NC = 3, IMG_SIZE = 64):
    raw_fake_images = np.zeros((NFAKE+BATCH_SIZE, NC, IMG_SIZE, IMG_SIZE))
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            if not Conditional:
                z = torch.randn(BATCH_SIZE, GAN_Latent_Length, 1, 1, dtype=torch.float).cuda()
                batch_fake_images = netG(z)
                batch_fake_images = batch_fake_images.detach()
            else:
                z = torch.randn(BATCH_SIZE, GAN_Latent_Length, dtype=torch.float).cuda()
                gen_labels = torch.from_numpy(np.random.randint(0,N_CLASS,BATCH_SIZE)).type(torch.long).cuda()
                gen_labels_onehot = torch.FloatTensor(BATCH_SIZE, N_CLASS).cuda()
                gen_labels_onehot.zero_()
                gen_labels_onehot.scatter_(1,gen_labels.reshape(BATCH_SIZE,1),1)
                batch_fake_images = netG(z, gen_labels_onehot)
                batch_fake_images = batch_fake_images.detach()
            raw_fake_images[tmp:(tmp+BATCH_SIZE)] = batch_fake_images.cpu().detach().numpy()
            tmp += BATCH_SIZE
    #remove unused entry and extra samples
    raw_fake_images = raw_fake_images[0:NFAKE]
    return raw_fake_images
