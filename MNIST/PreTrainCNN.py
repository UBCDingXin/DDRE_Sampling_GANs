"""
Train and Test CNN

Store model in disk

Pre-trained models are then used for GAN evaluation (i.e., InceptionV3) or DRE in feature space

"""

wd = '/home/xin/OneDrive/Working_directory/DDRE_Sampling_GANs/MNIST'

import argparse
import shutil
import os
os.chdir(wd)
import timeit
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import autograd
from torchvision.utils import save_image
import csv
from models import *
from tqdm import tqdm
import gc
import h5py
from utils import IMGs_dataset

#############################
# Settings
#############################

parser = argparse.ArgumentParser(description='Pre-train CNNs')
parser.add_argument('--CNN', type=str, default='ResNet34',
                    help='CNN for training (default: "ResNet34"); Candidates: VGGs(11,13,16,19), ResNet(18,34,50,101), InceptionV3')
parser.add_argument('--isometric_map', action='store_true', default=False,
                    help='isometric mapping? False for GAN evaluation; True for DRE in feature space')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train CNNs (default: 200)')
parser.add_argument('--batch_size_train', type=int, default=256, metavar='N',
                    help='input batch size for training')
parser.add_argument('--batch_size_test', type=int, default=256, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='learning rate, default=0.1')
parser.add_argument('--weight_dacay', type=float, default=1e-4,
                    help='Weigth decay, default=1e-4')
parser.add_argument('--seed', type=int, default=2019, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--transform', action='store_true', default=False,
                    help='crop images for CNN training')
parser.add_argument('--num_classes', type=int, default=10, metavar='N',
                    help='number of classes')
parser.add_argument('--N_TRAIN', type=int, default=5000, metavar='N',
                    help='number of training images')
args = parser.parse_args()

# cuda
device = torch.device("cuda")
ngpu = torch.cuda.device_count()  # number of gpus
args.base_lr = args.base_lr * ngpu
args.batch_size_train = args.batch_size_train*ngpu
args.batch_size_test = args.batch_size_test*ngpu

# random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# directories for checkpoint, images and log files
save_models_folder = wd + '/Output/saved_models/'
if not os.path.exists(save_models_folder):
    os.makedirs(save_models_folder)

save_logs_folder = wd + '/Output/saved_logs/'
if not os.path.exists(save_logs_folder):
    os.makedirs(save_logs_folder)



###########################################################################################################
# Necessary functions
###########################################################################################################

#initialize CNNs
def net_initialization(Pretrained_CNN_Name, isometric_map = False, num_classes=10, ngpu = 1):
    if Pretrained_CNN_Name == "ResNet18":
        net = ResNet18(isometric_map = isometric_map, num_classes=num_classes, ngpu = ngpu)
    elif Pretrained_CNN_Name == "ResNet34":
        net = ResNet34(isometric_map = isometric_map, num_classes=num_classes, ngpu = ngpu)
    elif Pretrained_CNN_Name == "ResNet50":
        net = ResNet50(isometric_map = isometric_map, num_classes=num_classes, ngpu = ngpu)
    elif Pretrained_CNN_Name == "ResNet101":
        net = ResNet101(isometric_map = isometric_map, num_classes=num_classes, ngpu = ngpu)
    elif Pretrained_CNN_Name == "InceptionV3":
        net = Inception3(num_classes=num_classes, aux_logits=True, transform_input=False)

    if isometric_map:
        net_name = 'PreCNNForDRE_' + Pretrained_CNN_Name #get the net's name
    else:
        net_name = 'PreCNNForEvalGANs_' + Pretrained_CNN_Name #get the net's name

    if Pretrained_CNN_Name == "InceptionV3" and ngpu>1:
        net = nn.DataParallel(net).cuda()
    else:
        net = net.to(device)

    return net, net_name

#adjust CNN learning rate
def adjust_learning_rate(optimizer, epoch, BASE_LR_CNN):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = BASE_LR_CNN
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (BASE_LR_CNN - 0.1) * epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 500:
        lr /= 10
    if epoch >= 1000:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_CNN():

    for epoch in range(args.epochs):
        net.train()
        train_loss = 0
        adjust_learning_rate(optimizer, epoch, args.base_lr)
        for batch_idx, (batch_train_images, batch_train_labels) in enumerate(trainloader):

            if args.CNN == "InceptionV3":
                batch_train_images = nn.functional.interpolate(batch_train_images, size = (299,299), scale_factor=None, mode='bilinear', align_corners=False)

            batch_train_images = batch_train_images.type(torch.float).cuda()
            batch_train_labels = batch_train_labels.type(torch.long).cuda()

            #Forward pass
            outputs,_ = net(batch_train_images)
            loss = criterion(outputs, batch_train_labels)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
        #end for batch_idx
        test_acc = test_CNN(False)

        print('CNN: [epoch %d/%d] train_loss:%.3f, test_acc:%.3f' % (epoch+1, args.epochs, train_loss/(batch_idx+1), test_acc))
    #end for epoch

    return net, optimizer


def test_CNN(verbose=True):

    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(testloader):
            if args.CNN == "InceptionV3":
                images = nn.functional.interpolate(images, size = (299,299), scale_factor=None, mode='bilinear', align_corners=False)
            images = images.type(torch.float).cuda()
            labels = labels.type(torch.long).cuda()
            outputs,_ = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        if verbose:
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100.0 * correct / total))
    return 100.0 * correct / total


###########################################################################################################
# Training and Testing
###########################################################################################################
# data loader
if args.N_TRAIN==60000:
    if args.transform:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=8)
else:
    h5py_file = wd+'/data/MNIST_reduced_trainset_'+str(args.N_TRAIN)+'.h5'
    hf = h5py.File(h5py_file, 'r')
    images_train = hf['images_train'][:]
    labels_train = hf['labels_train'][:]
    hf.close()
    if args.transform:
        trainset = IMGs_dataset(images_train, labels_train, normalize=True, rotate=True, degrees = 15, crop=True, crop_size=28, crop_pad=4)
    else:
        trainset = IMGs_dataset(images_train, labels_train, normalize=True, rotate=False, degrees = 15, crop=False, crop_size=28, crop_pad=4)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=8)
#end if args.N_TRAIN

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=8)

# model initialization
net, net_name = net_initialization(args.CNN, isometric_map = args.isometric_map, num_classes=args.num_classes, ngpu = ngpu)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = args.base_lr, momentum= 0.9, weight_decay=args.weight_dacay)

filename_ckpt = save_models_folder + '/ckpt_' + net_name + '_epoch_' + str(args.epochs) +  '_SEED_' + str(args.seed) + '_Transformation_' + str(args.transform) + '_NTRAIN_' + str(args.N_TRAIN)

# training
if not os.path.isfile(filename_ckpt):
    # TRAIN CNN
    print("\n Begin training CNN: ")
    start = timeit.default_timer()
    net, optimizer = train_CNN()
    stop = timeit.default_timer()
    print("Time elapses: {}s".format(stop - start))
    # save model
    torch.save({
    'net_state_dict': net.state_dict(),
    }, filename_ckpt)
else:
    print("\n Ckpt already exists")
    print("\n Loading...")
torch.cuda.empty_cache()#release GPU mem which is  not references

#testing
checkpoint = torch.load(filename_ckpt)
net.load_state_dict(checkpoint['net_state_dict'])
_ = test_CNN(True)
torch.cuda.empty_cache()
