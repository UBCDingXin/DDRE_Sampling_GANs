"""
Train and Test CNN

Pre-trained models are then used for Density ratio estimation in feature space

Finetune fc layers of ResNet
"""

wd = '/home/xin/OneDrive/Working_directory/DDRE_Sampling_GANs/STL10_64'


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
from tqdm import tqdm
import gc
import pickle
import h5py

from models import *
from utils import *

#######################################################################################
'''                                   Settings                                      '''
#######################################################################################

parser = argparse.ArgumentParser(description='Pre-train CNNs')
parser.add_argument('--CNN', type=str, default='ResNet34',
                    choices = ["ResNet18", "ResNet34", "ResNet50", "ResNet101"],
                    help='CNN for training (default: "ResNet34"); Candidates: VGGs(11,13,16,19), ResNet(18,34,50,101), InceptionV3')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train CNNs (default: 200)')
parser.add_argument('--batch_size_train', type=int, default=128, metavar='N',
                    help='input batch size for training')
parser.add_argument('--batch_size_test', type=int, default=256, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--base_lr', type=float, default=0.1,
                    help='learning rate, default=0.1')
parser.add_argument('--weight_dacay', type=float, default=1e-4,
                    help='Weigth decay, default=1e-4')
parser.add_argument('--seed', type=int, default=2019, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--transform', action='store_true', default=False,
                    help='flip images for CNN training')
#parser.add_argument('--num_classes', type=int, default=1000, metavar='N',
#                    help='number of classes')
parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda',
                    choices=['cuda', 'cuda:0', 'cuda:1'])
args = parser.parse_args()



IMG_SIZE = 64
N_CLASS = 10


# cuda
device = torch.device(args.device)
ngpu = torch.cuda.device_count()  # number of gpus
args.base_lr = args.base_lr * ngpu
args.batch_size_train = args.batch_size_train*ngpu
args.batch_size_test = args.batch_size_test*ngpu
NCPU = 0
#ngpu = 1

# random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed)

# directories for checkpoint, images and log files
save_models_folder = wd + '/Output/saved_models'
os.makedirs(save_models_folder, exist_ok=True)

save_models_duringTrain_folder = save_models_folder + '/ckpt_during_train'
os.makedirs(save_models_duringTrain_folder, exist_ok=True)




###########################################################################################################
'''                                          Necessary functions                                      '''
###########################################################################################################
#adjust CNN learning rate
def adjust_learning_rate(optimizer, epoch, BASE_LR_CNN):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = BASE_LR_CNN
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (BASE_LR_CNN - 0.1) * epoch / 10.
    if epoch >= 10:
        lr /= 10
    if epoch >= 30:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_fc():

    if args.resume_epoch>0:
        filename_ckpt = save_models_duringTrain_folder + "/ckpt_pretrained_" + args.CNN + "_keeptrain_fc_epoch_" + str(args.resume_epoch) + "_SEED_" + str(args.seed) + "_Transformation_" + str(args.transform)
        checkpoint = torch.load(filename_ckpt)
        fc_net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start = timeit.default_timer()
    for epoch in range(args.resume_epoch, args.epochs):
        fc_net.train()
        ResNet_net.eval()
        train_loss = 0
        adjust_learning_rate(optimizer, epoch, args.base_lr)
        for batch_idx, (batch_train_images, batch_train_labels) in enumerate(trainloader):

            batch_train_images = nn.functional.interpolate(batch_train_images, size = (224,224), scale_factor=None, mode='bilinear', align_corners=False)
            batch_train_images = batch_train_images.type(torch.float).to(device)
            batch_train_labels = batch_train_labels.type(torch.long).to(device)
            _, batch_train_features = ResNet_net(batch_train_images)

            #Forward pass
            outputs,_ = fc_net(batch_train_features.detach())
            loss = criterion(outputs, batch_train_labels)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()

            print('CNN: [idx %d/%d] [epoch %d/%d] [train_loss:%.3f] [elapse:%.3f]' % (batch_idx+1, len(trainloader), epoch+1, args.epochs, loss.cpu().item(), timeit.default_timer()-start))
        #end for batch_idx

        if (epoch+1)%20 == 0:
            filename_ckpt = save_models_duringTrain_folder + '/ckpt_pretrained_' + args.CNN + '_keeptrain_fc_epoch_' + str(epoch+1) +  '_SEED_' + str(args.seed) + '_Transformation_' + str(args.transform)
            torch.save({
            'epoch': epoch+1,
            'net_state_dict': fc_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, filename_ckpt)

        test_acc = test_fc(False)

        print('CNN: [epoch %d/%d] train_loss:%.3f, test_acc:%.3f, time elapses:%.3f' % (epoch+1, args.epochs, train_loss/(batch_idx+1), test_acc, timeit.default_timer()-start))


    #end for epoch
    stop = timeit.default_timer()
    print("CNN Training finished! Time elapses: {}s".format(stop - start))

    return fc_net, optimizer


def test_fc(verbose=True):
    ResNet_net.eval()
    fc_net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images = nn.functional.interpolate(images, size = (224,224), scale_factor=None, mode='bilinear', align_corners=False)
            images = images.type(torch.float).to(device)
            labels = labels.type(torch.long).to(device)
            _, features = ResNet_net(images)
            outputs,_ = fc_net(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        if verbose:
            print('fc; Test Accuracy of the model on the test images: {} %'.format(100.0 * correct / total))
    return 100.0 * correct / total


###########################################################################################################
'''                                           Training and Testing                                     '''
###########################################################################################################
# data loader
means = (0.5, 0.5, 0.5)
stds = (0.5, 0.5, 0.5)
transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.Resize(IMG_SIZE),
        transforms.RandomCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])

transform_test = transforms.Compose([
    # transforms.Resize(IMG_SIZE),
    transforms.RandomCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
])
trainset = torchvision.datasets.STL10(root='./data_STL10', split='test', folds=None, transform=transform_train) #"test" set has more data
testset = torchvision.datasets.STL10(root='./data_STL10', split='train', folds=None, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=NCPU)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=NCPU)


# model initialization
if args.CNN == "ResNet34":
    ResNet_net = resnet34(pretrained=True, progress=True)
elif args.CNN == "ResNet50":
    ResNet_net = resnet50(pretrained=True, progress=True)
ResNet_net = nn.DataParallel(ResNet_net).to(device)

# #load finetuned resnet
# ckpt_resnet = torch.load(ckpt_filename_ResNet)
# ResNet_net.load_state_dict(ckpt_resnet['net_state_dict'])

fc_net = ResNet_keeptrain_fc(ResNet_name = args.CNN, ngpu = ngpu, num_classes = N_CLASS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(fc_net.parameters(), lr = args.base_lr, momentum= 0.9, weight_decay=args.weight_dacay)

filename_ckpt = save_models_folder + '/ckpt_pretrained_' + args.CNN + '_keeptrain_fc_epoch_' + str(args.epochs) +  '_SEED_' + str(args.seed) + '_Transformation_' + str(args.transform)

# training
if not os.path.isfile(filename_ckpt):
    # TRAIN CNN
    print("\n Begin training CNN: ")
    start = timeit.default_timer()
    fc_net, optimizer = train_fc()
    stop = timeit.default_timer()
    print("Time elapses: {}s".format(stop - start))
    # save model
    torch.save({
    'net_state_dict': fc_net.state_dict(),
    }, filename_ckpt)
else:
    print("\n Ckpt already exists")
    print("\n Loading...")
torch.cuda.empty_cache()#release GPU mem which is  not references

#testing
checkpoint = torch.load(filename_ckpt)
fc_net.load_state_dict(checkpoint['net_state_dict'])
_ = test_fc(True)
torch.cuda.empty_cache()
