
import os
wd = '/home/xin/OneDrive/Working_directory/DDRE_Sampling_GANs/MNIST'

os.chdir(wd)
import timeit
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import autograd
from torchvision.utils import save_image
from tqdm import tqdm
import gc
from itertools import groupby
import argparse
from sklearn.linear_model import LogisticRegression
import multiprocessing
from multiprocessing import Pool
from scipy.stats import ks_2samp
import h5py

from utils import *
from models import *
from Train_DRE import *
from Train_DCGAN import train_DCGAN, SampDCGAN
from Train_WGAN import train_WGANGP, SampWGAN
from Train_MMDGAN import train_MMDGAN, SampMMDGAN
from SimpleProgressBar import SimpleProgressBar
from eval_metrics import FID, FID_RAW, IS_RAW

#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
parser = argparse.ArgumentParser(description='Density-ratio based sampling for GANs')
'''Overall Settings'''
parser.add_argument('--GAN', type=str, default='DCGAN',
                    choices=['DCGAN', 'WGANGP', 'MMDGAN'],
                    help='GAN model')
parser.add_argument('--DRE', type=str, default='None',
                    choices=['None', 'disc', 'disc_KeepTrain', 'disc_KeepTrain_MHcal', 'disc_MHcal', 'BayesClass',
                             'DRE_F_SP', 'DRE_F_uLSIF', 'DRE_F_DSKL', 'DRE_F_BARR',
                             'DRE_P_SP', 'DRE_P_uLSIF', 'DRE_P_DSKL', 'DRE_P_BARR'],
                    help='Density ratio estimation method') # disc: ratio=D/(1-D); disc_DRS: method in "Discriminator Rejction Sampling"; disc_MHcal: the calibration method in MH-GAN; BayesClass: a Bayes Optimal Binary classifier;
parser.add_argument('--Sampling', type=str, default='None',
                    choices=['None', 'RS', 'MH', 'SIR'],
                    help='Sampling/Resampling method for GANs; Candidiate: None, RS, MH, SIR') #RS: rejection sampling, MH: Metropolis-Hastings; SIR: Sampling-Importance Resampling
parser.add_argument('--seed', type=int, default=2019, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--N_TRAIN', type=int, default=1000, metavar='N',
                    help='number of training images')

''' GAN settings '''
parser.add_argument('--transform_GAN_and_DRE', action='store_true', default=True,
                    help='rotate or crop images for GAN and DRE training')
parser.add_argument('--epoch_gan', type=int, default=2000)
parser.add_argument('--lr_g_gan', type=float, default=1e-4,
                    help='learning rate for generator')
parser.add_argument('--lr_d_gan', type=float, default=1e-4,
                    help='learning rate for discriminator')
parser.add_argument('--dim_gan', type=int, default=128,
                    help='Latent dimension of GAN')
parser.add_argument('--batch_size_gan', type=int, default=256, metavar='N',
                    help='input batch size for training GAN')
parser.add_argument('--resumeTrain_gan', type=int, default=0)


'''DRE settings'''
## DRE_F_SP
parser.add_argument('--DR_Net', type=str, default='MLP5',
                    choices=['MLP3', 'MLP5', 'MLP7', 'MLP9',
                             'ResNet18', 'ResNet34','2layersCNN', '6layersCNN',
                             'VGG11', 'VGG13','VGG16','VGG19'],
                    help='DR Model; Candidates: ResNetXX(18,34,50,101,152) or MLPX(3,5,7,9)') # DRE in Pixel/Feature Space
parser.add_argument('--PreCNN_DR', type=str, default='ResNet34',
                    choices=['ResNet34'],
                    help='Pre-trained CNN for DRE in Feature Space; Candidate: ResNetXX')
parser.add_argument('--epoch_pretrainCNN', type=int, default=200)
parser.add_argument('--transform_PreCNN_DR', action='store_true', default=True,
                    help='rotate or crop images for CNN training')
parser.add_argument('--epoch_DRE', type=int, default=200) #default -1
parser.add_argument('--base_lr_DRE', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--not_decay_lr_DRE', action='store_true', default=False,
                    help='not decay learning rate')
parser.add_argument('--batch_size_DRE', type=int, default=512, metavar='N',
                    help='input batch size for training DRE')
parser.add_argument('--lambda_DRE', type=float, default=1,
                    help='penalty in DRE')
parser.add_argument('--weightdecay_DRE', type=float, default=1e-4,
                    help='weight decay in DRE')
parser.add_argument('--resumeTrain_DRE', type=int, default=0)
parser.add_argument('--DR_ResNet_fc', action='store_true', default=False,
                    help='Use fc layers in DR_ResNet?')
parser.add_argument('--DR_final_ActFn', type=str, default='ReLU',
                    help='Final layer of the Density-ratio model; Candidiate: Softplus or ReLU')
parser.add_argument('--replot_train_loss', action='store_true', default=False,
                    help='re-plot training loss')

'''Sampling and Comparing Settings'''
parser.add_argument('--samp_round', type=int, default=3)
parser.add_argument('--samp_nfake', type=int, default=50000)
parser.add_argument('--samp_batch_size', type=int, default=5000)
parser.add_argument('--samp_selectwithinclass', action='store_true', default=False,
                    help='In comparison, reselect samples within each class?')
parser.add_argument('--realdata_ISFID', action='store_true', default=False,
                    help='Print IS and FID for real data?')
parser.add_argument('--comp_ISFID', action='store_true', default=False,
                    help='Compute IS and FID for fake data?')
parser.add_argument('--IS_batch_size', type=int, default=100)
parser.add_argument('--FID_batch_size', type=int, default=100)
args = parser.parse_args()

#--------------------------------
# system
args.cuda = not args.no_cuda and torch.cuda.is_available()
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if args.cuda else "cpu")
NCPU = multiprocessing.cpu_count()
# NCPU = 0
cudnn.benchmark = True # For fast training

#-------------------------------
# GAN
N_CLASS = 10
NC = 1 #number of channels
IMG_SIZE = 28
ResumeEpoch_gan = args.resumeTrain_gan
resize = (299, 299)
ADAM_beta1 = 0.5 #parameters for ADAM optimizer
ADAM_beta2 = 0.999

#-------------------------------
# sampling parameters
NROUND = args.samp_round
NFAKE = args.samp_nfake
NPOOL = NFAKE*2 #Pool size for reselecting imgs
samp_batch_size = args.samp_batch_size #batch size for dsampling from GAN or enhanced sampler
if samp_batch_size>args.N_TRAIN:
    samp_batch_size = args.N_TRAIN
MH_K = 640
MH_mute = True #do not print sampling progress
DR_comp_batch_size = 1000
assert samp_batch_size>=DR_comp_batch_size

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed)

#-------------------------------
# output folders
save_models_folder = wd + '/Output/saved_models'
os.makedirs(save_models_folder,exist_ok=True)
save_images_folder = wd + '/Output/saved_images'
os.makedirs(save_images_folder,exist_ok=True)
save_GANimages_InTrain_folder = wd + '/Output/saved_images/'+args.GAN+'_InTrain/'
os.makedirs(save_GANimages_InTrain_folder,exist_ok=True)
save_traincurves_folder = wd + '/Output/Training_loss_fig'
os.makedirs(save_traincurves_folder,exist_ok=True)




#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
if args.N_TRAIN==60000:
    if args.transform_GAN_and_DRE:
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
    images_train = trainset.data.numpy()
    images_train = images_train[:,np.newaxis,:,:]
    labels_train = trainset.targets.numpy()
else:
    h5py_file = wd+'/data/MNIST_reduced_trainset_'+str(args.N_TRAIN)+'.h5'
    hf = h5py.File(h5py_file, 'r')
    images_train = hf['images_train'][:]
    labels_train = hf['labels_train'][:]
    hf.close()
    if args.transform_GAN_and_DRE:
        trainset = IMGs_dataset(images_train, labels_train, normalize=True, rotate=True, degrees = 15, crop=True, crop_size=28, crop_pad=4)
    else:
        trainset = IMGs_dataset(images_train, labels_train, normalize=True, rotate=False, degrees = 15, crop=False, crop_size=28, crop_pad=4)
#end if args.N_TRAIN
trainloader_GAN = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_gan, shuffle=True, num_workers=NCPU)
trainloader_DRE = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_DRE, shuffle=True, num_workers=NCPU)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
images_test = testset.data.numpy()
images_test = images_test[:,np.newaxis,:,:]
labels_test = testset.targets.numpy()

#######################################################################################
'''                             Train GAN or Load Pre-trained GAN                '''
#######################################################################################
Filename_GAN = save_models_folder + '/ckpt_'+ args.GAN +'_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed)
print("\n Begin Training GAN:")
start = timeit.default_timer()
#-------------------------------
## DCGAN
if args.GAN == "DCGAN" and not os.path.isfile(Filename_GAN):
    #model initialization
    netG = cnn_generator(NGPU, args.dim_gan)
    netD = cnn_discriminator(True, NGPU)
    if args.resumeTrain_gan==0:
        netG.apply(weights_init)
        netD.apply(weights_init)
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g_gan, betas=(ADAM_beta1, ADAM_beta2))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))

    # Start training
    netG, netD, optimizerG, optimizerD = train_DCGAN(args.epoch_gan, args.dim_gan, trainloader_GAN, netG, netD, optimizerG, optimizerD, criterion, save_GANimages_InTrain_folder, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan)

    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, Filename_GAN)
#-------------------------------
## WGANGP
elif args.GAN == "WGANGP" and not os.path.isfile(Filename_GAN):
    #model initialization
    netG = cnn_generator(NGPU, args.dim_gan)
    netD = cnn_discriminator(False, NGPU)
    if args.resumeTrain_gan==0:
        netG.apply(weights_init)
        netD.apply(weights_init)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g_gan, betas=(ADAM_beta1, ADAM_beta2))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))

    # Start training
    netG, netD, optimizerG, optimizerD = train_WGANGP(args.epoch_gan, args.dim_gan, trainloader_GAN, netG, netD, optimizerG, optimizerD, save_GANimages_InTrain_folder, LAMBDA = 10, CRITIC_ITERS=5, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan)
    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, Filename_GAN)
#-------------------------------
## MMDGAN
elif args.GAN == "MMDGAN" and not os.path.isfile(Filename_GAN):
    #model initialization
    # construct encoder/decoder modules
    hidden_dim = args.dim_gan
    G_decoder = MMDGAN_Decoder(IMG_SIZE, NC, k=args.dim_gan, ngf=64, ngpu=NGPU)
    D_encoder = MMDGAN_Encoder(IMG_SIZE, NC, k=hidden_dim, ndf=64, ngpu=NGPU)
    D_decoder = MMDGAN_Decoder(IMG_SIZE, NC, k=hidden_dim, ngf=64, ngpu=NGPU)

    netG = MMDGAN_G(G_decoder)
    netD = MMDGAN_D(D_encoder, D_decoder)
    one_sided = MMDGAN_ONE_SIDED()

    if args.resumeTrain_gan==0:
        netG.apply(MMDGAN_weights_init)
        netD.apply(MMDGAN_weights_init)
        one_sided.apply(MMDGAN_weights_init)

    # put variable into cuda device
    fixed_noise = torch.cuda.FloatTensor(64, args.dim_gan, 1, 1).normal_(0, 1)
    one = torch.cuda.FloatTensor([1])
    mone = one * -1

    fixed_noise = torch.autograd.Variable(fixed_noise, requires_grad=False)

    # setup optimizer
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.lr_g_gan)
    optimizerD = torch.optim.RMSprop(netD.parameters(), lr=args.lr_d_gan)

    # Start training
    netG, netD, optimizerG, optimizerD = train_MMDGAN(args.epoch_gan, args.dim_gan, trainloader_GAN, netG, netD, one_sided, optimizerG, optimizerD, save_GANimages_InTrain_folder, CRITIC_ITERS=5, NC = NC, IMG_SIZE = IMG_SIZE, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan)
    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, Filename_GAN)

torch.cuda.empty_cache()
stop = timeit.default_timer()
print("GAN training finished! Time elapses: {}s".format(stop - start))



###############################################################################
'''                      Define Density-ratio function                      '''
###############################################################################
def CNN_net_init(Pretrained_CNN_Name, N_CLASS, NGPU, isometric_map = False):
    if Pretrained_CNN_Name == "ResNet18":
        net = ResNet18(isometric_map = isometric_map, num_classes=N_CLASS, ngpu = NGPU)
    elif Pretrained_CNN_Name == "ResNet34":
        net = ResNet34(isometric_map = isometric_map, num_classes=N_CLASS, ngpu = NGPU)
    elif Pretrained_CNN_Name == "ResNet50":
        net = ResNet50(isometric_map = isometric_map, num_classes=N_CLASS, ngpu = NGPU)
    elif Pretrained_CNN_Name == "ResNet101":
        net = ResNet101(isometric_map = isometric_map, num_classes=N_CLASS, ngpu = NGPU)
    elif Pretrained_CNN_Name == "InceptionV3":
        net = Inception3(num_classes=10, aux_logits=True, transform_input=False)

    if isometric_map:
        net_name = 'PreCNNForDRE_' + Pretrained_CNN_Name #get the net's name
    else:
        net_name = 'PreCNNForEvalGANs_' + Pretrained_CNN_Name #get the net's name
    return net, net_name

#######################################################
# Construct a function to compute density-ratio
###################
# Approximate DR by NN
if args.DRE in ['DRE_F_SP', 'DRE_F_uLSIF', 'DRE_F_DSKL', 'DRE_F_BARR',
                'DRE_P_SP', 'DRE_P_uLSIF', 'DRE_P_DSKL', 'DRE_P_BARR']:

    DRE_loss_type = args.DRE[6:]

    def DR_net_init(DR_net_name):
        if DR_net_name == "ResNet18":
            assert args.DRE[4] == "P"
            net = DR_ResNet18(ngpu=NGPU, fc_layers=args.DR_ResNet_fc, final_ActFn=args.DR_final_ActFn)
        elif DR_net_name == "ResNet34":
            assert args.DRE[4] == "P"
            net =DR_ResNet34(ngpu=NGPU, fc_layers=args.DR_ResNet_fc, final_ActFn=args.DR_final_ActFn)
        elif DR_net_name in ["MLP3", "MLP5", "MLP7", "MLP9"]:
            assert args.DRE[4] == "F"
            net = DR_MLP(DR_net_name, ngpu=NGPU, final_ActFn=args.DR_final_ActFn)
        elif DR_net_name == "2layersCNN":
            assert args.DRE[4] == "P"
            net = DR_2layersCNN(ngpu=NGPU)
        elif DR_net_name == "6layersCNN":
            assert args.DRE[4] == "P"
            net = DR_6layersCNN(ngpu=NGPU)
        elif DR_net_name in ['VGG11','VGG13','VGG16','VGG19']:
            assert args.DRE[4] == "P"
            net = DR_VGG(DR_net_name, fc_layers=args.DR_ResNet_fc, NGPU=NGPU, final_ActFn=args.DR_final_ActFn)
        else:
            raise Exception("Select a valid density ratio model!!!")
        return net

    # Load Pre-trained GAN
    checkpoint = torch.load(Filename_GAN)
    if args.GAN in ["DCGAN","WGANGP"]:
        netG = cnn_generator(NGPU, args.dim_gan).to(device)
    elif args.GAN == "MMDGAN":
        G_decoder = MMDGAN_Decoder(IMG_SIZE, NC, k=args.dim_gan, ngf=64, ngpu=NGPU)
        netG = MMDGAN_G(G_decoder).to(device)
    netG.load_state_dict(checkpoint['netG_state_dict'])

    # PreTrain CNN for DRE in feature space
    _, net_name = CNN_net_init(args.PreCNN_DR, N_CLASS, NGPU, isometric_map = True)
    Filename_PreCNNForDRE = save_models_folder + '/ckpt_' + net_name + '_epoch_' + str(args.epoch_pretrainCNN) + '_SEED_' + str(args.seed) + '_Transformation_' + str(args.transform_PreCNN_DR) + '_NTRAIN_' + str(args.N_TRAIN)

    #-----------------------------------------
    # Train DR model
    start = timeit.default_timer()
    # initialize DRE model
    netDR = DR_net_init(args.DR_Net)

    if DRE_loss_type == "DSKL" and args.DRE=="DRE_P_DSKL":
        optimizer = torch.optim.RMSprop(netDR.parameters(), lr= args.base_lr_DRE, alpha=0.99, eps=1e-08, weight_decay=args.weightdecay_DRE, momentum=0.9, centered=False)
    else:
        optimizer = torch.optim.Adam(netDR.parameters(), lr = args.base_lr_DRE, betas=(ADAM_beta1, ADAM_beta2), weight_decay=args.weightdecay_DRE)

    Filename_DRE = save_models_folder + '/ckpt_'+ args.DRE +'_' + args.DR_Net + '_' + args.DR_final_ActFn + '_epoch_' + str(args.epoch_DRE) + '_SEED_' + str(args.seed) + '_Lambda_' + str(args.lambda_DRE) + "_" + args.GAN + "_epoch_" + str(args.epoch_gan)

    if args.DRE[4] == "P" and not os.path.isfile(Filename_DRE): #DRE in pixel space
        print("\n Begin Training DR in Pixel Space: >>>\n")
        netDR, optimizer, avg_train_loss = train_DREP(NGPU, args.epoch_DRE, args.base_lr_DRE, trainloader_DRE, netDR, optimizer, netG, args.dim_gan, LAMBDA = args.lambda_DRE, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_DRE, loss_type = DRE_loss_type, device=device, not_decay_lr=args.not_decay_lr_DRE, name_gan = args.GAN)
    elif args.DRE[4] == "F" and not os.path.isfile(Filename_DRE): #DRE in feature space
        print("\n Begin Training DR in Feature Space: >>>\n")
        ### load pretrained CNN
        PreNetDRE, _ = CNN_net_init(args.PreCNN_DR, N_CLASS, NGPU, isometric_map = True)
        checkpoint = torch.load(Filename_PreCNNForDRE)
        # PreNetDRE = PreNetDRE.to(device)
        PreNetDRE.load_state_dict(checkpoint['net_state_dict'])
        netDR, optimizer, avg_train_loss = train_DREF(NGPU, args.epoch_DRE, args.base_lr_DRE, trainloader_DRE, netDR, optimizer, PreNetDRE, netG, args.dim_gan, LAMBDA = args.lambda_DRE, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_DRE, loss_type = DRE_loss_type, device=device, not_decay_lr=args.not_decay_lr_DRE, name_gan = args.GAN)

    if not os.path.isfile(Filename_DRE):
        # Plot loss
        filename = save_traincurves_folder + '/' + args.DRE + '_' + args.DR_Net + "_LAMBDA"+ str(args.lambda_DRE) + "_epochDRE" + str(args.epoch_DRE) + "_" + args.GAN + "_epochGAN" + str(args.epoch_gan) + "_TrainLoss"
        PlotLoss(avg_train_loss, filename+".pdf")
        np.save(filename, np.array(avg_train_loss))
        # save model
        torch.save({
        'net_state_dict': netDR.state_dict(),
        }, Filename_DRE)
    else:
        if args.replot_train_loss:
            filename = save_traincurves_folder + '/' + args.DRE + '_' + args.DR_Net + "_LAMBDA"+ str(args.lambda_DRE) + "_epochDRE" + str(args.epoch_DRE) + "_" + args.GAN + "_epochGAN" + str(args.epoch_gan) + "_TrainLoss"
            avg_train_loss = np.load(filename+".npy")
            PlotLoss(avg_train_loss, filename+".pdf")
    torch.cuda.empty_cache()

    #-----------------------------------------
    # if already trained, load pre-trained DR model
    if args.DRE[4] == "P": #pixel space
        PreNetDRE = None
    else: #feature space
        PreNetDRE, _ = CNN_net_init(args.PreCNN_DR, N_CLASS, NGPU, isometric_map = True)
        checkpoint = torch.load(Filename_PreCNNForDRE)
        PreNetDRE = PreNetDRE.to(device)
        PreNetDRE.load_state_dict(checkpoint['net_state_dict'])
    checkpoint_netDR = torch.load(Filename_DRE)
    netDR = DR_net_init(args.DR_Net)
    netDR.load_state_dict(checkpoint_netDR['net_state_dict'])
    netDR = netDR.to(device)

    stop = timeit.default_timer()
    print("DRE fitting finished; Time elapses: {}s".format(stop - start))

    # function for computing a bunch of images in a numpy array
    # def comp_density_ratio(imgs, netDR, PreNetDRE=None):
    def comp_density_ratio(imgs):
        #imgs: an numpy array
        n_imgs = imgs.shape[0]
        batch_size_tmp = DR_comp_batch_size
        dataset_tmp = IMGs_dataset(imgs)
        dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
        data_iter = iter(dataloader_tmp)
        density_ratios = np.zeros((n_imgs+batch_size_tmp, 1))

        netDR.eval()
        if args.DRE[4] == "F": #in feature space
            PreNetDRE.eval()
        # print("\n Begin computing density ratio for images >>")
        with torch.no_grad():
            tmp = 0
            while tmp < n_imgs:
                batch_imgs = data_iter.next()
                batch_imgs = batch_imgs.type(torch.float).to(device)
                if args.DRE[4] == "P":
                    batch_weights = netDR(batch_imgs)
                else:
                    _, batch_features = PreNetDRE(batch_imgs)
                    batch_weights = netDR(batch_features)
                density_ratios[tmp:(tmp+batch_size_tmp)] = batch_weights.cpu().detach().numpy()
                tmp += batch_size_tmp
            #end while
        # print("\n End computing density ratio.")
        return density_ratios[0:n_imgs]


###################
# DRE based on GAN property
elif args.DRE in ['disc', 'disc_KeepTrain', 'disc_KeepTrain_MHcal', 'disc_MHcal']:
    # Load Pre-trained GAN
    checkpoint = torch.load(Filename_GAN)
    if args.GAN == "DCGAN":
        netG = cnn_generator(NGPU, args.dim_gan).to(device)
        netD = cnn_discriminator(True, NGPU).to(device)
        def fn_sampleGAN(nfake, batch_size):
            return SampDCGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
    elif args.GAN == "WGANGP":
        netG = cnn_generator(NGPU, args.dim_gan).to(device)
        netD = cnn_discriminator(False, NGPU).to(device)
        def fn_sampleGAN(nfake, batch_size):
            return SampWGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
    elif args.GAN == "MMDGAN":
        raise Exception('Current DRE method "%s" is not suitable for MMDGAN!' % (args.DRE))
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])


    #-----------------------------------
    if args.DRE == 'disc': #use GAN property to compute density ratio; ratio=D/(1-D); #for DCGAN, WGAN,
        # function for computing a bunch of images
        # def comp_density_ratio(imgs, netD):
        def comp_density_ratio(imgs):
            #imgs: an numpy array
            n_imgs = imgs.shape[0]
            batch_size_tmp = DR_comp_batch_size
            dataset_tmp = IMGs_dataset(imgs)
            dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
            data_iter = iter(dataloader_tmp)
            density_ratios = np.zeros((n_imgs+batch_size_tmp, 1))

            # print("\n Begin computing density ratio for images >>")
            netD.eval()
            with torch.no_grad():
                tmp = 0
                while tmp < n_imgs:
                    batch_imgs = data_iter.next() #if labels is not None, then batch_imgs is a tuple (images, labels)
                    batch_imgs = batch_imgs.type(torch.float).to(device)
                    if args.GAN == "DCGAN":
                        disc_probs = netD(batch_imgs).cpu().detach().numpy()
                        disc_probs = np.clip(disc_probs.astype(np.float), 1e-14, 1 - 1e-14)
                        density_ratios[tmp:(tmp+batch_size_tmp)] = np.divide(disc_probs, 1-disc_probs)
                    elif args.GAN == "WGANGP":
                        disc_scores_exp = np.exp(netD(batch_imgs).cpu().detach().numpy())
                        disc_scores_exp = np.clip(disc_scores_exp.astype(np.float), 1e-14)
                        density_ratios[tmp:(tmp+batch_size_tmp)] = disc_scores_exp
                    tmp += batch_size_tmp
                #end while
            # print("\n End computing density ratio.")
            return density_ratios[0:n_imgs]


    #-----------------------------------
    if args.DRE in ['disc_KeepTrain','disc_KeepTrain_MHcal']: #for DCGAN only
        assert args.GAN == "DCGAN"
        epoch_KeepTrain = 2
        batch_size_KeepTrain = 256
        Filename_KeepTrain_Disc = save_models_folder + '/ckpt_KeepTrainDisc_epoch_'+str(epoch_KeepTrain)+'_'+args.GAN +'_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed)
        if not os.path.isfile(Filename_KeepTrain_Disc):
            print("keep train the discriminator for another %d epochs" % epoch_KeepTrain)
            # keep train the discriminator
            n_test = testset.data.shape[0]
            batch_size_tmp = 500
            cal_labels = np.concatenate((np.zeros((n_test,1)), np.ones((n_test,1))), axis=0) #fake or real
            cal_imgs_fake = fn_sampleGAN(nfake=n_test, batch_size=batch_size_tmp)
            cal_imgs_real = np.transpose(testset.data, (0, 3, 1, 2))
            #standarize real images
            cal_imgs_real = cal_imgs_real/255.0
            for i in range(NC):
                cal_imgs_real[:,i,:,:] = (cal_imgs_real[:,i,:,:]-means[i])/stds[i]
            cal_imgs = np.concatenate((cal_imgs_fake, cal_imgs_real), axis=0)
            del cal_imgs_fake, cal_imgs_real; gc.collect()
            cal_dataset = IMGs_dataset(cal_imgs, cal_labels)
            cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=batch_size_KeepTrain, shuffle=True, num_workers=0)

            criterion_KeepTrain = nn.BCELoss()
            # optimizerD_KeepTrain = torch.optim.SGD(netD.parameters(), lr = 1e-3, momentum= 0.9, weight_decay=1e-4)
            optimizerD_KeepTrain = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))

            for epoch in range(epoch_KeepTrain):
                netD.train()
                train_loss = 0
                for batch_idx, (batch_train_images, batch_train_labels) in enumerate(cal_dataloader):

                    batch_train_images = batch_train_images.type(torch.float).cuda()
                    batch_train_labels = batch_train_labels.type(torch.float).cuda()

                    #Forward pass
                    outputs = netD(batch_train_images)
                    loss = criterion_KeepTrain(outputs, batch_train_labels)

                    #backward pass
                    optimizerD_KeepTrain.zero_grad()
                    loss.backward()
                    optimizerD_KeepTrain.step()

                    train_loss += loss.cpu().item()
                #end for batch_idx
                print('KeepTrain netD: [epoch %d/%d] train_loss:%.3f' % (epoch+1, epoch_KeepTrain, train_loss/(batch_idx+1)))
            #end for epoch
            # save model
            torch.save({
            'net_state_dict': netD.state_dict(),
            }, Filename_KeepTrain_Disc)
            # release memory
            batch_train_images = batch_train_images.cpu()
            batch_train_labels = batch_train_labels.cpu()
            del batch_train_images, batch_train_labels, cal_dataset, cal_dataloader; gc.collect()
            torch.cuda.empty_cache()
        else:
            print("Keep Training: load ckpt>>")
            checkpoint = torch.load(Filename_KeepTrain_Disc)
            netD.load_state_dict(checkpoint['netD_state_dict'])

        # function for computing a bunch of images
        # def comp_density_ratio(imgs, netD):
        def comp_density_ratio(imgs):
           #imgs: an numpy array
           n_imgs = imgs.shape[0]
           batch_size_tmp = DR_comp_batch_size
           dataset_tmp = IMGs_dataset(imgs)
           dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
           data_iter = iter(dataloader_tmp)
           density_ratios = np.zeros((n_imgs+batch_size_tmp, 1))

           # print("\n Begin computing density ratio for images >>")
           netD.eval()
           with torch.no_grad():
               tmp = 0
               while tmp < n_imgs:
                   batch_imgs = data_iter.next()
                   batch_imgs = batch_imgs.type(torch.float).to(device)
                   disc_probs = netD(batch_imgs).cpu().detach().numpy()
                   disc_probs = np.clip(disc_probs.astype(np.float), 1e-14, 1 - 1e-14)
                   density_ratios[tmp:(tmp+batch_size_tmp)] = np.divide(disc_probs, 1-disc_probs)
                   tmp += batch_size_tmp
               #end while
           # print("\n End computing density ratio.")
           return density_ratios[0:n_imgs]

    #-----------------------------------
    if args.DRE in ['disc_MHcal','disc_KeepTrain_MHcal']: #use the calibration method in MH-GAN to calibrate disc #for DCGAN and WGAN
        n_test = testset.data.shape[0]
        batch_size_tmp = DR_comp_batch_size
        cal_labels_fake = np.zeros((n_test,1))
        cal_labels_real = np.ones((n_test,1))
        cal_imgs_fake = fn_sampleGAN(nfake=n_test, batch_size=batch_size_tmp)
        cal_imgs_real = images_test
        #standarize real images
        cal_imgs_real = cal_imgs_real/255.0
        cal_imgs_real = (cal_imgs_real-0.5)/0.5
        dataset_fake = IMGs_dataset(cal_imgs_fake)
        dataloader_fake = torch.utils.data.DataLoader(dataset_fake, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
        dataset_real = IMGs_dataset(cal_imgs_real)
        dataloader_real = torch.utils.data.DataLoader(dataset_real, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
        del cal_imgs_fake, cal_imgs_real; gc.collect()

        # get the output of disc before the final sigmoid layer; the \tilde{D} in Eq.(4) in "Discriminator Rejection Sampling"
        # def comp_disc_scores(imgs_dataloader, netD):
        def comp_disc_scores(imgs_dataloader):
            # imgs_dataloader: the data loader for images
            n_imgs = len(imgs_dataloader.dataset)
            data_iter = iter(imgs_dataloader)
            batch_size_tmp = imgs_dataloader.batch_size
            disc_scores = np.zeros((n_imgs+batch_size_tmp, 1))
            netD.eval()
            with torch.no_grad():
                tmp = 0
                while tmp < n_imgs:
                    batch_imgs = data_iter.next()
                    batch_imgs = batch_imgs.type(torch.float).to(device)
                    disc_probs = netD(batch_imgs).cpu().detach().numpy()
                    disc_probs = np.clip(disc_probs.astype(np.float), 1e-14, 1 - 1e-14)
                    disc_scores[tmp:(tmp+batch_size_tmp)] = np.log(np.divide(disc_probs, 1-disc_probs))
                    tmp += batch_size_tmp
                #end while
            return disc_scores[0:n_imgs]

        # compute disc score of a given img which is in tensor format
        # def fn_disc_score(img, netD):
        def fn_disc_score(img):
            #img must be a tensor: 1*NC*IMG_SIZE*IMG_SIZE
            netD.eval()
            with torch.no_grad():
                img = img.type(torch.float).to(device)
                disc_prob = netD(img).cpu().detach().numpy()
                disc_prob = np.clip(disc_prob.astype(np.float), 1e-14, 1-1e-14)
                return np.log(disc_prob/(1-disc_prob))

        cal_disc_scores_fake = comp_disc_scores(dataloader_fake) #discriminator scores for fake images
        cal_disc_scores_real = comp_disc_scores(dataloader_real) #discriminator scores for real images

        # Train a logistic regression model
        X_train = np.concatenate((cal_disc_scores_fake, cal_disc_scores_real),axis=0).reshape(-1,1)
        y_train = np.concatenate((cal_labels_fake, cal_labels_real), axis=0).reshape(-1)
        #del cal_disc_scores_fake, cal_disc_scores_real; gc.collect()
        cal_logReg = LogisticRegression(solver="liblinear").fit(X_train, y_train)

        # function for computing a bunch of images
        # def comp_density_ratio(imgs, netD):
        def comp_density_ratio(imgs):
           #imgs: an numpy array
           dataset_tmp = IMGs_dataset(imgs)
           dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
           disc_scores = comp_disc_scores(dataloader_tmp)
           disc_probs = (cal_logReg.predict_proba(disc_scores))[:,1] #second column corresponds to the real class
           disc_probs = np.clip(disc_probs.astype(np.float), 1e-14, 1 - 1e-14)
           density_ratios = np.divide(disc_probs, 1-disc_probs)
           return density_ratios.reshape(-1,1)


# ###################
# # DRE based on property of a Bayes optimal classifier
# elif args.DRE == 'BayesClass': #use the property of a Bayes optimal classifier
#     ####################################
#     # Load Pre-trained GAN
#     checkpoint = torch.load(Filename_GAN)
#     if args.GAN == "DCGAN":
#         netG = cnn_generator(NGPU, args.dim_gan).to(device)
#         # netD = cnn_discriminator(True, NGPU).to(device)
#         def fn_sampleGAN(nfake, batch_size):
#             return SampDCGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
#     elif args.GAN == "WGANGP":
#         netG = cnn_generator(NGPU, args.dim_gan).to(device)
#         # netD = cnn_discriminator(False, NGPU).to(device)
#         def fn_sampleGAN(nfake, batch_size):
#             return SampWGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
#     elif args.GAN == "MMDGAN":
#         # raise Exception('Current DRE method "%s" is not suitable for MMDGAN!' % (args.DRE))
#         G_decoder = MMDGAN_Decoder(IMG_SIZE, NC, k=args.dim_gan, ngf=64, ngpu=NGPU)
#         netG = MMDGAN_G(G_decoder).to(device)
#         def fn_sampleGAN(nfake, batch_size):
#             return SampMMDGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
#     netG.load_state_dict(checkpoint['netG_state_dict'])
#
#     ####################################
#     # Train a Bayes optimal classifier
#     Filename_BayesClass = save_models_folder + '/ckpt_BayesClass_SEED_' + str(args.seed) + "_" + args.GAN + "_epoch_" + str(args.epoch_gan)
#     if not os.path.isfile(Filename_BayesClass):
#         # train a binary classifier to distinguish real and fake images
#         n_test = testset.data.shape[0]
#         batch_size_tmp = 100
#         Bayes_imgs_fake = fn_sampleGAN(nfake=n_test, batch_size=batch_size_tmp)
#         Bayes_imgs_real = np.transpose(testset.data, (0, 3, 1, 2))
#         #standarize real images
#         Bayes_imgs_real = Bayes_imgs_real/255.0
#         for i in range(NC):
#             Bayes_imgs_real[:,i,:,:] = (Bayes_imgs_real[:,i,:,:]-means[i])/stds[i]
#         Bayes_labels = np.concatenate((np.zeros((n_test,1)), np.ones((n_test,1))), axis=0)
#         dataset_tmp = IMGs_dataset(np.concatenate((Bayes_imgs_fake, Bayes_imgs_real), axis=0), Bayes_labels)
#         dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=batch_size_tmp, shuffle=True, num_workers=0)
#         del Bayes_imgs_fake, Bayes_imgs_real; gc.collect()
#
#         #--------------------------------
#         # Train a binary classifier
#         BayesClass_net = BiClassifier(ngpu=NGPU, num_channels=NC, image_size=IMG_SIZE).to(device)
#         criterion = nn.BCELoss()
#         optimizer = torch.optim.Adam(BayesClass_net.parameters(), lr=1e-3)
#         # optimizer = torch.optim.SGD(BayesClass_net.parameters(), lr = 1e-4, momentum= 0.9)
#
#         for epoch in range(100):
#             BayesClass_net.train()
#             train_loss = 0
#             for batch_idx, (images, labels) in enumerate(dataloader_tmp):
#                 images = images.type(torch.float).to(device)
#                 labels = labels.type(torch.float).to(device)
#                 optimizer.zero_grad()
#                 outputs = BayesClass_net(images)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 train_loss += loss.item()
#                 optimizer.step()
#                 if batch_idx % 10 == 0:
#                     print('BayesClass Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
#                         epoch+1, batch_idx * len(images), len(dataloader_tmp.dataset),
#                         100. * batch_idx / len(dataloader_tmp),
#                         loss.item() / len(images)))
#             #end for batch_idx
#             train_loss = train_loss / len(dataloader_tmp.dataset)
#             print('====> Epoch: {} Average loss: {:.6f}'.format(
#                   epoch+1, train_loss))
#             del images, labels, outputs; gc.collect()
#             torch.cuda.empty_cache()
#         # save model
#         torch.save({
#         'net_state_dict': BayesClass_net.state_dict(),
#         }, Filename_BayesClass)
#     else:
#         checkpoint_BayesClass = torch.load(Filename_BayesClass)
#         BayesClass_net = BiClassifier(ngpu=NGPU, num_channels=NC, image_size=IMG_SIZE)
#         BayesClass_net.load_state_dict(checkpoint_BayesClass['net_state_dict'])
#         BayesClass_net = BayesClass_net.to(device)
#
#     # function for computing a bunch of images
#     def comp_density_ratio(imgs):
#        #imgs: an numpy array
#        n_imgs = imgs.shape[0]
#        batch_size_tmp = DR_comp_batch_size
#        dataset_tmp = IMGs_dataset(imgs)
#        dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
#        data_iter = iter(dataloader_tmp)
#        density_ratios = np.zeros((n_imgs+batch_size_tmp, 1))
#
#        #print("\n Begin computing density ratio for images >>")
#        BayesClass_net.eval()
#        with torch.no_grad():
#            tmp = 0
#            while tmp < n_imgs:
#                batch_imgs = data_iter.next()
#                batch_imgs = batch_imgs.type(torch.float).to(device)
#                disc_probs = BayesClass_net(batch_imgs).cpu().detach().numpy()
#                disc_probs = np.clip(disc_probs.astype(np.float), 1e-14, 1 - 1e-14)
#                density_ratios[tmp:(tmp+batch_size_tmp)] = np.divide(disc_probs, 1-disc_probs)
#                tmp += batch_size_tmp
#            #end while
#        #print("\n End computing density ratio.")
#        return density_ratios[0:n_imgs]

###############################################################################
'''                Function for different sampling methods                  '''
###############################################################################
##########################################
# Load Pre-trained GAN
checkpoint = torch.load(Filename_GAN)
if args.GAN == "DCGAN":
    netG = cnn_generator(NGPU, args.dim_gan).to(device)
    def fn_sampleGAN(nfake, batch_size=samp_batch_size):
        return SampDCGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
elif args.GAN == "WGANGP":
    netG = cnn_generator(NGPU, args.dim_gan).to(device)
    def fn_sampleGAN(nfake, batch_size=samp_batch_size):
        return SampWGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
elif args.GAN == "MMDGAN":
    G_decoder = MMDGAN_Decoder(IMG_SIZE, NC, k=args.dim_gan, ngf=64, ngpu=NGPU)
    netG = MMDGAN_G(G_decoder).to(device)
    def fn_sampleGAN(nfake, batch_size=samp_batch_size):
        return SampMMDGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
netG.load_state_dict(checkpoint['netG_state_dict'])

##########################################
# Rejection Sampling: "Discriminator Rejection Sampling"; based on https://github.com/shinseung428/DRS_Tensorflow/blob/master/config.py
if args.Sampling == "RS":
    def fn_enhanceSampler(nfake, batch_size=samp_batch_size):
        ## Burn-in Stage
        n_burnin = 50000
        burnin_imgs = fn_sampleGAN(n_burnin, batch_size=samp_batch_size)
        burnin_densityratios = comp_density_ratio(burnin_imgs)
        print((burnin_densityratios.min(),np.median(burnin_densityratios),burnin_densityratios.max()))
        M_bar = np.max(burnin_densityratios)
        del burnin_imgs, burnin_densityratios; gc.collect()
        torch.cuda.empty_cache()
        ## Burn-in with real images
        images_train_norm = images_train/255.0
        images_train_norm = (images_train_norm-0.5)/0.5
        burnin_densityratios2 = comp_density_ratio(images_train_norm)
        print((burnin_densityratios2.min(),np.median(burnin_densityratios2),burnin_densityratios2.max()))
        ## Rejection sampling
        enhanced_imgs = np.zeros((1, NC, IMG_SIZE, IMG_SIZE)) #initilize
        pb = SimpleProgressBar()
        num_imgs = 0
        while num_imgs < nfake:
            pb.update(float(num_imgs)*100/nfake)
            batch_imgs = fn_sampleGAN(batch_size, batch_size)
            batch_ratios = comp_density_ratio(batch_imgs)
            M_bar = np.max([M_bar, np.max(batch_ratios)])
            #threshold
            if args.DRE in ["disc", "disc_KeepTrain", "disc_KeepTrain_MHcal", "disc_MHcal"]:
                epsilon_tmp = 1e-8;
                D_tilde_M = np.log(M_bar)
                batch_F = np.log(batch_ratios) - D_tilde_M - np.log(1-np.exp(np.log(batch_ratios)-D_tilde_M-epsilon_tmp))
                gamma_tmp = np.percentile(batch_F, 80) #80 percentile of each batch; follow DRS's setting
                batch_F_hat = batch_F - gamma_tmp
                batch_p = 1/(1+np.exp(-batch_F_hat))
            else:
                batch_p = batch_ratios/M_bar
            batch_psi = np.random.uniform(size=batch_size).reshape(-1,1)
            indx_accept = np.where((batch_psi<=batch_p)==True)[0]
            if len(indx_accept)>0:
                enhanced_imgs = np.concatenate((enhanced_imgs, batch_imgs[indx_accept]))
            num_imgs=len(enhanced_imgs)-1
            del batch_imgs, batch_ratios; gc.collect()
            torch.cuda.empty_cache()
        return enhanced_imgs[1:(nfake+1)] #remove the first all zero array

##########################################
# MCMC, Metropolis-Hastings algorithm: MH-GAN
elif args.Sampling == "MH":
    trainloader_MH = torch.utils.data.DataLoader(trainset, batch_size=samp_batch_size, shuffle=True, num_workers=0)
    def fn_enhanceSampler(nfake, batch_size=samp_batch_size):
        enhanced_imgs = np.zeros((1, NC, IMG_SIZE, IMG_SIZE)) #initilize
        pb = SimpleProgressBar()
        num_imgs = 0
        while num_imgs < nfake:
            data_iter = iter(trainloader_MH)
            batch_imgs_new, _ = data_iter.next()
            batch_imgs_new = batch_imgs_new.cpu().detach().numpy()
            batch_update_flags = np.zeros(batch_size) #if an img in a batch is updated during MH, replace corresponding entry with 1
            for k in tqdm(range(MH_K)):
                if not MH_mute:
                    print((k, num_imgs))
                batch_imgs_old = fn_sampleGAN(batch_size, batch_size)
                batch_U = np.random.uniform(size=batch_size).reshape(-1,1)
                batch_ratios_old = comp_density_ratio(batch_imgs_old)
                batch_ratios_new = comp_density_ratio(batch_imgs_new)
                batch_p = batch_ratios_old/(batch_ratios_new+1e-14)
                batch_p[batch_p>1]=1
                indx_accept = np.where((batch_U<=batch_p)==True)[0]
                if len(indx_accept)>0:
                    batch_imgs_new[indx_accept] = batch_imgs_old[indx_accept]
                    batch_update_flags[indx_accept] = 1 #if an img in a batch is updated during MH, replace corresponding entry with 1
            indx_updated = np.where(batch_update_flags==1)[0]
            enhanced_imgs = np.concatenate((enhanced_imgs, batch_imgs_new[indx_updated]))
            num_imgs=len(enhanced_imgs)-1
            del batch_imgs_new, batch_imgs_old; gc.collect()
            torch.cuda.empty_cache()
        return enhanced_imgs[1:(nfake+1)] #remove the first all zero array

##########################################
# Sampling-Importance Resampling
elif args.Sampling == "SIR":
   def fn_enhanceSampler(nfake, batch_size=samp_batch_size):
       enhanced_imgs = fn_sampleGAN(nfake*2, batch_size)
       enhanced_ratios = comp_density_ratio(enhanced_imgs)
       weights = enhanced_ratios / np.sum(enhanced_ratios) #normlaize to [0,1]
       resampl_indx = np.random.choice(a = np.arange(len(weights)), size = nfake, replace = True, p = weights.reshape(weights.shape[0]))
       enhanced_imgs = enhanced_imgs[resampl_indx]
       return enhanced_imgs






###############################################################################
'''                             Compute FID and IS                          '''
###############################################################################
#load pre-trained InceptionV3 (pretrained on CIFAR-10)
PreNetFIDIS = Inception3(num_classes=10, aux_logits=True, transform_input=False)
Filename_PreCNNForEvalGANs = save_models_folder + '/ckpt_PreCNNForEvalGANs_InceptionV3_epoch_200_SEED_2019_Transformation_True'
checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
PreNetFIDIS = nn.DataParallel(PreNetFIDIS).cuda()
PreNetFIDIS.load_state_dict(checkpoint_PreNet['net_state_dict'])

#----------------------------------
# IS for training data
#load training data
images_train_norm = images_train/255.0
images_train_norm = (images_train_norm-0.5)/0.5

#----------------------------------
## FID and IS for testing data
images_test_norm = images_test/255.0
images_test_norm = (images_test_norm-0.5)/0.5

if args.realdata_ISFID:
    #----------------------------------
    ## IS for training data
    (IS_train_avg, IS_train_std) = IS_RAW(PreNetFIDIS, images_train_norm , batch_size = args.IS_batch_size, splits=10, NGPU=NGPU, resize=resize)
    #----------------------------------
    ## IS for test data
    (IS_test_avg, IS_test_std) = IS_RAW(PreNetFIDIS, images_test_norm, batch_size = args.IS_batch_size, splits=10, NGPU=NGPU, resize=resize)
    #----------------------------------
    ## FID for test data
    FID_test = FID_RAW(PreNetFIDIS, images_train_norm , images_test_norm, batch_size = args.FID_batch_size, NGPU=NGPU, resize=resize)

    print("\n IS train >>> mean: %.3f, std: %.3f" % (IS_train_avg, IS_train_std))
    print("\n IS test >> mean: %.3f, std %.3f" % (IS_test_avg, IS_test_std))
    print("\n FID test: %.3f" % (FID_test))


#-----------------------------------------
# Compute average density ratio on test set to select best lambda
if args.DRE in ['DRE_F_SP', 'DRE_F_uLSIF', 'DRE_F_DSKL', 'DRE_F_BARR',
                'DRE_P_SP', 'DRE_P_uLSIF', 'DRE_P_DSKL', 'DRE_P_BARR']:
    train_densityratios = comp_density_ratio(images_train_norm)
    print("Med/Mean/STD of density ratio on training set: %.3f,%.3f,%.3f" % (np.median(train_densityratios), np.mean(train_densityratios), np.std(train_densityratios)))
    test_densityratios = comp_density_ratio(images_test_norm)
    print("Med/Mean/STD of density ratio on test set: %.3f,%.3f,%.3f" % (np.median(test_densityratios), np.mean(test_densityratios), np.std(test_densityratios)))
    ks_test = ks_2samp(train_densityratios.reshape(-1), test_densityratios.reshape(-1))
    print("Kolmogorov-Smirnov test: stat. %.4E, pval %.4E" % (ks_test.statistic, ks_test.pvalue))

#----------------------------------------
# Compute FID for fake images in NROUND rounds
if args.comp_ISFID:
    FID_EnhanceSampling_all = np.zeros(NROUND)
    IS_EnhanceSampling_all = np.zeros(NROUND)

    start = timeit.default_timer()
    for nround in range(NROUND):
        print("Round " + str(nround) + ", %s+%s+%s:" % (args.GAN, args.DRE, args.Sampling))
        timer_samp1 = timeit.default_timer()
        if args.DRE == "None" and args.Sampling == "None":
            print("Directly sample from GAN >>>")
            fake_imgs = fn_sampleGAN(NFAKE, samp_batch_size)
        else:
            print("Enhanced Sampling >>>")
            fake_imgs = fn_enhanceSampler(NFAKE, batch_size=samp_batch_size)
        timer_samp2 = timeit.default_timer()
        print("Sampling %d samples takes %f s" % (NFAKE, timer_samp2-timer_samp1))
        #----------------------------------
        ## IS for fake imgs
        print("\n Computing IS for %s+%s+%s >>> " % (args.GAN, args.DRE, args.Sampling))
        (IS_fake_avg, IS_fake_std) = IS_RAW(PreNetFIDIS, fake_imgs, batch_size = args.IS_batch_size, splits=10, NGPU=NGPU, resize=resize)
        torch.cuda.empty_cache()
        IS_EnhanceSampling_all[nround] = IS_fake_avg
        print("\n IS for %s+%s_%.3f+%s: %.3f(%.3f)" % (args.GAN, args.DRE, args.lambda_DRE, args.Sampling, IS_fake_avg, IS_fake_std))
        #----------------------------------
        ## FID for fake imgs
        print("\n Computing FID for %s+%s+%s >>> " % (args.GAN, args.DRE, args.Sampling))
        FID_EnhanceSampling_all[nround] = FID_RAW(PreNetFIDIS, images_train_norm, fake_imgs, batch_size = args.FID_batch_size, NGPU=NGPU, resize=resize)
        torch.cuda.empty_cache()
        print("\n FID for %s+%s_%.3f+%s: %.3f" % (args.GAN, args.DRE, args.lambda_DRE, args.Sampling, FID_EnhanceSampling_all[nround]))
    #end for nround
    stop = timeit.default_timer()
    print("Sampling and evaluation finished! Time elapses: {}s".format(stop - start))


    ####################################
    # Print resutls
    FID_mean = np.mean(FID_EnhanceSampling_all)
    FID_std = np.std(FID_EnhanceSampling_all)

    IS_mean = np.mean(IS_EnhanceSampling_all)
    IS_std = np.std(IS_EnhanceSampling_all)

    print("\n %s+%s_%.3f+%s" % (args.GAN, args.DRE, args.lambda_DRE, args.Sampling))
    print("\n FID mean: %.3f; std: %.3f" % (FID_mean, FID_std))
    print("\n IS: mean, %.3f; std, %.3f" % (IS_mean, IS_std))
