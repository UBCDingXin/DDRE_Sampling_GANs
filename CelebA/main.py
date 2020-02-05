
import os
wd = './CelebA'

os.chdir(wd)
import timeit
import h5py
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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gc
from itertools import groupby
import argparse
from sklearn.linear_model import LogisticRegression
import multiprocessing
# from multiprocessing import Pool
from scipy.stats import ks_2samp

from utils import *
from models import *
from Train_DRE import *
from Train_DCGAN import train_DCGAN, SampDCGAN
from Train_WGAN import train_WGANGP, SampWGAN
from Train_SNGAN import train_SNGAN, SampSNGAN
from metrics.Inception_Score import inception_score
from metrics.fid_score import fid_score



#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
parser = argparse.ArgumentParser(description='Density-ratio based sampling for GANs')
'''Overall Settings'''
parser.add_argument('--GAN', type=str, default='DCGAN',
                    choices=['DCGAN', 'WGANGP', 'SNGAN'],
                    help='GAN model')
parser.add_argument('--DRE', type=str, default='None',
                    choices=['None', 'disc', 'disc_KeepTrain', 'disc_MHcal','DRE_F_SP'],
                    help='Density ratio estimation method') # disc: ratio=D/(1-D); disc_DRS: method in "Discriminator Rejction Sampling"; disc_MHcal: the calibration method in MH-GAN; BayesClass: a Bayes Optimal Binary classifier;
parser.add_argument('--Sampling', type=str, default='None',
                    choices=['None', 'RS', 'MH', 'SIR'],
                    help='Sampling/Resampling method for GANs; Candidiate: None, RS, MH, SIR') #RS: rejection sampling, MH: Metropolis-Hastings; SIR: Sampling-Importance Resampling
parser.add_argument('--seed', type=int, default=2019, metavar='S',
                    help='random seed (default: 1)')

''' GAN settings '''
parser.add_argument('--epoch_gan', type=int, default=60)
parser.add_argument('--lr_g_gan', type=float, default=2e-4,
                    help='learning rate for generator')
parser.add_argument('--lr_d_gan', type=float, default=2e-4,
                    help='learning rate for discriminator')
parser.add_argument('--dim_gan', type=int, default=128,
                    help='Latent dimension of GAN')
parser.add_argument('--batch_size_gan', type=int, default=128, metavar='N',
                    help='input batch size for training GAN')
parser.add_argument('--resumeTrain_gan', type=int, default=0)


'''DRE settings'''
## DRE_F_SP
parser.add_argument('--DR_Net', type=str, default='MLP5',
                    choices=['MLP3', 'MLP5', 'MLP7', 'MLP9'],
                    help='DR Model')
parser.add_argument('--PreCNN_DR', type=str, default='ResNet34',
                    choices=['ResNet34', 'ResNet50'],
                    help='Pre-trained CNN for DRE in Feature Space; Candidate: ResNetXX')
parser.add_argument('--epoch_pretrainCNN', type=int, default=100)
parser.add_argument('--transform_PreCNN_DR', action='store_true', default=True,
                    help='flip images for CNN training')
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
parser.add_argument('--DR_final_ActFn', type=str, default='ReLU',
                    help='Final layer of the Density-ratio model; Candidiate: Softplus or ReLU')
parser.add_argument('--replot_train_loss', action='store_true', default=False,
                    help='re-plot training loss')

'''Sampling and Comparing Settings'''
parser.add_argument('--samp_round', type=int, default=3)
parser.add_argument('--samp_nfake', type=int, default=50000)
parser.add_argument('--samp_batch_size', type=int, default=512)
parser.add_argument('--realdata_ISFID', action='store_true', default=False,
                    help='Print IS and FID for real data?')
parser.add_argument('--comp_ISFID', action='store_true', default=False,
                    help='Compute IS and FID for fake data?')
parser.add_argument('--IS_batch_size', type=int, default=100)
parser.add_argument('--FID_batch_size', type=int, default=100)
args = parser.parse_args()

#--------------------------------
# system
NGPU = torch.cuda.device_count()
device = torch.device("cuda")
NCPU = multiprocessing.cpu_count()
# NCPU = 0
cudnn.benchmark = True # For fast training


N_CLASS = 6
NC = 3 #number of channels
IMG_SIZE = 64
ResumeEpoch_gan = args.resumeTrain_gan
resize = (299, 299)
ADAM_beta1 = 0.5 #parameters for ADAM optimizer;
ADAM_beta2 = 0.999

#-------------------------------
# sampling parameters
NROUND = args.samp_round
NFAKE = args.samp_nfake
NPOOL_SIR_FACTOR=20
samp_batch_size = args.samp_batch_size #batch size for sampling from GAN or enhanced sampler
MH_K = 640
MH_mute = True #do not print sampling progress of MH
DR_comp_batch_size = 1000
if samp_batch_size<DR_comp_batch_size:
    DR_comp_batch_size = samp_batch_size

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed)

#-------------------------------
# output folders
save_models_folder = wd + '/Output/saved_models'
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = wd + '/Output/saved_images'
os.makedirs(save_images_folder, exist_ok=True)
save_GANimages_InTrain_folder = wd + '/Output/saved_images/'+args.GAN+'_InTrain/'
os.makedirs(save_GANimages_InTrain_folder, exist_ok=True)
save_traincurves_folder = wd + '/Output/Training_loss_fig'
os.makedirs(save_traincurves_folder, exist_ok=True)
save_temp_folder = wd + '/Output/Temp'
os.makedirs(save_temp_folder, exist_ok=True)



#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
means = (0.5, 0.5, 0.5)
stds = (0.5, 0.5, 0.5)

h5py_filename = wd+"/data/celeba_64x64.h5"

hf = h5py.File(h5py_filename, 'r')
images_train = hf['images_train'][:]
labels_train = hf['labels_train'][:]
images_valid = hf['images_valid'][:]
labels_valid = hf['labels_valid'][:]

num_classes_train = len(set(labels_train))
num_classes_valid = len(set(labels_valid))
assert num_classes_train == num_classes_valid
if num_classes_train != N_CLASS:
    print("\n The num_classes do not match! Reset!")
    N_CLASS = num_classes_train

NTRAIN = len(images_train)
NVALID = len(images_valid)

trainset = celeba_dataset(images_train, labels_train, normalize_img = True, random_transform = False, means_imgs = means, stds_imgs = stds)
testset = celeba_dataset(images_valid, labels_valid, normalize_img = True, random_transform = False, means_imgs = means, stds_imgs = stds)
hf.close()

trainloader_GAN = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_gan, shuffle=True, num_workers=NCPU)
trainloader_DRE = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_DRE, shuffle=True, num_workers=NCPU)
testloader_shuffle = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=NCPU)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=NCPU)


#----------------------------------
# FID and IS for training data
images_train_norm = images_train/255.0
images_train_norm = (images_train_norm - 0.5)/0.5

images_valid_norm = images_valid/255.0
images_valid_norm = (images_valid_norm - 0.5)/0.5

FID_path_real_stat = wd + "/metrics/fid_stats_celeba.npz"

if args.realdata_ISFID or not os.path.isfile(FID_path_real_stat):
    print("\n Start Computing IS and FID of real images >>>")
    #----------------------------------
    ## IS for training data
    (IS_train_avg, IS_train_std) = inception_score(IMGs_dataset(images_train_norm), cuda=True, batch_size=args.IS_batch_size, resize=True, splits=10, ngpu=NGPU)
    #----------------------------------
    ## IS for test data
    (IS_test_avg, IS_test_std) = inception_score(IMGs_dataset(images_valid_norm), cuda=True, batch_size=args.IS_batch_size, resize=True, splits=10, ngpu=NGPU)
    print("\r IS train >>> mean: %.3f, std: %.3f" % (IS_train_avg, IS_train_std))
    print("\r IS test >> mean: %.3f, std %.3f" % (IS_test_avg, IS_test_std))

    #----------------------------------
    ## FID for test data
    FID_test = fid_score(images_train_norm, images_valid_norm, batch_size=args.FID_batch_size, cuda=True, dims=2048, path_stat=FID_path_real_stat)
    print("\r FID test >> %.3f" % (FID_test))


#######################################################################################
'''                             Train GAN or Load Pre-trained GAN                '''
#######################################################################################
Filename_GAN = save_models_folder + '/ckpt_'+ args.GAN +'_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed)
tfboard_writer = SummaryWriter(wd+'/Output/saved_logs')

fid_stat_path = wd + "/metrics/fid_stats_celeba.npz"

print("\n Begin Training GAN:")
start = timeit.default_timer()
#-------------------------------
## DCGAN
if args.GAN == "DCGAN" and not os.path.isfile(Filename_GAN):
    #model initialization
    netG = cnn_generator(NGPU, args.dim_gan)
    netG.apply(weights_init)
    netD = cnn_discriminator(True, NGPU)
    netD.apply(weights_init)
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g_gan, betas=(ADAM_beta1, ADAM_beta2))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))

    # Start training
    netG, netD, optimizerG, optimizerD = train_DCGAN(args.epoch_gan, args.dim_gan, trainloader_GAN, netG, netD, optimizerG, optimizerD, criterion, save_GANimages_InTrain_folder, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, tfboard_writer=tfboard_writer, fid_stat_path=fid_stat_path)

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
    netG.apply(weights_init)
    netD = cnn_discriminator(False, NGPU)
    netD.apply(weights_init)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g_gan, betas=(ADAM_beta1, ADAM_beta2))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))

    # Start training
    netG, netD, optimizerG, optimizerD = train_WGANGP(args.epoch_gan, args.dim_gan, trainloader_GAN, netG, netD, optimizerG, optimizerD, save_GANimages_InTrain_folder, LAMBDA = 10, CRITIC_ITERS=5, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, tfboard_writer=tfboard_writer, fid_stat_path=fid_stat_path)
    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, Filename_GAN)
#-------------------------------
## SNGAN
elif args.GAN == "SNGAN" and not os.path.isfile(Filename_GAN):
    #model initialization
    netG = SNGAN_Generator(z_dim=args.dim_gan, ngpu = NGPU)
    netD = SNGAN_Discriminator(ngpu = NGPU)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g_gan, betas=(ADAM_beta1, ADAM_beta2))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))
    criterion = nn.BCELoss()

    # Start training
    netG, netD, optimizerG, optimizerD = train_SNGAN(args.epoch_gan, args.dim_gan, trainloader_GAN, netG, netD, optimizerG, optimizerD, criterion, save_GANimages_InTrain_folder, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, device=device, tfboard_writer=tfboard_writer, fid_stat_path=fid_stat_path)

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

    if isometric_map:
        net_name = 'PreCNNForDRE_' + Pretrained_CNN_Name #get the net's name
    else:
        net_name = 'PreCNNForEvalGANs_' + Pretrained_CNN_Name #get the net's name
    return net, net_name

#######################################################
# Construct a function to compute density-ratio
###################
# Approximate DR by NN
if args.DRE in ['DRE_F_SP']:

    DRE_loss_type = args.DRE[6:]

    def DR_net_init(DR_net_name):
        if DR_net_name in ["MLP3", "MLP5", "MLP7", "MLP9"]:
            assert args.DRE[4] == "F"
            net = DR_MLP(DR_net_name, ngpu=NGPU, final_ActFn=args.DR_final_ActFn)
        else:
            raise Exception("Select a valid density ratio model!!!")
        return net

    # Load Pre-trained GAN
    checkpoint = torch.load(Filename_GAN)
    if args.GAN in ['DCGAN' , 'WGANGP']:
        netG = cnn_generator(NGPU, args.dim_gan).to(device)
    elif args.GAN == "SNGAN":
        netG = SNGAN_Generator(z_dim=args.dim_gan, ngpu = NGPU).to(device)
    netG.load_state_dict(checkpoint['netG_state_dict'])

    # PreTrain CNN for DRE in feature space
    _, net_name = CNN_net_init(args.PreCNN_DR, N_CLASS, NGPU, isometric_map = True)
    Filename_PreCNNForDRE = save_models_folder + '/ckpt_' + net_name + '_epoch_' + str(args.epoch_pretrainCNN) + '_SEED_' + str(args.seed) + '_Transformation_' + str(args.transform_PreCNN_DR)

    #-----------------------------------------
    # Train DR model
    start = timeit.default_timer()
    # initialize DRE model
    netDR = DR_net_init(args.DR_Net)
    # netDR.apply(weights_init)
    # optimizer = torch.optim.SGD(net.parameters(), lr = args.base_lr_DRE, momentum= 0.9, weight_decay=WEIGHT_DECAY, nesterov=False)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr= args.base_lr_DRE, alpha=0.99, eps=1e-08, weight_decay=args.weightdecay_DRE, momentum=0.9, centered=False)
    # optimizer = torch.optim.Adam(netDR.parameters(), lr = args.base_lr_DRE, betas=(ADAM_beta1, ADAM_beta2), weight_decay=args.weightdecay_DRE)

    optimizer = torch.optim.Adam(netDR.parameters(), lr = args.base_lr_DRE, betas=(ADAM_beta1, ADAM_beta2), weight_decay=args.weightdecay_DRE)

    Filename_DRE = save_models_folder + '/ckpt_'+ args.DRE +'_' + args.DR_Net + '_' + args.DR_final_ActFn + '_epoch_' + str(args.epoch_DRE) + '_SEED_' + str(args.seed) + '_Lambda_' + str(args.lambda_DRE) + "_" + args.GAN + "_epoch_" + str(args.epoch_gan)

    if not os.path.isfile(Filename_DRE):
        print("\n Begin Training DR in Feature Space: >>>\n")
        ### load pretrained CNN
        PreNetDRE, _ = CNN_net_init(args.PreCNN_DR, N_CLASS, NGPU, isometric_map = True)
        checkpoint = torch.load(Filename_PreCNNForDRE)
        # PreNetDRE = PreNetDRE.to(device)
        PreNetDRE.load_state_dict(checkpoint['net_state_dict'])
        netDR, optimizer, avg_train_loss = train_DREF(NGPU, args.epoch_DRE, args.base_lr_DRE, trainloader_DRE, netDR, optimizer, PreNetDRE, netG, args.dim_gan, LAMBDA = args.lambda_DRE, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_DRE, loss_type = DRE_loss_type, device=device, not_decay_lr=args.not_decay_lr_DRE, name_gan = args.GAN)

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
        PreNetDRE.eval()
        # print("\n Begin computing density ratio for images >>")
        with torch.no_grad():
            tmp = 0
            while tmp < n_imgs:
                batch_imgs = data_iter.next()
                batch_imgs = batch_imgs.type(torch.float).to(device)
                _, batch_features = PreNetDRE(batch_imgs)
                batch_weights = netDR(batch_features)
                density_ratios[tmp:(tmp+len(batch_weights))] = batch_weights.cpu().detach().numpy()
                tmp += batch_size_tmp
            #end while
        # print("\n End computing density ratio.")
        return density_ratios[0:n_imgs]


###################
# DRE based on GAN property
elif args.DRE in ['disc', 'disc_KeepTrain', 'disc_MHcal']:
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
    elif args.GAN == "SNGAN":
        netG = SNGAN_Generator(z_dim=args.dim_gan, ngpu = NGPU).to(device)
        netD = SNGAN_Discriminator(ngpu = NGPU).to(device)
        netAux = SNGAN_Aux_Classifier(ngpu = NGPU).to(device)
        def fn_sampleGAN(nfake, batch_size):
            return SampSNGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])


    #-----------------------------------
    # train auxiliary fc layers for SNGAN
    if args.GAN in ["SNGAN"]:
        # train a auxiliary classifier on the top of the discrimnator
        epoch_aux = 100
        batch_size_aux = 256
        Filename_Aux_Disc = save_models_folder + '/ckpt_aux_epoch_'+str(epoch_aux)+'_'+args.GAN +'_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed)
        if not os.path.isfile(Filename_Aux_Disc):
            ## load NVALID fake image and NVALID held-out real images
            holdout_imgs_iter = iter(testloader_shuffle)
            cal_imgs_fake = fn_sampleGAN(nfake=NVALID, batch_size=100)
            cal_imgs_real = np.zeros((NVALID, NC, IMG_SIZE, IMG_SIZE))
            img_got = 0
            while img_got<NVALID:
                batch_imgs_tmp, _ = holdout_imgs_iter.next()
                batch_size_tmp = len(batch_imgs_tmp)
                cal_imgs_real[img_got:(img_got+batch_size_tmp)] = batch_imgs_tmp
                img_got+=batch_size_tmp
            #end while
            cal_imgs = np.concatenate((cal_imgs_fake, cal_imgs_real), axis=0)
            del cal_imgs_fake, cal_imgs_real; gc.collect()
            cal_labels = np.concatenate((np.zeros((NVALID,1)), np.ones((NVALID,1))), axis=0)
            cal_dataset = IMGs_dataset(cal_imgs, cal_labels)
            cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=batch_size_aux, shuffle=True, num_workers=0)

            criterion_aux = nn.BCELoss()
            optimizerD_aux = torch.optim.Adam(netAux.parameters(), lr = 1e-4, betas=(ADAM_beta1, ADAM_beta2))

            for epoch in range(epoch_aux):
                netAux.train()
                train_loss = 0
                for batch_idx, (batch_train_images, batch_train_labels) in enumerate(cal_dataloader):

                    batch_train_images = batch_train_images.type(torch.float).to(device)
                    batch_train_labels = batch_train_labels.type(torch.float).to(device)

                    #Forward pass
                    netD.eval()
                    _, batch_cal_features = netD(batch_train_images)
                    outputs = netAux(batch_cal_features.detach())
                    loss = criterion_aux(outputs, batch_train_labels)

                    #backward pass
                    optimizerD_aux.zero_grad()
                    loss.backward()
                    optimizerD_aux.step()

                    train_loss += loss.cpu().item()
                #end for batch_idx
                print('Aux netD: [epoch %d/%d] train_loss:%.3f' % (epoch+1, epoch_aux, train_loss/(batch_idx+1)))
            #end for epoch
            # save model
            torch.save({
            'net_state_dict': netAux.state_dict(),
            }, Filename_Aux_Disc)
            # release memory
            del batch_train_images, batch_cal_features, batch_train_labels, cal_dataset, cal_dataloader; gc.collect()
            torch.cuda.empty_cache()
        else:
            checkpoint = torch.load(Filename_Aux_Disc)
            netAux.load_state_dict(checkpoint['net_state_dict'])


    #-----------------------------------
    if args.DRE in ['disc']: #use GAN property to compute density ratio; ratio=D/(1-D); #for DCGAN, WGAN
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
                    batch_size_tmp = len(batch_imgs)
                    if args.GAN == "DCGAN":
                        disc_probs = netD(batch_imgs).cpu().detach().numpy()
                        disc_probs = np.clip(disc_probs.astype(np.float), 1e-14, 1 - 1e-14)
                        density_ratios[tmp:(tmp+batch_size_tmp)] = np.divide(disc_probs, 1-disc_probs).reshape((-1,1))
                    elif args.GAN == "WGANGP":
                        disc_scores_exp = np.exp(netD(batch_imgs).cpu().detach().numpy())
                        disc_scores_exp = np.clip(disc_scores_exp.astype(np.float), 1e-14)
                        density_ratios[tmp:(tmp+batch_size_tmp)] = disc_scores_exp.reshape((-1,1))
                    elif args.GAN == "SNGAN":
                        netAux.eval()
                        _, disc_features = netD(batch_imgs)
                        disc_probs = netAux(disc_features).cpu().detach().numpy()
                        disc_probs = np.clip(disc_probs.astype(np.float), 1e-14, 1 - 1e-14)
                        density_ratios[tmp:(tmp+batch_size_tmp)] = np.divide(disc_probs, 1-disc_probs).reshape((-1,1))
                    tmp += batch_size_tmp
                #end while
            # print("\n End computing density ratio.")
            return density_ratios[0:n_imgs]

    #-----------------------------------
    if args.DRE in ['disc_MHcal']: #use the calibration method in MH-GAN to calibrate disc #for DCGAN and WGAN
        n_test = len(images_valid)
        batch_size_tmp = DR_comp_batch_size
        cal_labels_fake = np.zeros((n_test,1))
        cal_labels_real = np.ones((n_test,1))
        cal_imgs_fake = fn_sampleGAN(nfake=n_test, batch_size=batch_size_tmp)
        #standarize real images
        cal_imgs_real = images_valid/255.0
        cal_imgs_real = (cal_imgs_real - 0.5) / 0.5
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
                    batch_size_tmp = len(batch_imgs)
                    if args.GAN in ["DCGAN", "WGANGP"]:
                        disc_probs = netD(batch_imgs).cpu().detach().numpy()
                    elif args.GAN == "SNGAN":
                        disc_probs, _ = netD(batch_imgs)
                        disc_probs = disc_probs.cpu().detach().numpy()
                    disc_probs = np.clip(disc_probs.astype(np.float), 1e-14, 1 - 1e-14).reshape((-1,1))
                    disc_scores[tmp:(tmp+batch_size_tmp)] = np.log(np.divide(disc_probs, 1-disc_probs))
                    tmp += batch_size_tmp
                #end while
            return disc_scores[0:n_imgs]

        # # compute disc score of a given img which is in tensor format
        # # def fn_disc_score(img, netD):
        # def fn_disc_score(img):
        #     #img must be a tensor: 1*NC*IMG_SIZE*IMG_SIZE
        #     netD.eval()
        #     with torch.no_grad():
        #         img = img.type(torch.float).to(device)
        #         disc_prob = netD(img).cpu().detach().numpy()
        #         disc_prob = np.clip(disc_prob.astype(np.float), 1e-14, 1-1e-14)
        #         return np.log(disc_prob/(1-disc_prob))

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


    #-----------------------------------
    if args.DRE in ['disc_KeepTrain']: #for DCGAN only
        epoch_KeepTrain = 1
        batch_size_KeepTrain = 128
        Filename_KeepTrain_Disc = save_models_folder + '/ckpt_KeepTrainDisc_epoch_'+str(epoch_KeepTrain)+'_'+args.GAN +'_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed)
        if not os.path.isfile(Filename_KeepTrain_Disc):
            print("keep train the discriminator for another %d epochs" % epoch_KeepTrain)
            # keep train the discriminator
            n_test = len(images_valid)
            batch_size_tmp = 500
            cal_labels = np.concatenate((np.zeros((n_test,1)), np.ones((n_test,1))), axis=0)
            cal_imgs_fake = fn_sampleGAN(nfake=n_test, batch_size=batch_size_tmp)
            cal_imgs_real = images_valid/255.0
            cal_imgs_real = (cal_imgs_real - 0.5) / 0.5
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
            netD.load_state_dict(checkpoint['net_state_dict'])

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
                   disc_probs = np.clip(disc_probs.astype(np.float), 1e-14, 1 - 1e-14).reshape((-1,1))
                   density_ratios[tmp:(tmp+len(disc_probs))] = np.divide(disc_probs, 1-disc_probs)
                   tmp += batch_size_tmp
               #end while
           # print("\n End computing density ratio.")
           return density_ratios[0:n_imgs]



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
elif args.GAN == "SNGAN":
    netG = SNGAN_Generator(z_dim=args.dim_gan, ngpu = NGPU).to(device)
    def fn_sampleGAN(nfake, batch_size):
        return SampSNGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
netG.load_state_dict(checkpoint['netG_state_dict'])

##########################################
# Rejection Sampling: "Discriminator Rejection Sampling"; based on https://github.com/shinseung428/DRS_Tensorflow/blob/master/config.py
if args.Sampling == "RS":
    def fn_enhanceSampler(nfake, batch_size=samp_batch_size):
        ## Burn-in Stage
        n_burnin = 50000
        burnin_imgs = fn_sampleGAN(n_burnin, batch_size=samp_batch_size)
        burnin_densityratios = comp_density_ratio(burnin_imgs)
        # print((burnin_densityratios.min(),np.median(burnin_densityratios),burnin_densityratios.max()))
        M_bar = np.max(burnin_densityratios)
        del burnin_imgs, burnin_densityratios; gc.collect()
        torch.cuda.empty_cache()
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
            if args.DRE in ["disc", "disc_MHcal"]:
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
            print("\r Already got %d fake images" % num_imgs)
            del batch_imgs_new, batch_imgs_old; gc.collect()
            torch.cuda.empty_cache()
        return enhanced_imgs[1:(nfake+1)] #remove the first all zero array

##########################################
# Sampling-Importance Resampling
elif args.Sampling == "SIR":
    def fn_enhanceSampler(nfake, batch_size=samp_batch_size):
        if NPOOL_SIR_FACTOR>5:

            enhanced_ratios = []
            for i in range(NPOOL_SIR_FACTOR):
               enhanced_imgs_i = fn_sampleGAN(nfake, batch_size)
               enhanced_ratios_i = comp_density_ratio(enhanced_imgs_i)
               enhanced_ratios.extend(list(enhanced_ratios_i))

               enhanced_imgs_i = (255*(enhanced_imgs_i*0.5+0.5)).astype(np.int32)
               datafile_tmp = save_temp_folder + "/data_SIR_" + str(i) + ".npy"
               np.save(datafile_tmp, enhanced_imgs_i)


            enhanced_ratios = np.array(enhanced_ratios)
            weights = enhanced_ratios / np.sum(enhanced_ratios) #normlaize to [0,1]
            resampl_indx = np.random.choice(a = np.arange(len(weights)), size = nfake, replace = True, p = weights.reshape(weights.shape[0]))

            for i in range(NPOOL_SIR_FACTOR):
                datafile_tmp = save_temp_folder + "/data_SIR_" + str(i) + ".npy"
                enhanced_imgs_i = np.load(datafile_tmp)
                enhanced_imgs_i = (enhanced_imgs_i/255.0 - 0.5)/0.5

                indx_i = resampl_indx[(resampl_indx>=(nfake*i))*(resampl_indx<(nfake*(i+1)))] - i*nfake

                if i == 0:
                    enhanced_imgs = enhanced_imgs_i[indx_i]
                else:
                    enhanced_imgs = np.concatenate((enhanced_imgs, enhanced_imgs_i[indx_i]), axis=0)
                os.remove(datafile_tmp)
        else:
            enhanced_imgs = fn_sampleGAN(nfake*NPOOL_SIR_FACTOR, batch_size)
            enhanced_ratios = comp_density_ratio(enhanced_imgs)
            weights = enhanced_ratios / np.sum(enhanced_ratios) #normlaize to [0,1]
            resampl_indx = np.random.choice(a = np.arange(len(weights)), size = nfake, replace = True, p = weights.reshape(weights.shape[0]))
            enhanced_imgs = enhanced_imgs[resampl_indx]

        return enhanced_imgs






###############################################################################
'''                             Compute FID and IS                          '''
###############################################################################



#-----------------------------------------
# Compute average density ratio on test set to select best lambda
if args.DRE == 'DRE_F_SP':
    train_densityratios = comp_density_ratio(images_train_norm)
    print("Med/Mean/STD of density ratio on training set: %.3f,%.3f,%.3f" % (np.median(train_densityratios), np.mean(train_densityratios), np.std(train_densityratios)))
    test_densityratios = comp_density_ratio(images_valid_norm)
    print("Med/Mean/STD of density ratio on test set: %.3f,%.3f,%.3f" % (np.median(test_densityratios), np.mean(test_densityratios), np.std(test_densityratios)))
    ks_test = ks_2samp(train_densityratios.reshape(-1), test_densityratios.reshape(-1))
    print("Kolmogorov-Smirnov test: stat. %.4E, pval %.4E" % (ks_test.statistic, ks_test.pvalue))

#----------------------------------------
# Compute FID for fake images in NROUND rounds
if args.comp_ISFID:
    #----------------------------------
    # Compute FID for fake images in NROUND
    FID_EnhanceSampling_all = np.zeros(NROUND)
    IS_EnhanceSampling_all = np.zeros(NROUND)

    print("\n Start Computing IS and FID of fake images >>>")
    start = timeit.default_timer()
    for nround in range(NROUND):
        print("\n Round " + str(nround) + ", %s+%s+%s:" % (args.GAN, args.DRE, args.Sampling))

        if args.DRE == "None" and args.Sampling == "None":
            print("\r Start sampling from GAN >>>")
            fake_imgs = fn_sampleGAN(NFAKE, samp_batch_size)
        else:
            assert args.DRE != "None"
            print("\r Start enhanced sampling >>>")
            fake_imgs = fn_enhanceSampler(NFAKE, batch_size=samp_batch_size)
        indx_tmp = np.arange(len(fake_imgs))
        np.random.shuffle(indx_tmp)
        fake_imgs = fake_imgs[indx_tmp]
        torch.cuda.empty_cache()
        # #dump sampled images to npy
        # fake_imgs = fake_imgs.astype(np.float32)
        # filename_fake_imgs = path_data_dump + "/Dump_"+str(NFAKE)+"_fake_imgs_"+args.GAN+"_epoch"+str(args.epoch_gan)+"_"+args.DRE+"_lambda"+str(args.lambda_DRE)+"_"+args.Sampling+"_Round_"+str(nround)
        # np.save(filename_fake_imgs, fake_imgs)

        #----------------------------------
        ## IS for fake imgs
        print("\r Computing IS for %s+%s+%s >>> " % (args.GAN, args.DRE, args.Sampling))
        (IS_EnhanceSampling_all[nround], _) = inception_score(IMGs_dataset(fake_imgs), cuda=True, batch_size=args.IS_batch_size, resize=True, splits=10, ngpu=NGPU)
        print("\r IS for %s+%s_%.3f+%s: %.4f" % (args.GAN, args.DRE, args.lambda_DRE, args.Sampling, IS_EnhanceSampling_all[nround]))
        #----------------------------------
        ## FID for fake imgs
        print("\r Computing FID for %s+%s+%s >>> " % (args.GAN, args.DRE, args.Sampling))
        FID_EnhanceSampling_all[nround] = fid_score(fake_imgs, images_train_norm, batch_size=args.FID_batch_size, cuda=True, dims=2048, path_stat=FID_path_real_stat)
        print("\r FID for %s+%s_%.3f+%s: %.4f" % (args.GAN, args.DRE, args.lambda_DRE, args.Sampling, FID_EnhanceSampling_all[nround]))
    #end for nround
    stop = timeit.default_timer()
    print("\r Sampling and evaluation finished! Time elapses: {}s".format(stop - start))


    ####################################
    # Print resutls for fake images
    FID_mean = np.mean(FID_EnhanceSampling_all)
    FID_std = np.std(FID_EnhanceSampling_all)

    IS_mean = np.mean(IS_EnhanceSampling_all)
    IS_std = np.std(IS_EnhanceSampling_all)

    print("\n %s+%s_%.3f+%s" % (args.GAN, args.DRE, args.lambda_DRE, args.Sampling))
    print("\n FID mean: %.3f; std: %.3f" % (FID_mean, FID_std))
    print("\n IS: mean, %.3f; std, %.3f" % (IS_mean, IS_std))
