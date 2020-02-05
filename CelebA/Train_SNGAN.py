"""
Train SNGAN

"""

import torch
from torchvision.utils import save_image
import numpy as np
from torch import autograd
import os
import timeit
import gc

from metrics.Inception_Score import inception_score
from metrics.fid_score import fid_score
from torch.utils.tensorboard import SummaryWriter
from utils import IMGs_dataset



NC=3
IMG_SIZE=64

NFAKE_FID_TRAIN = 10000
BATCH_SIZE_FID_TRAIN = 100
FID_BATCH_SIZE = 100
NGPU = torch.cuda.device_count()
N_ITER_FID = 1000 #compute FID every N_ITER_FID iterations and meanwhile output 100 sample images

############################################################################################
# Train SNGAN

def adjust_learning_rate(optimizerG, optimizerD, epoch, base_lr_g=1e-4, base_lr_d=4e-4):
    lr_g = base_lr_g
    lr_d = base_lr_d
    # if epoch >= 600:
    #     lr_g = 1e-7
    #     lr_d = 1e-7
    for param_group in optimizerG.param_groups:
        param_group['lr'] = lr_g
    for param_group in optimizerD.param_groups:
        param_group['lr'] = lr_d



def train_SNGAN(EPOCHS_GAN, GAN_Latent_Length, trainloader, netG, netD, optimizerG, optimizerD, criterion, save_SNGANimages_folder, save_models_folder = None, ResumeEpoch = 0, device="cuda", tfboard_writer=None, fid_stat_path=None):


    netG = netG.to(device)
    netD = netD.to(device)

    if save_models_folder is not None and ResumeEpoch>0:
        print("\r Resume training >>>")
        save_file = save_models_folder + "/SNGAN_checkpoint_intrain/SNGAN_checkpoint_epoch" + str(ResumeEpoch) + ".pth"
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        gen_iterations = checkpoint['gen_iterations']
    else:
        gen_iterations = 0
    #end if

    n_row=10
    z_fixed = torch.randn(n_row**2, GAN_Latent_Length, 1, 1, dtype=torch.float).to(device)

    start_tmp = timeit.default_timer()
    for epoch in range(ResumeEpoch, EPOCHS_GAN):
        adjust_learning_rate(optimizerG, optimizerD, epoch, base_lr_g=1e-4, base_lr_d=4e-4)
        for batch_idx, (batch_train_images, _) in enumerate(trainloader):

            BATCH_SIZE = batch_train_images.shape[0]
            batch_train_images = batch_train_images.type(torch.float).to(device)

            '''

            Train Discriminator: hinge loss

            '''
            d_out_real,_ = netD(batch_train_images)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            z = torch.randn(BATCH_SIZE, GAN_Latent_Length, 1, 1, dtype=torch.float).to(device)
            gen_imgs = netG(z)
            d_out_fake,_ = netD(gen_imgs.detach())
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake
            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            '''

            Train Generator: hinge loss

            '''
            z = torch.randn(BATCH_SIZE, GAN_Latent_Length, 1, 1, dtype=torch.float).to(device)
            gen_imgs = netG(z)
            g_out_fake,_ = netD(gen_imgs)

            g_loss = - g_out_fake.mean()
            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()

            gen_iterations += 1

            if gen_iterations % N_ITER_FID == 0:
                with torch.no_grad():
                    gen_imgs = netG(z_fixed)
                    gen_imgs = gen_imgs.detach()
                save_image(gen_imgs.data, save_SNGANimages_folder +'%d.png' % gen_iterations, nrow=n_row, normalize=True)

            if tfboard_writer:
                tfboard_writer.add_scalar('D loss', d_loss.item(), gen_iterations)
                tfboard_writer.add_scalar('G loss', g_loss.item(), gen_iterations)


            if gen_iterations%20 == 0 and gen_iterations%N_ITER_FID != 0:
                print ("SNGAN: [Iter %d/%d] [Epoch %d/%d] [D loss: %.4f] [G loss: %.4f] [Time: %.4f]" % (gen_iterations, len(trainloader)*EPOCHS_GAN, epoch+1, EPOCHS_GAN, d_loss.item(), g_loss.item(), timeit.default_timer()-start_tmp))
            elif gen_iterations%N_ITER_FID == 0: #compute inception score
                del gen_imgs, batch_train_images; gc.collect()
                fake_images = np.zeros((NFAKE_FID_TRAIN+BATCH_SIZE_FID_TRAIN, NC, IMG_SIZE, IMG_SIZE))
                netG.eval()
                with torch.no_grad():
                    tmp = 0
                    while tmp < NFAKE_FID_TRAIN:
                        z = torch.randn(BATCH_SIZE_FID_TRAIN, GAN_Latent_Length, 1, 1, dtype=torch.float).to(device)
                        batch_fake_images = netG(z)
                        fake_images[tmp:(tmp+BATCH_SIZE_FID_TRAIN)] = batch_fake_images.cpu().detach().numpy()
                        tmp += BATCH_SIZE_FID_TRAIN
                fake_images = fake_images[0:NFAKE_FID_TRAIN]
                del batch_fake_images; gc.collect()

                FID = fid_score(fake_images, fid_stat_path, batch_size=FID_BATCH_SIZE, cuda=True, dims=2048)

                if tfboard_writer:
                    tfboard_writer.add_scalar('Frechet Inception Distance', FID, gen_iterations)

                print ("SNGAN: [Iter %d/%d] [Epoch %d/%d] [D loss: %.3f] [G loss: %.3f] [Time: %.3f] [FID: %.3f]" % (gen_iterations, len(trainloader)*EPOCHS_GAN, epoch+1, EPOCHS_GAN, d_loss.item(), g_loss.item(), timeit.default_timer()-start_tmp, FID))


        if save_models_folder is not None and (epoch+1) % 10 == 0:
            save_file = save_models_folder + "/SNGAN_checkpoint_intrain/"
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = save_file + "SNGAN_checkpoint_epoch" + str(epoch+1) + ".pth"
            torch.save({
                    'gen_iterations': gen_iterations,
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict()
            }, save_file)
    #end for epoch

    return netG, netD, optimizerG, optimizerD


def SampSNGAN(netG, GAN_Latent_Length = 128, NFAKE = 10000, batch_size = 100, device="cuda"):
    #netD: whether assign weights to fake images via inversing f function (the f in f-GAN)
    raw_fake_images = np.zeros((NFAKE+batch_size, NC, IMG_SIZE, IMG_SIZE))
    netG=netG.to(device)
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            z = torch.randn(batch_size, GAN_Latent_Length, 1, 1, dtype=torch.float).to(device)
            batch_fake_images = netG(z)
            raw_fake_images[tmp:(tmp+batch_size)] = batch_fake_images.cpu().detach().numpy()
            tmp += batch_size

    #remove extra entries
    raw_fake_images = raw_fake_images[0:NFAKE]

    return raw_fake_images
