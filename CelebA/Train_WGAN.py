"""
Train WGAN and WGAN-GP

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
# Train WGANs
############################################################################################
#-------------------------------------------------------------------------------------------
#  Train WGAN-GP

## function for computing gradient penalty
def calc_gradient_penalty_WGAN(netD, real_data, fake_data, LAMBDA=10, device="cuda"):
    #LAMBDA: Gradient penalty lambda hyperparameter
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size(0), NC*IMG_SIZE*IMG_SIZE)
    alpha = alpha.view(real_data.size(0), NC, IMG_SIZE, IMG_SIZE)
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device) ,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def train_WGANGP(EPOCHS_GAN, GAN_Latent_Length, trainloader, netG, netD, optimizerG, optimizerD, save_GANimages_folder, LAMBDA = 10, CRITIC_ITERS=5, save_models_folder = None, ResumeEpoch = 0, device="cuda", tfboard_writer=None, fid_stat_path=None):

    assert fid_stat_path is not None # must specify stat file for computing FID score

    netG = netG.to(device)
    netD = netD.to(device)

    if save_models_folder is not None and ResumeEpoch>0:
        save_file = save_models_folder + "/WGANGP_checkpoint_intrain/WGANGP_checkpoint_epoch" + str(ResumeEpoch) + ".pth"
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

        data_iter = iter(trainloader)
        batch_idx = 0
        while (batch_idx < len(trainloader)):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            for batch_idx_G in range(CRITIC_ITERS):

                if batch_idx == len(trainloader):
                    break

                (batch_train_images,_) = data_iter.next()
                batch_idx += 1

                BATCH_SIZE = batch_train_images.shape[0]
                batch_train_images = batch_train_images.type(torch.float).to(device)
                netD.zero_grad()
                # Train D on real
                D_real = netD(batch_train_images)
                D_real = D_real.mean()
                # Generate fake images
                z = torch.randn(BATCH_SIZE, GAN_Latent_Length, 1, 1, dtype=torch.float).to(device)
                gen_imgs = netG(z)
                # Train D on fake
                D_fake = netD(gen_imgs)
                D_fake = D_fake.mean()
                # train with gradient penalty
                gradient_penalty = calc_gradient_penalty_WGAN(netD, batch_train_images.data, gen_imgs.data, LAMBDA, device=device)

                D_cost = D_fake - D_real + gradient_penalty
                D_cost.backward()
                optimizerD.step()

                Wasserstein_D = D_real.cpu().item() - D_fake.cpu().item()

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()
            # Generate fake images
            z = torch.randn(BATCH_SIZE, GAN_Latent_Length, 1, 1, dtype=torch.float).to(device)
            gen_imgs = netG(z)
            G_cost = - netD(gen_imgs).mean()
            G_cost.backward()
            optimizerG.step()
            gen_iterations += 1


            if gen_iterations % N_ITER_FID == 0:
                with torch.no_grad():
                    gen_imgs = netG(z_fixed)
                    gen_imgs = gen_imgs.detach()
                save_image(gen_imgs.data, save_GANimages_folder +'%d.png' % gen_iterations, nrow=n_row, normalize=True)

            if tfboard_writer:
                tfboard_writer.add_scalar('D loss', D_cost.item(), gen_iterations)
                tfboard_writer.add_scalar('G loss', G_cost.item(), gen_iterations)
                tfboard_writer.add_scalar('W distance', Wasserstein_D, gen_iterations)

            if gen_iterations%20 == 0 and gen_iterations%N_ITER_FID != 0:
                print("WGAN: [Step %d/%d] [Epoch %d/%d] [G_iter %d] [D loss: %.3f] [G loss: %.3f][W Dist: %.3f][Time: %.3f]" % (batch_idx+1, len(trainloader), epoch +1, EPOCHS_GAN, gen_iterations, D_cost.item(), G_cost.item(), Wasserstein_D, timeit.default_timer()-start_tmp))

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

                print("WGAN: [Step %d/%d] [Epoch %d/%d] [G_iter %d] [D loss: %.3f] [G loss: %.3f][W Dist: %.3f][Time: %.3f] [FID: %.3f]" % (batch_idx+1, len(trainloader), epoch +1, EPOCHS_GAN, gen_iterations, D_cost.item(), G_cost.item(), Wasserstein_D, timeit.default_timer()-start_tmp, FID))

        #end for batch_idx

        if save_models_folder is not None and (epoch+1) % 20 == 0:
            save_file = save_models_folder + "/WGANGP_checkpoint_intrain/"
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = save_file + "WGANGP_checkpoint_epoch" + str(epoch+1) + ".pth"
            torch.save({
                    'gen_iterations': gen_iterations,
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict()
            }, save_file)

    #end for epoch

    return netG, netD, optimizerG, optimizerD

#--------------------------------------------------------------------------------
# Sample WGAN and WGAN-GP
def SampWGAN(netG, GAN_Latent_Length = 100, NFAKE = 10000, batch_size = 500, device="cuda"):
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

    #remove unused entry and extra samples
    raw_fake_images = raw_fake_images[0:NFAKE]
    return raw_fake_images
