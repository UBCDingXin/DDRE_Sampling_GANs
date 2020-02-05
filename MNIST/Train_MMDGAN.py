"""
Train MMD-GAN
WITH samplers

"""


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import os
import timeit
from models import *
import numpy as np

###########################################################################
# MMD stuffs

min_var_est = 1e-8

# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss

# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = (alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c)
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = (alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c)
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = (alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c)
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = (alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c)
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _mix_rbf_kernel(X, Y, sigma_list):
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)

def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)

def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)                     # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X      # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y      # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum  = (K_XY ** 2).sum()                    # \| K_{XY} \|_F^2

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    var_est = (
        2.0 / (m**2 * (m - 1.0)**2) * (2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
        - (4.0*m - 6.0) / (m**3 * (m - 1.0)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0*(m - 2.0) / (m**3 * (m - 1.0)**2) * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0*(m - 3.0) / (m**3 * (m - 1.0)**2) * (K_XY_2_sum) - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0 / (m**3 * (m - 1.0)) * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0))
        )
    return mmd2, var_est


############################################################################################
# util

def normalize(x, dim=1):
    return x.div(x.norm(2, dim=dim).expand_as(x))

def match(x, y, dist):
    '''
    Computes distance between corresponding points points in `x` and `y`
    using distance `dist`.
    '''
    if dist == 'L2':
        return (x - y).pow(2).mean()
    elif dist == 'L1':
        return (x - y).abs().mean()
    elif dist == 'cos':
        x_n = normalize(x)
        y_n = normalize(y)
        return 2 - (x_n).mul(y_n).mean()
    else:
        assert dist == 'none'

def grad_norm(m, norm_type=2):
    total_norm = 0.0
    for p in m.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

############################################################################################
# Train MMDGAN


def train_MMDGAN(EPOCHS_GAN, GAN_Latent_Length, trainloader, netG, netD, one_sided, optimizerG, optimizerD, save_GANimages_InTrain_folder, CRITIC_ITERS=5, NC = 1, IMG_SIZE = 28, save_models_folder = None, ResumeEpoch = 0):

    # sigma for MMD
    base = 1.0
    sigma_list = [1, 2, 4, 8, 16]
    sigma_list = [sigma / base for sigma in sigma_list]

    # put variable into cuda device
    fixed_noise = torch.cuda.FloatTensor(100, GAN_Latent_Length).normal_(0, 1)
    one = torch.cuda.FloatTensor([1])
    mone = one * -1

    fixed_noise = Variable(fixed_noise, requires_grad=False)

#    lambda_MMD = 1.0
    lambda_AE_X = 8.0
    lambda_AE_Y = 8.0
    lambda_rg = 16.0

    netG = netG.cuda()
    netD = netD.cuda()

    if save_models_folder is not None and ResumeEpoch>0:
        save_file = save_models_folder + "/MMDGAN_checkpoint_intrain/MMDGAN_checkpoint_epoch" + str(ResumeEpoch) + ".pth"
        checkpoint = torch.load(save_file)
        EPOCHS_GAN = EPOCHS_GAN - ResumeEpoch
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        gen_iterations = checkpoint['gen_iterations']
    else:
        gen_iterations = 0
    #end if

    time = timeit.default_timer()
    # gen_iterations = 0
    for t in range(EPOCHS_GAN):
        data_iter = iter(trainloader)
        i = 0
        while (i < len(trainloader)):
#            data_iter = iter(trainloader)
            # ---------------------------
            # Optimize over NetD
            # ---------------------------
            for p in netD.parameters():
                p.requires_grad = True

            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 50 #default 100 (100 does not work)
                Giters = 1
            else:
                Diters = 5 #default 5
                Giters = 1

            for j in range(Diters):
                if gen_iterations < 25 or gen_iterations % 500 == 0:
                    print("Warm up or Improving NetD ..." + "("+str(j) + ","  + str(gen_iterations) + ")" )

                if i == len(trainloader):
                    break

                # clamp parameters of NetD encoder to a cube
                # do not clamp paramters of NetD decoder!!!
                for p in netD.encoder.parameters():
                    p.data.clamp_(-0.01, 0.01)

                data = data_iter.next()
                i += 1
                netD.zero_grad()

                x_cpu, _ = data
                x = Variable(x_cpu.cuda())
                batch_size = x.size(0)

                f_enc_X_D, f_dec_X_D = netD(x)

                noise = torch.cuda.FloatTensor(batch_size, GAN_Latent_Length).normal_(0, 1)
                noise = Variable(noise, volatile=True)  # total freeze netG
                y = Variable(netG(noise).data)

                f_enc_Y_D, f_dec_Y_D = netD(y)

                # compute biased MMD2 and use ReLU to prevent negative value
                mmd2_D = mix_rbf_mmd2(f_enc_X_D, f_enc_Y_D, sigma_list)
                mmd2_D = F.relu(mmd2_D)

                # compute rank hinge loss
                #print('f_enc_X_D:', f_enc_X_D.size())
                #print('f_enc_Y_D:', f_enc_Y_D.size())
                one_side_errD = one_sided(f_enc_X_D.mean(0) - f_enc_Y_D.mean(0))

                # compute L2-loss of AE
                L2_AE_X_D = match(x.view(batch_size, -1), f_dec_X_D, 'L2')
                L2_AE_Y_D = match(y.view(batch_size, -1), f_dec_Y_D, 'L2')

                errD = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD - lambda_AE_X * L2_AE_X_D - lambda_AE_Y * L2_AE_Y_D
                errD.backward(mone)
                optimizerD.step()


            # ---------------------------
            # Optimize over NetG
            # ---------------------------
            for p in netD.parameters():
                p.requires_grad = False

            for j in range(Giters):
                if i == len(trainloader):
                    break

                data = data_iter.next()
                i += 1
                netG.zero_grad()

                x_cpu, _ = data
                x = Variable(x_cpu.cuda())
                batch_size = x.size(0)

                f_enc_X, f_dec_X = netD(x)

                noise = torch.cuda.FloatTensor(batch_size, GAN_Latent_Length).normal_(0, 1)
                noise = Variable(noise)
                y = netG(noise)

                f_enc_Y, f_dec_Y = netD(y)

                # compute biased MMD2 and use ReLU to prevent negative value
                mmd2_G = mix_rbf_mmd2(f_enc_X, f_enc_Y, sigma_list)
                mmd2_G = F.relu(mmd2_G)

                # compute rank hinge loss
                one_side_errG = one_sided(f_enc_X.mean(0) - f_enc_Y.mean(0))

                errG = torch.sqrt(mmd2_G) + lambda_rg * one_side_errG
                errG.backward(one)
                optimizerG.step()

                gen_iterations += 1

            run_time = (timeit.default_timer() - time) / 60.0
            if gen_iterations > 1:
                print('[%3d/%3d][%3d/%3d] [%5d] (%.2f m) MMD2_D %.6f hinge %.6f L2_AE_X %.6f L2_AE_Y %.6f loss_D %.6f Loss_G %.6f f_X %.6f f_Y %.6f |gD| %.4f |gG| %.4f'
                  % (t, EPOCHS_GAN, i, len(trainloader), gen_iterations, run_time,
                     mmd2_D.item(), one_side_errD.item(),
                     L2_AE_X_D.item(), L2_AE_Y_D.item(),
                     errD.item(), errG.item(),
                     f_enc_X_D.mean().item(), f_enc_Y_D.mean().item(),
                     grad_norm(netD), grad_norm(netG)))

            if gen_iterations % 100 == 0:
                y_fixed = netG(fixed_noise)
                y_fixed.data = y_fixed.data.mul(0.5).add(0.5)
                f_dec_X_D = f_dec_X_D.view(f_dec_X_D.size(0), NC, IMG_SIZE, IMG_SIZE)
                f_dec_X_D.data = f_dec_X_D.data.mul(0.5).add(0.5)
                vutils.save_image(y_fixed.data, save_GANimages_InTrain_folder +'fake_samples_'+'%d.png' % gen_iterations, nrow=10, normalize=True)
                vutils.save_image(f_dec_X_D.data, save_GANimages_InTrain_folder +'decode_samples_'+'%d.png' % gen_iterations, nrow=10, normalize=True)

        #save models every 100 epochs
        if save_models_folder is not None and (t+1) % 100 == 0:
            save_file = save_models_folder + "/MMDGAN_checkpoint_intrain/"
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = save_file + "MMDGAN_checkpoint_epoch" + str(t+1+ResumeEpoch) + ".pth"
            torch.save({
                    'epoch': t,
                    'gen_iterations': gen_iterations,
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict()
            }, save_file)

    return netG, netD, optimizerG, optimizerD



def SampMMDGAN(netG, GAN_Latent_Length = 128, NFAKE = 10000, batch_size = 500):
    raw_fake_images = np.zeros((NFAKE+batch_size, 1, 28, 28))
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            z = torch.randn(batch_size, GAN_Latent_Length, dtype=torch.float).cuda()
            batch_fake_images = netG(z)
            raw_fake_images[tmp:(tmp+batch_size)] = batch_fake_images.cpu().detach().numpy()
            tmp += batch_size

    #remove unused entry and extra samples
    raw_fake_images = raw_fake_images[0:NFAKE]

    return raw_fake_images
