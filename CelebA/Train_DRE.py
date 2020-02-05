'''

Functions for Training Density-ratio model

'''


import torch
import torch.nn as nn
import numpy as np
import os
import timeit

# def adjust_learning_rate(optimizer, epoch, BASE_LR_DRE):
#     lr = BASE_LR_DRE #1e-4
#
#     if epoch >= 50:
#         lr /= 10
#
#     if epoch >= 100:
#         lr /= 10
#
#     if epoch >= 150:
#         lr /= 10
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, BASE_LR_DRE):
    lr = BASE_LR_DRE #1e-4

    if epoch >= 30:
        lr /= 10

    if epoch >= 70:
        lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



###############################################################################
# DRE in Feature Space
def train_DREF(NGPU, EPOCHS_DRE, BASE_LR_DRE, trainloader, net, optimizer, PreNetDRE, netG, GAN_Latent_Length, LAMBDA=0.1, save_models_folder = None, ResumeEpoch = 0, loss_type = "SP", device="cuda", not_decay_lr=False, name_gan = None):

    net = net.to(device)
    PreNetDRE = PreNetDRE.to(device)

    if save_models_folder is not None and ResumeEpoch>0:
        print("Loading ckpt to resume training>>>")
        save_file = save_models_folder + "/DRE_checkpoint_epoch/DREF_checkpoint_epoch" + str(ResumeEpoch) + ".pth"
        checkpoint = torch.load(save_file)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        #load d_loss and g_loss
        log_file = save_models_folder + "/DRE_checkpoint_epoch/DREF_train_loss_epoch" + str(ResumeEpoch) + ".npz"
        if os.path.isfile(log_file):
            avg_train_loss = list(np.load(log_file))
        else:
            avg_train_loss = []
    else:
        avg_train_loss = []


    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1)
    start_tmp = timeit.default_timer()
    for epoch in range(ResumeEpoch, EPOCHS_DRE):
        if not not_decay_lr:
            adjust_learning_rate(optimizer, epoch, BASE_LR_DRE)

        train_loss = 0

        for batch_idx, (batch_real_images, batch_real_labels) in enumerate(trainloader):

            net.train()

            BATCH_SIZE = batch_real_images.shape[0]

            batch_real_images = batch_real_images.type(torch.float).to(device)

            PreNetDRE.eval()
            netG.eval()
            with torch.no_grad():
                if name_gan is None or name_gan[0]!="c":
                    z = torch.randn(BATCH_SIZE, GAN_Latent_Length, 1, 1, dtype=torch.float).to(device)
                    batch_fake_images = netG(z)
                elif name_gan[0]=="c":
                    z = torch.randn(BATCH_SIZE, GAN_Latent_Length, dtype=torch.float).to(device)
                    batch_real_labels = batch_real_labels.type(torch.long).to(device)
                    batch_fake_images = netG(z,batch_real_labels)
                batch_fake_images = batch_fake_images.detach()
                _, batch_features_real = PreNetDRE(batch_real_images)
                batch_features_real = batch_features_real.detach()
                _, batch_features_fake = PreNetDRE(batch_fake_images)
                batch_features_fake = batch_features_fake.detach()

            #Forward pass
            DR_real = net(batch_features_real)
            DR_fake = net(batch_features_fake)

            if loss_type == "SP":
                #Softplus loss
                softplus_fn = torch.nn.Softplus(beta=1,threshold=20)
                sigmoid_fn = torch.nn.Sigmoid()
                SP_div = torch.mean(sigmoid_fn(DR_fake) * DR_fake) - torch.mean(softplus_fn(DR_fake)) - torch.mean(sigmoid_fn(DR_real))
                #penalty term: prevent assigning zero to all fake image
                penalty = LAMBDA * (torch.mean(DR_fake) - 1)**2
                loss = SP_div + penalty
            elif loss_type == "uLSIF": #uLSIF loss, also known as SQ loss
                #SQ loss
                loss = 0.5* torch.mean(DR_fake**2) - torch.mean(DR_real) + LAMBDA * (torch.mean(DR_fake) - 1)**2
            elif loss_type == "DSKL": #DSKL loss proposed in "Deep density ratio estimation for change point detection"
                # loss = - torch.mean(torch.log(DR_fake + 1e-14)) + torch.mean(torch.log(DR_real + 1e-14)) + LAMBDA * (torch.mean(DR_fake) - 1)**2
                loss = - torch.mean(torch.log(DR_real + 1e-14)) + torch.mean(torch.log(DR_fake + 1e-14)) + LAMBDA * (torch.mean(DR_fake) - 1)**2
            elif loss_type == "BARR": #BARR loss proposed in "Deep density ratio estimation for change point detection"
                # loss = - torch.mean(torch.log(DR_fake + 1e-14)) + LAMBDA * (torch.abs(torch.mean(DR_real)-1))
                loss = - torch.mean(torch.log(DR_real + 1e-14)) + LAMBDA * (torch.abs(torch.mean(DR_fake)-1))

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
        print('DRE_F_'+loss_type+'-LAMBDA'+str(LAMBDA)+': [epoch %d/%d] [train_loss %.3f] [Time %.3f]' % (epoch+1, EPOCHS_DRE, train_loss/(batch_idx+1), timeit.default_timer()-start_tmp))

        avg_train_loss.append(train_loss/(batch_idx+1))

        # save checkpoint
        if save_models_folder is not None and (epoch+1) % 50 == 0:
            save_file = save_models_folder + "/DRE_checkpoint_epoch/"
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = save_file + "DREF_checkpoint_epoch" + str(epoch+1) + ".pth"
            torch.save({
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
            }, save_file)

            # save loss
            log_file = save_file + "DREF_train_loss_epoch" + str(epoch+1) + ".npz"
            np.savez(log_file, np.array(avg_train_loss))

    #end for epoch

    return net, optimizer, avg_train_loss
