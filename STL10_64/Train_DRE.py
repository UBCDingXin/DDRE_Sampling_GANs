'''

Functions for Training Density-ratio model

'''


import torch
import torch.nn as nn
import numpy as np
import os
import timeit
import gc

# device_ids=[1,0]
resize_size = (224, 224)

# def adjust_learning_rate(optimizer, epoch, BASE_LR_DRE):
#     lr = BASE_LR_DRE
#     # if epoch >= 25: #SQ loss only decay at epoch 50
#     #     lr /= 10
#     # if epoch >= 50:
#     #     lr /= 10
#     # if epoch >= 150:
#     #     lr /= 10
#     if epoch >= 100:
#         lr /= 10
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
def train_DREF(NGPU, EPOCHS_DRE, BASE_LR_DRE, trainloader, net, optimizer, PreNetDRE_ResNet, PreNetDRE_fc, netG, GAN_Latent_Length, LAMBDA=10, n_classes = 10, save_models_folder = None, ResumeEpoch = 0, loss_type = "SP", device="cuda", not_decay_lr=False):

    net = net.to(device)
    netG = netG.to(device)
    PreNetDRE_ResNet = PreNetDRE_ResNet.to(device)
    PreNetDRE_fc = PreNetDRE_fc.to(device)

    if save_models_folder is not None and ResumeEpoch>0:
        print("Loading ckpt to resume training>>>")
        save_file = save_models_folder + "/DRE_checkpoint_epoch/DREF_checkpoint_epoch" + str(ResumeEpoch)+ "_LAMBDA_" + str(LAMBDA) + ".pth"
        # checkpoint = torch.load(save_file)
        checkpoint = torch.load(save_file, map_location=device)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #load d_loss and g_loss
        log_file = save_models_folder + "/DRE_checkpoint_epoch/DREF_train_loss_epoch" + str(ResumeEpoch)+ "_LAMBDA_" + str(LAMBDA) + ".npz"
        if os.path.isfile(log_file):
            avg_train_loss = list((np.load(log_file))['arr_0'])
        else:
            avg_train_loss = []
    else:
        avg_train_loss = []

    start_tmp = timeit.default_timer()
    for epoch in range(ResumeEpoch, EPOCHS_DRE):
        if not not_decay_lr:
            adjust_learning_rate(optimizer, epoch, BASE_LR_DRE)

        train_loss = 0

        for batch_idx, (batch_real_images, _) in enumerate(trainloader):

            net.train()

            BATCH_SIZE = batch_real_images.shape[0]

            batch_real_images = batch_real_images.type(torch.float).to(device)
            batch_real_images = nn.functional.interpolate(batch_real_images, size = resize_size, scale_factor=None, mode='bilinear', align_corners=False)

            PreNetDRE_ResNet.eval()
            PreNetDRE_fc.eval()
            netG.eval()
            with torch.no_grad():
                z = torch.randn(BATCH_SIZE, GAN_Latent_Length, dtype=torch.float).to(device)
                batch_fake_images = netG(z)
                batch_fake_images = batch_fake_images.detach()
                batch_fake_images = nn.functional.interpolate(batch_fake_images, size = resize_size, scale_factor=None, mode='bilinear', align_corners=False)

                _, batch_features_real_pre = PreNetDRE_ResNet(batch_real_images)
                _, batch_features_real = PreNetDRE_fc(batch_features_real_pre)
                batch_features_real = batch_features_real.detach()
                _, batch_features_fake_pre = PreNetDRE_ResNet(batch_fake_images)
                _, batch_features_fake = PreNetDRE_fc(batch_features_fake_pre)
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
                loss = - torch.mean(torch.log(DR_real + 1e-14)) + torch.mean(torch.log(DR_fake + 1e-14)) + LAMBDA * (torch.mean(DR_fake) - 1)**2
            elif loss_type == "BARR": #BARR loss proposed in "Deep density ratio estimation for change point detection"
                loss = - torch.mean(torch.log(DR_real + 1e-14)) + LAMBDA * (torch.abs(torch.mean(DR_fake)-1))

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()

            # ### debugging
            # if (batch_idx==(len(trainloader)-1)) or (batch_idx==int(len(trainloader)/2)):
            #     print("[step %d/%d] [epoch %d/%d] [training:%f/%f] [loss %.3f] [Time:%.3f]" % (batch_idx, len(trainloader),epoch+1, EPOCHS_DRE, \
            #                                                                         DR_real.mean(), DR_fake.mean(), loss.detach().cpu().item(), \
            #                                                                         timeit.default_timer()-start_tmp))
            #     del batch_fake_images; gc.collect()
            #     PreNetDRE_ResNet.eval()
            #     PreNetDRE_fc.eval()
            #     net.eval()
            #     netG.eval()
            #     with torch.no_grad():
            #         z = torch.randn(BATCH_SIZE, GAN_Latent_Length, dtype=torch.float).to(device)
            #         batch_fake_images = netG(z)
            #         batch_fake_images = batch_fake_images.detach()
            #         _, batch_features_real_pre = PreNetDRE_ResNet(batch_real_images)
            #         _, batch_features_real = PreNetDRE_fc(batch_features_real_pre)
            #         batch_features_real = batch_features_real.detach()
            #         _, batch_features_fake_pre = PreNetDRE_ResNet(batch_fake_images)
            #         _, batch_features_fake = PreNetDRE_fc(batch_features_fake_pre)
            #         batch_features_fake = batch_features_fake.detach()
            #         DR_real2 = net(batch_features_real).detach().cpu().numpy()
            #         DR_fake2 = net(batch_features_fake).detach().cpu().numpy()
            #         print("[step %d/%d] [epoch %d/%d] [Debuging:%f/%f]" % (batch_idx, len(trainloader),epoch+1, EPOCHS_DRE, DR_real2.mean(),DR_fake2.mean()))
            #     del batch_real_images, batch_fake_images; gc.collect()
        #end for batch_idx
        stop_tmp = timeit.default_timer()

        print('\r DRE_F_'+loss_type+'-LAMBDA'+str(LAMBDA)+': [epoch %d/%d] train_loss:%.3f Time:%.3f' % (epoch+1, EPOCHS_DRE, train_loss/(batch_idx+1), stop_tmp-start_tmp))

        avg_train_loss.append(train_loss/(batch_idx+1))

        # save checkpoint
        if save_models_folder is not None and (epoch+1) % 50 == 0:
            save_file = save_models_folder + "/DRE_checkpoint_epoch/"
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_ckpt = save_file + "DREF_checkpoint_epoch" + str(epoch+1) + "_LAMBDA_" + str(LAMBDA) + ".pth"
            torch.save({
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
            }, save_ckpt)

            # save loss
            log_file = save_file + "DREF_train_loss_epoch" + str(epoch+1) + "_LAMBDA_" + str(LAMBDA) + ".npz"
            np.savez(log_file, np.array(avg_train_loss))

    #end for epoch

    return net, optimizer, avg_train_loss
