'''

Functions for Training Density-ratio model

'''


import torch
import torch.nn as nn
import numpy as np
import os


def adjust_learning_rate(optimizer, epoch, BASE_LR_DRE, EPOCHS_DRE, decay_epochs=400):
   lr = BASE_LR_DRE
   for i in range(EPOCHS_DRE//decay_epochs):
       if epoch >= (i+1)*decay_epochs:
           lr /= 10
   for param_group in optimizer.param_groups:
       param_group['lr'] = lr

###############################################################################
# Train the DR net: we have a proposal distribution
def train_DRE_prop(net, optimizer, BASE_LR_DRE, EPOCHS_DRE, LAMBDA, tar_dataloader, prop_dataloader, PreNetDRE = None, decay_lr = False, decay_epochs=100, loss_type="SP", save_models_folder = None, ResumeEpoch=0, NGPU=1, device="cuda"):
    #loss_type: SP, uLSIF, DSKL, BARR
    net = net.to(device)
    if PreNetDRE is not None:
        PreNetDRE = PreNetDRE.to(device)

    if save_models_folder is not None and ResumeEpoch>0:
        print("Loading ckpt to resume training>>>")
        save_file = save_models_folder + "/DRE_prop_checkpoint/ckpt_DRE_"+loss_type+"_LAMBDA_"+str(LAMBDA)+"_checkpoint_epoch" + str(ResumeEpoch) + ".pth"
        checkpoint = torch.load(save_file)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        #load d_loss and g_loss
        log_file = save_models_folder + "/DRE_prop_checkpoint/DRE_"+loss_type+"_LAMBDA_"+str(LAMBDA)+"_train_loss_epoch" + str(ResumeEpoch) + ".npz"
        if os.path.isfile(log_file):
            avg_train_loss = list(np.load(log_file))
        else:
            avg_train_loss = []
    else:
        avg_train_loss = []

    for epoch in range(ResumeEpoch, EPOCHS_DRE):
        if decay_lr:
            adjust_learning_rate(optimizer, epoch, BASE_LR_DRE, EPOCHS_DRE, decay_epochs)
        train_loss = 0

        tar_data_iter = iter(tar_dataloader)
        prop_data_iter = iter(prop_dataloader)
        assert len(tar_dataloader) == len(prop_dataloader)

        batch_idx = 0
        while batch_idx<len(tar_dataloader):

            net.train()

            batch_tar_samples, _ = tar_data_iter.next()
            batch_prop_samples, _ = prop_data_iter.next()
            batch_tar_samples = batch_tar_samples.type(torch.float).to(device)
            batch_prop_samples = batch_prop_samples.type(torch.float).to(device)

            if PreNetDRE is None:
                #Forward pass
                DR_tar = net(batch_tar_samples)
                DR_prop = net(batch_prop_samples)

            else:
                PreNetDRE.eval()
                with torch.no_grad():
                    _, batch_tar_features = PreNetDRE(batch_tar_samples)
                    batch_tar_features = batch_tar_features.detach()
                    _, batch_prop_features = PreNetDRE(batch_prop_samples)
                    batch_prop_features = batch_prop_features.detach()
                #Forward pass
                DR_tar = net(batch_tar_features)
                DR_prop = net(batch_prop_features)

            if loss_type == "SP":
                #Softplus loss
                softplus_fn = torch.nn.Softplus(beta=1,threshold=20)
                sigmoid_fn = torch.nn.Sigmoid()
                SP_div = torch.mean(sigmoid_fn(DR_prop) * DR_prop) - torch.mean(softplus_fn(DR_prop)) - torch.mean(sigmoid_fn(DR_tar))
                #penalty term: prevent assigning zero to all fake image
                penalty = LAMBDA * (torch.mean(DR_prop) - 1)**2
                loss = SP_div + penalty
            elif loss_type == "uLSIF": #uLSIF loss, also known as SQ loss
               #SQ loss
                loss = 0.5* torch.mean(DR_prop**2) - torch.mean(DR_tar) + LAMBDA * (torch.mean(DR_prop) - 1)**2
            elif loss_type == "DSKL": #DSKL loss proposed in "Deep density ratio estimation for change point detection"
                loss = - torch.mean(torch.log(DR_tar + 1e-14)) + torch.mean(torch.log(DR_prop + 1e-14)) + LAMBDA * (torch.mean(DR_prop) - 1)**2
            elif loss_type == "BARR": #BARR loss proposed in "Deep density ratio estimation for change point detection"
                loss = - torch.mean(torch.log(DR_tar + 1e-14)) + LAMBDA * (torch.abs(torch.mean(DR_prop)-1))

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
            batch_idx+=1

        print("DRE+%s+lambda%.3f: [epoch %d/%d] train_loss:%.3f" % ( loss_type, LAMBDA, epoch+1, EPOCHS_DRE, train_loss/(batch_idx+1)))

        avg_train_loss.append(train_loss/(batch_idx+1))

        # save checkpoint
        if save_models_folder is not None and (epoch+1) % 50 == 0:
            save_file = save_models_folder + "/DRE_prop_checkpoint/"
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = save_file + "ckpt_DRE_" + loss_type + "_LAMBDA_" + str(LAMBDA) +  "_checkpoint_epoch" + str(epoch+1) + ".pth"
            torch.save({
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
            }, save_file)

            # save loss
            log_file = save_models_folder + "/DRE_prop_checkpoint/DRE_"+loss_type+"_LAMBDA_"+str(LAMBDA)+"_train_loss_epoch" + str(epoch+1) + ".npz"
            np.savez(log_file, np.array(avg_train_loss))
    #end for epoch

    return net, optimizer, avg_train_loss


###############################################################################
# Train the DR net: we have a trained GAN
def train_DRE_GAN(net, optimizer, BASE_LR_DRE, EPOCHS_DRE, LAMBDA, tar_dataloader, netG, dim_gan, PreNetDRE = None, decay_lr=False, decay_epochs=400, loss_type="SP", save_models_folder = None, ResumeEpoch=0, NGPU=1, device="cuda"):
    #loss_type: SP, uLSIF, DSKL, BARR
    net = net.to(device)
    if PreNetDRE is not None:
        PreNetDRE = PreNetDRE.to(device)

    if save_models_folder is not None and ResumeEpoch>0:
        print("Loading ckpt to resume training>>>")
        save_file = save_models_folder + "/DRE_GAN_checkpoint/ckpt_DRE_"+loss_type+"_LAMBDA_"+str(LAMBDA)+"_checkpoint_epoch" + str(ResumeEpoch) + ".pth"
        checkpoint = torch.load(save_file)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        #load d_loss and g_loss
        log_file = save_models_folder + "/DRE_GAN_checkpoint/DRE_"+loss_type+"_LAMBDA_"+str(LAMBDA)+"_train_loss_epoch" + str(ResumeEpoch) + ".npz"
        if os.path.isfile(log_file):
            avg_train_loss = list(np.load(log_file))
        else:
            avg_train_loss = []
    else:
        avg_train_loss = []

    avg_train_loss = []

    for epoch in range(ResumeEpoch, EPOCHS_DRE):
        if decay_lr:
            adjust_learning_rate(optimizer, epoch, BASE_LR_DRE, EPOCHS_DRE, decay_epochs)

        train_loss = 0

        tar_data_iter = iter(tar_dataloader)

        batch_idx = 0
        while batch_idx<len(tar_dataloader):
            net.train()

            batch_tar_samples, _ = tar_data_iter.next()
            batch_tar_samples = batch_tar_samples.type(torch.float).to(device)

            BATCH_SIZE = batch_tar_samples.shape[0]

            if PreNetDRE is None:
                netG.eval()
                with torch.no_grad():
                    z = torch.randn(BATCH_SIZE, dim_gan, dtype=torch.float).to(device)
                    batch_prop_samples = netG(z)
                    batch_prop_samples = batch_prop_samples.detach()
                #Forward pass
                DR_tar = net(batch_tar_samples)
                DR_prop = net(batch_prop_samples)

                # print("training:%f/%f" % (DR_tar.mean(),DR_prop.mean()))
            else:
                netG.eval()
                PreNetDRE.eval()
                with torch.no_grad():
                    z = torch.randn(BATCH_SIZE, dim_gan, dtype=torch.float).to(device)
                    batch_prop_samples = netG(z)
                    batch_prop_samples = batch_prop_samples.detach()
                    _, batch_prop_features = PreNetDRE(batch_prop_samples)
                    batch_prop_features = batch_prop_features.detach()
                    _, batch_tar_features = PreNetDRE(batch_tar_samples)
                    batch_tar_features = batch_tar_features.detach()

                #Forward pass
                DR_tar = net(batch_tar_features)
                DR_prop = net(batch_prop_features)

            if loss_type == "SP":
                #Softplus loss
                softplus_fn = torch.nn.Softplus(beta=1,threshold=20)
                sigmoid_fn = torch.nn.Sigmoid()
                SP_div = torch.mean(sigmoid_fn(DR_prop) * DR_prop) - torch.mean(softplus_fn(DR_prop)) - torch.mean(sigmoid_fn(DR_tar))

                #penalty term: prevent assigning zero to all fake image
                penalty = LAMBDA * (torch.mean(DR_prop) - 1)**2
                loss = SP_div + penalty
            elif loss_type == "uLSIF": #uLSIF loss, also known as SQ loss
                #SQ loss
                loss = 0.5* torch.mean(DR_prop**2) - torch.mean(DR_tar) + LAMBDA * (torch.mean(DR_prop) - 1)**2
            elif loss_type == "DSKL": #DSKL loss proposed in "Deep density ratio estimation for change point detection"
                loss = - torch.mean(torch.log(DR_tar + 1e-14)) + torch.mean(torch.log(DR_prop + 1e-14))
            elif loss_type == "BARR": #BARR loss proposed in "Deep density ratio estimation for change point detection"
                loss = - torch.mean(torch.log(DR_tar + 1e-14)) + LAMBDA * (torch.abs(torch.mean(DR_prop)-1))

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()

            batch_idx += 1

            # #### debugging
            # net.eval()
            # netG.eval()
            # #with torch.no_grad():
            # z = torch.randn(BATCH_SIZE, dim_gan, dtype=torch.float).to(device)
            # batch_prop_samples = netG(z)
            # DR_tar2 = net(batch_tar_samples).detach().cpu().numpy()
            # DR_prop2 = net(batch_prop_samples.detach()).detach().cpu().numpy()
            # print("Debug:%f/%f" % (DR_tar2.mean(),DR_prop2.mean()))
        #end while

        print("DRE+%s+lambda%.3f: [epoch %d/%d] train_loss:%.3f" % ( loss_type, LAMBDA, epoch+1, EPOCHS_DRE, train_loss/(batch_idx+1)))

        avg_train_loss.append(train_loss/(batch_idx+1))

        # save checkpoint
        if save_models_folder is not None and (epoch+1) % 50 == 0:
            save_file = save_models_folder + "/DRE_GAN_checkpoint/"
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = save_file + "ckpt_DRE_" + loss_type + "_LAMBDA_" + str(LAMBDA) +  "_checkpoint_epoch" + str(epoch+1) + ".pth"
            torch.save({
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
            }, save_file)

            # save loss
            log_file = save_models_folder + "/DRE_GAN_checkpoint/DRE_"+loss_type+"_LAMBDA_"+str(LAMBDA)+"_train_loss_epoch" + str(epoch+1) + ".npz"
            np.savez(log_file, np.array(avg_train_loss))
    #end for epoch

    # n_samples = 100000
    # batch_size_tmp = 1000
    # net.eval()
    # with torch.no_grad():
    #     tmp = 0
    #     while tmp < n_samples:
    #         tar_data_iter = iter(tar_dataloader)
    #         batch_samples,_ = tar_data_iter.next()
    #         batch_samples = batch_samples.type(torch.float).to(device)
    #         batch_weights = net(batch_samples).detach().cpu().numpy()
    #         print((batch_weights.min(),batch_weights.mean(),batch_weights.max()))
    #         tmp += batch_size_tmp
    #     #end while


    return net, optimizer, avg_train_loss
