#!/bin/bash

## tensorboard --logdir /home/xin/OneDrive/Working_directory/DDRE_Sampling_GANs/CelebA/Output/saved_logs


##---------------------------------------------------------------------------------------------------------
## Binary Attribute Selection


# # for attr_idx in {0..39}
# for attr_idx in 0 2 8 11 18 19 20 21 31 32 33 34 36 39
# do
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "Pre-train ResNet34 with $attr_idx -th binary attribute"
#   CUDA_VISIBLE_DEVICES=1,0 python3 PreTrainCNN.py --CNN ResNet34 --isometric_map --epochs 30 --batch_size_train 256 --batch_size_test 128 --base_lr 1e-3 --transform --attr_idx $attr_idx
# done
#
#
# for attr_idx in 0 2 8 11 18 19 20 21 31 32 33 34 36 39
# do
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "Pre-train ResNet50 with $attr_idx -th binary attribute"
#   CUDA_VISIBLE_DEVICES=1,0 python3 PreTrainCNN.py --CNN ResNet50 --isometric_map --epochs 30 --batch_size_train 128 --batch_size_test 128 --base_lr 1e-3 --transform --attr_idx $attr_idx
# done


##---------------------------------------------------------------------------------------------------------
## Feature extraction

# echo "-------------------------------------------------------------------------------------------------"
# echo "Pre-train a ResNet34 for DRE in feature space"
# CUDA_VISIBLE_DEVICES=1,0 python3 PreTrainCNN.py --CNN ResNet34 --isometric_map --epochs 100 --batch_size_train 256 --batch_size_test 128 --base_lr 1e-3 --num_classes 6 --transform


# echo "-------------------------------------------------------------------------------------------------"
# echo "Pre-train a ResNet34 for DRE in feature space"
# CUDA_VISIBLE_DEVICES=1,0 python3 PreTrainCNN.py --CNN ResNet50 --isometric_map --epochs 100 --batch_size_train 256 --batch_size_test 128 --base_lr 1e-3 --num_classes 6 --transform



##########################################################################################################
# Experiment 1: reproduce MH-GAN, DCGAN, epoch_gan = 60
##########################################################################################################

##---------------------------------------------------------------------------------------------------------
## DCGAN  Experiments
##---------------------------------------------------------------------------------------------------------

# EPOCH_DCGAN=60
# EPOCH_DRE=200

# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN $EPOCH_GAN epoch"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan $EPOCH_DCGAN --lr_g_gan 2e-4 --lr_d_gan 2e-4 --batch_size_gan 512 --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID #--samp_round 3 #--resumeTrain_gan 0 #--realdata_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan $EPOCH_DCGAN --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE disc --Sampling RS --samp_nfake 50000 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan $EPOCH_DCGAN --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS keepTrain Does not work, acceptance rate is far too low"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan $EPOCH_DCGAN --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE disc_KeepTrain --Sampling RS --samp_nfake 50000 --comp_ISFID --samp_round 3


# for lambda_DRE in 0.0 0.005 0.05 0.1
# do
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "DCGAN DRE-F-SP+RS $lambda_DRE ReLU ResNet34"
#   CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan $EPOCH_DCGAN --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --PreCNN_DR ResNet34 --epoch_pretrainCNN 100 --lambda_DRE $lambda_DRE --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE $EPOCH_DRE --base_lr_DRE 1e-4 #--comp_ISFID
# done

# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+MH 0.01 ReLU ResNet34"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan $EPOCH_DCGAN --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling MH --PreCNN_DR ResNet34 --epoch_pretrainCNN 100 --lambda_DRE 0.01 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE $EPOCH_DRE --base_lr_DRE 1e-4 #--comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+SIR 0.01 ReLU ResNet34"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan $EPOCH_DCGAN --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling SIR --PreCNN_DR ResNet34 --epoch_pretrainCNN 100 --lambda_DRE 0.01 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE $EPOCH_DRE --base_lr_DRE 1e-4 --comp_ISFID




# for lambda_DRE in 0.01 # 0.0 0.005 0.01 0.05 0.1
# do
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "DCGAN DRE-F-SP+RS $lambda_DRE ReLU ResNet50"
#   CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan $EPOCH_DCGAN --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --PreCNN_DR ResNet50 --epoch_pretrainCNN 100 --lambda_DRE $lambda_DRE --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE $EPOCH_DRE --base_lr_DRE 1e-4 --comp_ISFID
# done




##---------------------------------------------------------------------------------------------------------
## WGAN  Experiments
##---------------------------------------------------------------------------------------------------------
# EPOCH_WGAN=2000

# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP $EPOCH_GAN epoch"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN WGANGP --epoch_gan $EPOCH_WGAN --lr_g_gan 2e-4 --lr_d_gan 2e-4 --batch_size_gan 512 --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN WGANGP --epoch_gan $EPOCH_WGAN --DRE disc_MHcal --Sampling RS --samp_nfake 50000 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN WGANGP --epoch_gan $EPOCH_WGAN --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID
#
# for lambda_DRE in 0.0 0.005 0.01 0.05 0.1
# do
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "WGANGP DRE-F-SP+RS $lambda_DRE ReLU ResNet34"
#   CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN WGANGP --epoch_gan $EPOCH_WGAN --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --PreCNN_DR ResNet34 --lambda_DRE $lambda_DRE --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 100 --base_lr_DRE 1e-4 --comp_ISFID
# done


##########################################################################################################
# Experiment 2: SNGAN
##########################################################################################################
EPOCH_SNGAN=100
EPOCH_DRE=100
EPOCH_PRECNN=100

# echo "-------------------------------------------------------------------------------------------------"
# echo "SNGAN"
# echo "Baseline"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN SNGAN --epoch_gan $EPOCH_SNGAN --lr_g 1e-4 --lr_d 4e-4 --DRE None --Sampling None --samp_nfake 50000 --batch_size_gan 256 --resumeTrain_gan 0 --comp_ISFID --samp_batch_size 1000 --samp_round 1 --realdata_ISFID


# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN SNGAN --DRE disc --Sampling RS --epoch_gan $EPOCH_SNGAN --samp_nfake 50000 --comp_ISFID --samp_batch_size 1000
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS disc_MHcal"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN SNGAN --DRE disc_MHcal --Sampling RS --epoch_gan $EPOCH_SNGAN --samp_nfake 50000 --comp_ISFID --samp_batch_size 1000
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN SNGAN --DRE disc_MHcal --Sampling MH --epoch_gan $EPOCH_SNGAN --samp_nfake 50000 --comp_ISFID --samp_batch_size 1000
#
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "SNGAN DRE-F-SP+MH $lambda_DRE ReLU ResNet34"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN SNGAN --epoch_gan $EPOCH_SNGAN --DRE DRE_F_SP --Sampling MH --PreCNN_DR ResNet34 --lambda_DRE 0.01 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE $EPOCH_DRE --base_lr_DRE 1e-3 --comp_ISFID --samp_batch_size 1000


for lambda_DRE in 0 0.01 0.05 0.1  # 0 0.005 0.01 0.05 0.1
do
  echo "-------------------------------------------------------------------------------------------------"
  echo "SNGAN DRE-F-SP+RS $lambda_DRE ReLU ResNet34"
  CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN SNGAN --epoch_gan $EPOCH_SNGAN --DRE DRE_F_SP --Sampling RS --PreCNN_DR ResNet34 --epoch_pretrainCNN $EPOCH_PRECNN --DR_Net MLP5 --lambda_DRE $lambda_DRE --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE $EPOCH_DRE --base_lr_DRE 1e-4 --comp_ISFID --samp_batch_size 1000 --samp_round 1
done


# lambda_DRE=0.005
# echo "-------------------------------------------------------------------------------------------------"
# echo "SNGAN DRE-F-SP+MH $lambda_DRE ReLU ResNet34"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN SNGAN --epoch_gan $EPOCH_SNGAN --DRE DRE_F_SP --Sampling MH --PreCNN_DR ResNet34 --epoch_pretrainCNN $EPOCH_PRECNN --DR_Net MLP5 --lambda_DRE $lambda_DRE --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE $EPOCH_DRE --base_lr_DRE 1e-4 --comp_ISFID --samp_batch_size 1000 --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "SNGAN DRE-F-SP+SIR $lambda_DRE ReLU ResNet34"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN SNGAN --epoch_gan $EPOCH_SNGAN --DRE DRE_F_SP --Sampling SIR --PreCNN_DR ResNet34 --epoch_pretrainCNN $EPOCH_PRECNN --DR_Net MLP5 --lambda_DRE $lambda_DRE --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE $EPOCH_DRE --base_lr_DRE 1e-4 --comp_ISFID --samp_batch_size 1000 --samp_round 3




# ## reset fan speed
nvidia-settings -a "[gpu:0]/GPUFanControlState=0"
nvidia-settings -a "[gpu:1]/GPUFanControlState=0"
