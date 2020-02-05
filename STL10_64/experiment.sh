#!/bin/bash

EPOCH_SNGAN=500
EPOCH_DRE=100
EPOCH_PREFC=50
PRE_CNN="ResNet34" #ResNet34 (91.22), ResNet50 (91.84 %)


# echo "-------------------------------------------------------------------------------------------------"
# echo "Pre-train a CNN for DRE in feature space"
# CUDA_VISIBLE_DEVICES=1,0 python3 PreTrainFC.py --CNN $PRE_CNN --epochs $EPOCH_PREFC --batch_size_train 64 --batch_size_test 64 --base_lr 0.01 --transform --resume_epoch 0


# echo "-------------------------------------------------------------------------------------------------"
# echo "SNGAN"
# echo "Baseline"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN SNGAN --epoch_gan $EPOCH_SNGAN --DRE None --Sampling None --samp_nfake 50000 --batch_size_gan 256 --resumeTrain_gan 0 --lr_g 2e-4 --lr_d 2e-4 --comp_ISFID --samp_batch_size 1000  #--realdata_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN SNGAN --epoch_gan $EPOCH_SNGAN --DRE disc --Sampling RS --samp_nfake 50000 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN SNGAN --epoch_gan $EPOCH_SNGAN --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID



for lambda_DRE in 0 0.005 0.01 0.05 0.1
do
  # echo "-------------------------------------------------------------------------------------------------"
  # echo "SNGAN DRE-F-SP+RS $lambda_DRE ReLU ResNet34"
  # CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN SNGAN --epoch_gan $EPOCH_SNGAN --DRE DRE_F_SP --Sampling RS --PreCNN_DR $PRE_CNN --epoch_fc $EPOCH_PREFC --DR_Net MLP5 --lambda_DRE $lambda_DRE --samp_nfake 50000 --epoch_DRE $EPOCH_DRE --base_lr_DRE 1e-4 --KS_test --comp_ISFID --samp_batch_size 1000

  echo "-------------------------------------------------------------------------------------------------"
  echo "SNGAN DRE-F-SP+SIR $lambda_DRE ReLU ResNet34"
  CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN SNGAN --epoch_gan $EPOCH_SNGAN --DRE DRE_F_SP --Sampling SIR --PreCNN_DR $PRE_CNN --epoch_fc $EPOCH_PREFC --DR_Net MLP5 --lambda_DRE $lambda_DRE --samp_nfake 50000 --epoch_DRE $EPOCH_DRE --base_lr_DRE 1e-4 --KS_test --comp_ISFID --samp_batch_size 1000
done




## reset fan speed
nvidia-settings -a "[gpu:0]/GPUFanControlState=0"
nvidia-settings -a "[gpu:1]/GPUFanControlState=0"
