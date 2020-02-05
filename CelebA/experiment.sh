#!/bin/bash


##---------------------------------------------------------------------------------------------------------
## Feature extraction

# echo "-------------------------------------------------------------------------------------------------"
# echo "Pre-train a ResNet34 for DRE in feature space"
# python3 PreTrainCNN.py --CNN ResNet34 --isometric_map --epochs 100 --batch_size_train 256 --batch_size_test 128 --base_lr 1e-3 --num_classes 6 --transform



##########################################################################################################
# SNGAN
##########################################################################################################
EPOCH_SNGAN=100
EPOCH_DRE=100
EPOCH_PRECNN=100

# echo "-------------------------------------------------------------------------------------------------"
# echo "SNGAN"
# echo "Baseline"
# python3 main.py --GAN SNGAN --epoch_gan $EPOCH_SNGAN --lr_g 1e-4 --lr_d 4e-4 --DRE None --Sampling None --samp_nfake 50000 --batch_size_gan 256 --resumeTrain_gan 0 --comp_ISFID --samp_batch_size 1000 --samp_round 1 --realdata_ISFID


# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# python3 main.py --GAN SNGAN --DRE disc --Sampling RS --epoch_gan $EPOCH_SNGAN --samp_nfake 50000 --comp_ISFID --samp_batch_size 1000
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS disc_MHcal"
# python3 main.py --GAN SNGAN --DRE disc_MHcal --Sampling RS --epoch_gan $EPOCH_SNGAN --samp_nfake 50000 --comp_ISFID --samp_batch_size 1000
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# python3 main.py --GAN SNGAN --DRE disc_MHcal --Sampling MH --epoch_gan $EPOCH_SNGAN --samp_nfake 50000 --comp_ISFID --samp_batch_size 1000

for lambda_DRE in 0 0.005 0.01 0.05 0.1
do
  echo "-------------------------------------------------------------------------------------------------"
  echo "SNGAN DRE-F-SP+RS $lambda_DRE ReLU ResNet34"
  python3 main.py --GAN SNGAN --epoch_gan $EPOCH_SNGAN --DRE DRE_F_SP --Sampling RS --PreCNN_DR ResNet34 --epoch_pretrainCNN $EPOCH_PRECNN --DR_Net MLP5 --lambda_DRE $lambda_DRE --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE $EPOCH_DRE --base_lr_DRE 1e-4 --comp_ISFID --samp_batch_size 1000 --samp_round 1
done


