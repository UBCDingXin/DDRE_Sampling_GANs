#!/bin/bash


# echo "-------------------------------------------------------------------------------------------------"
# echo "PreTrainCNN ResNet34"
# python3 PreTrainCNN.py --CNN ResNet34 --N_TRAIN 5000 --isometric_map --transform --batch_size_train 512 --base_lr 0.01 --epochs 200


# echo "-------------------------------------------------------------------------------------------------"
# echo "PreTrainCNN InceptionV3"
# python3 PreTrainCNN.py --CNN InceptionV3 --transform --batch_size_train 32


# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline: 1.723 0.004; 7.791 0.004"
# python3 main_unsupervised.py --N_TRAIN 5000 --GAN DCGAN --epoch_gan 500 --DRE None --Sampling None --lr_g_ga 1e-4 --lr_d_gan 1e-4 --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS: 1.658 0.007; 8.032 0.007"
# python3 main_unsupervised.py --N_TRAIN 5000 --GAN DCGAN --epoch_gan 500 --DRE disc --Sampling RS --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# python3 main_unsupervised.py --N_TRAIN 5000 --GAN DCGAN --epoch_gan 500 --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID --samp_round 3


# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.01 ReLU"
# python3 main_unsupervised.py --N_TRAIN 5000 --GAN DCGAN --epoch_gan 500 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.01 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 500 --epoch_pretrainCNN 200 --base_lr_DRE 1e-4  --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.1 ReLU"
# python3 main_unsupervised.py --N_TRAIN 5000 --GAN DCGAN --epoch_gan 500 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.1 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 500 --epoch_pretrainCNN 200 --base_lr_DRE 1e-4  --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+MH 0.1 ReLU"
# python3 main_unsupervised.py --N_TRAIN 5000 --GAN DCGAN --epoch_gan 500 --DRE DRE_F_SP --Sampling MH --lambda_DRE 0.1 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 500 --epoch_pretrainCNN 200 --base_lr_DRE 1e-4  --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+SIR 0.1 ReLU"
# python3 main_unsupervised.py --N_TRAIN 5000 --GAN DCGAN --epoch_gan 500 --DRE DRE_F_SP --Sampling SIR --lambda_DRE 0.1 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 500 --epoch_pretrainCNN 200 --base_lr_DRE 1e-4  --comp_ISFID --samp_round 3



echo "-------------------------------------------------------------------------------------------------"
echo "WGANGP"
echo "-------------------------------------------------------------------------------------------------"
echo "Baseline: "
python3 main_unsupervised.py --N_TRAIN 5000 --GAN WGANGP --epoch_gan 1000 --DRE None --Sampling None --lr_g_ga 1e-4 --lr_d_gan 1e-4 --samp_nfake 50000 --comp_ISFID --samp_round 3
echo "-------------------------------------------------------------------------------------------------"
echo "DRS: "
python3 main_unsupervised.py --N_TRAIN 5000 --GAN WGANGP --epoch_gan 1000 --DRE disc_MHcal --Sampling RS --samp_nfake 50000 --comp_ISFID --samp_round 3
echo "-------------------------------------------------------------------------------------------------"
echo "MH-GAN"
python3 main_unsupervised.py --N_TRAIN 5000 --GAN WGANGP --epoch_gan 1000 --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID --samp_round 3


echo "-------------------------------------------------------------------------------------------------"
echo "WGANGP DRE-F-SP+RS 0.01 ReLU"
python3 main_unsupervised.py --N_TRAIN 5000 --GAN WGANGP --epoch_gan 1000 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.01 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 500 --epoch_pretrainCNN 200 --base_lr_DRE 1e-4  --comp_ISFID --samp_round 3
echo "-------------------------------------------------------------------------------------------------"
echo "WGANGP DRE-F-SP+RS 0.1 ReLU"
python3 main_unsupervised.py --N_TRAIN 5000 --GAN WGANGP --epoch_gan 1000 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.1 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 500 --epoch_pretrainCNN 200 --base_lr_DRE 1e-4  --comp_ISFID --samp_round 3
echo "-------------------------------------------------------------------------------------------------"
echo "WGANGP DRE-F-SP+MH 0.1 ReLU"
python3 main_unsupervised.py --N_TRAIN 5000 --GAN WGANGP --epoch_gan 1000 --DRE DRE_F_SP --Sampling MH --lambda_DRE 0.1 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 500 --epoch_pretrainCNN 200 --base_lr_DRE 1e-4  --comp_ISFID --samp_round 3
echo "-------------------------------------------------------------------------------------------------"
echo "WGANGP DRE-F-SP+SIR 0.1 ReLU"
python3 main_unsupervised.py --N_TRAIN 5000 --GAN WGANGP --epoch_gan 1000 --DRE DRE_F_SP --Sampling SIR --lambda_DRE 0.1 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 500 --epoch_pretrainCNN 200 --base_lr_DRE 1e-4  --comp_ISFID --samp_round 3










##############################################################################################################
# echo "-------------------------------------------------------------------------------------------------"
# echo "cDCGAN"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline FID:0.968; IS:8.972(0.042)"
# python3 main_supervised.py --N_TRAIN 100 --GAN cDCGAN --DRE None --Sampling None --epoch_gan 5000 --resumeTrain_gan 0 --lr_g_ga 1e-4 --lr_d_gan 1e-4 --samp_nfake 50000 --comp_ISFID --samp_round 1
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS 1.137, 9.115(0.034), "
# python3 main_supervised.py --N_TRAIN 100 --GAN cDCGAN --DRE disc --Sampling RS --epoch_gan 5000 --samp_nfake 50000 --comp_ISFID --samp_round 1

# echo "-------------------------------------------------------------------------------------------------"
# echo "cDCGAN cDRE-F-SP+RS 0.01 ReLU"
# python3 main_supervised.py --N_TRAIN 100 --GAN cDCGAN --DRE cDRE_F_SP --Sampling RS --epoch_gan 5000 --lambda_DRE 0.01 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 5000 --base_lr_DRE 1e-4  --da_output --da_nfake 500000 #--comp_ISFID --samp_round 1

## no da acc: 88.28
# python3 subsequent_CNN.py --N_TRAIN 100 --CNN ResNet34 --epochs 2000 --transform --batch_size_train 512 --da_nfake 0 --base_lr 0.01
## da acc: 89.13 %
# python3 subsequent_CNN.py --N_TRAIN 100 --CNN ResNet34 --epochs 200 --transform --batch_size_train 512 --da_nfake 500000 --base_lr 0.01




# echo "-------------------------------------------------------------------------------------------------"
# echo "cDCGAN 1000"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline FID: 0.357 ; IS: 9.714"
# python3 main_supervised.py --N_TRAIN 1000 --GAN cDCGAN --DRE None --Sampling None --epoch_gan 5000 --resumeTrain_gan 0 --lr_g_ga 1e-4 --lr_d_gan 1e-4 --samp_nfake 50000  --da_output --da_nfake 500000 #--comp_ISFID --samp_round 1
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS  0.343,  9.738 "
# python3 main_supervised.py --N_TRAIN 1000 --GAN cDCGAN --DRE disc --Sampling RS --epoch_gan 5000 --samp_nfake 50000 --comp_ISFID --samp_round 1
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "PreTrainCNN ResNet34"
# python3 PreTrainCNN.py --CNN ResNet34 --N_TRAIN 1000 --isometric_map --transform --batch_size_train 512 --base_lr 0.01 --epochs 200

# echo "-------------------------------------------------------------------------------------------------"
# echo "cDCGAN cDRE-F-SP+RS 0.01 ReLU: 0.255, 9.864; da acc: 97.93"
# python3 main_supervised.py --N_TRAIN 1000 --GAN cDCGAN --DRE cDRE_F_SP --Sampling RS --epoch_gan 5000 --lambda_DRE 0.01 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_pretrainCNN 200 --epoch_DRE 5000 --base_lr_DRE 1e-4  --da_output --da_nfake 500000 --comp_ISFID --samp_round 1

# ## no da acc: 97.03
# python3 subsequent_CNN.py --N_TRAIN 1000 --CNN ResNet34 --epochs 2000 --transform --batch_size_train 512 --da_nfake 0 --base_lr 0.01
## subsampling da acc: 97.93
# python3 subsequent_CNN.py --N_TRAIN 1000 --CNN ResNet34 --epochs 200 --transform --batch_size_train 512 --da_nfake 500000 --base_lr 0.01
## no subsampling da acc: 97.46
# python3 subsequent_CNN.py --N_TRAIN 1000 --CNN ResNet34 --epochs 200 --transform --batch_size_train 512 --da_nfake 500000 --base_lr 0.01


# python3 subsequent_CNN.py --N_TRAIN 1450 --CNN ResNet34 --epochs 2000 --transform --batch_size_train 512 --da_nfake 0 --base_lr 0.01
