#!/bin/bash

##########################################################################################################
# Experiment 1:
# Try to reproduce the result in MH-GAN. Reduce epoch_gan to 60.

# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 60 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID --samp_round 3 #--realdata_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 60 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc --Sampling RS --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 60 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID --samp_round 3


### DRE
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 60 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 --samp_round 3 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.001 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 60 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.001 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.005 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 60 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.005 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.01 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 60 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.01 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.05 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 60 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.05 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.1 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 60 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.1 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4



# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+MH 0 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 60 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling MH --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --base_lr_DRE 1e-4 --samp_round 3 --comp_ISFID

# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+SIR 0 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 60 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling SIR --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --base_lr_DRE 1e-4 --samp_round 3 --comp_ISFID







##########################################################################################################
# Experiment 2:
# IS increment led by MH-GAN and DRS versus epoch_gan (20, 60, 100, 200, 300, 500)


# echo "================================================================================================="
# echo "epoch_gan = 20"
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 20 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 20 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc --Sampling RS --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 20 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.01 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 20 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 --samp_nfake 50000 --comp_ISFID --samp_round 3


# echo "================================================================================================="
# echo "epoch_gan = 40"
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 40 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 40 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc --Sampling RS --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 40 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.01 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 40 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 --samp_nfake 50000 --comp_ISFID --samp_round 3
#
#
# echo "================================================================================================="
# echo "epoch_gan = 80"
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 80 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 80 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc --Sampling RS --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 80 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.01 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 80 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 --samp_nfake 50000 --comp_ISFID --samp_round 3

# echo "================================================================================================="
# echo "epoch_gan = 100"
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 100 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID --samp_round 3 #--realdata_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 100 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc --Sampling RS --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 100 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.01 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 100 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 --samp_nfake 50000 --comp_ISFID --samp_round 3


# echo "================================================================================================="
# echo "epoch_gan = 200"
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 200 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID --samp_round 3 #--realdata_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 200 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc --Sampling RS --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 200 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.01 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 200 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 --samp_nfake 50000 --comp_ISFID --samp_round 3


# echo "================================================================================================="
# echo "epoch_gan = 300"
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 300 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID --samp_round 3 #--realdata_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 300 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc --Sampling RS --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 300 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.01 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 300 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 --samp_nfake 50000 --comp_ISFID --samp_round 3


# echo "================================================================================================="
# echo "epoch_gan = 400"
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 400 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID --samp_round 3 #--realdata_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 400 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc --Sampling RS --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 400 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.01 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 400 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 --samp_nfake 50000 --comp_ISFID --samp_round 3


# echo "================================================================================================="
# echo "epoch_gan = 600"
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 600 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID --samp_round 3 #--realdata_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 600 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc --Sampling RS --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 600 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.01 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 600 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 --samp_nfake 50000 --comp_ISFID --samp_round 3


# echo "================================================================================================="
# echo "epoch_gan = 700"
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 700 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID --samp_round 3 #--realdata_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 700 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc --Sampling RS --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 700 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.01 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 700 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 --samp_nfake 50000 --comp_ISFID --samp_round 3


# echo "================================================================================================="
# echo "epoch_gan = 800"
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 800 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID --samp_round 3 #--realdata_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 800 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc --Sampling RS --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 800 --lr_g_gan 2e-4 --lr_d_gan 2e-4  --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID --samp_round 3
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.01 ReLU"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN DCGAN --epoch_gan 800 --lr_g_gan 2e-4 --lr_d_gan 2e-4 --DRE DRE_F_SP --Sampling RS --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 --samp_nfake 50000 --comp_ISFID --samp_round 3








# ## reset fan speed
nvidia-settings -a "[gpu:0]/GPUFanControlState=0"
nvidia-settings -a "[gpu:1]/GPUFanControlState=0"
