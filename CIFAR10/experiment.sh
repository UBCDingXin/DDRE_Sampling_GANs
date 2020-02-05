#!/bin/bash


# echo "-------------------------------------------------------------------------------------------------"
# echo "Pre-train a ResNet34 for feature extraction"
# python3 PreTrainCNN.py --CNN ResNet34 --isometric_map --transform
#
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "Pre-train an Inceptionv3 for evaluation"
# python3 PreTrainCNN.py --CNN InceptionV3 --transform


# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# python3 main.py --GAN DCGAN --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID --samp_round 1 --realdata_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# python3 main.py --GAN DCGAN --DRE disc --Sampling RS --samp_nfake 50000 --comp_ISFID --samp_round 1
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS keep train"
# python3 main.py --GAN DCGAN --DRE disc_KeepTrain --Sampling RS --samp_nfake 50000 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# python3 main.py --GAN DCGAN --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID --samp_round 1
#
# ### baseline DREs
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRS+DRE_P_uLSIF"
# python3 main.py --GAN DCGAN --DRE DRE_P_uLSIF --DR_Net 2layersCNN --lambda_DRE 0 --not_decay_lr_DRE --Sampling RS --samp_nfake 50000 --base_lr_DRE 1e-4 --batch_size_DRE 512 --comp_ISFID
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRS+DRE_F_DSKL"
# python3 main.py --GAN DCGAN --DRE DRE_P_DSKL --DR_Net 6layersCNN --lambda_DRE 0 --Sampling RS --samp_nfake 50000 --base_lr_DRE 1e-4 --batch_size_DRE 512 --comp_ISFID
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRS+DRE_P_BARR lambda 10"
# python3 main.py --GAN DCGAN --DRE DRE_P_BARR --DR_Net 6layersCNN --lambda_DRE 10 --not_decay_lr_DRE --Sampling RS --samp_nfake 50000 --base_lr_DRE 1e-4 --batch_size_DRE 512 --comp_ISFID
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRS+BayesClass"
# python3 main.py --GAN DCGAN --DRE BayesClass --Sampling RS --samp_nfake 50000 --comp_ISFID


# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0 ReLU"
# python3 main.py --GAN DCGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.005 ReLU"
# python3 main.py --GAN DCGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.005 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.01 ReLU"
# python3 main.py --GAN DCGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.01 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.05 ReLU"
# python3 main.py --GAN DCGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.05 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.1 ReLU"
# python3 main.py --GAN DCGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.1 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 0.5 ReLU"
# python3 main.py --GAN DCGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.5 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS 1 ReLU"
# python3 main.py --GAN DCGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 1 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4


# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+SIR 0 ReLU"
# python3 main.py --GAN DCGAN --DRE DRE_F_SP --Sampling SIR --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --base_lr_DRE 1e-4 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+MH 0 ReLU"
# python3 main.py --GAN DCGAN --DRE DRE_F_SP --Sampling MH --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --base_lr_DRE 1e-4 --comp_ISFID


# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE in Pixel Space"
# echo "DCGAN DRE-P-SP VGG11 Softplus 0"
# python3 main.py --GAN DCGAN --DRE DRE_P_SP --DR_Net VGG11 --Sampling RS --lambda_DRE 0 --batch_size_DRE 256 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn Softplus --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 100 --IS_batch_size 100 --comp_ISFID --base_lr_DRE 1e-4 --samp_round 1
#
# echo "DCGAN DRE-P-SP VGG11 Softplus 0.005"
# python3 main.py --GAN DCGAN --DRE DRE_P_SP --DR_Net VGG11 --Sampling RS --lambda_DRE 0.005 --batch_size_DRE 256 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn Softplus --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 100 --IS_batch_size 100 #--comp_ISFID
#
# echo "DCGAN DRE-P-SP VGG11 Softplus 0.01"
# python3 main.py --GAN DCGAN --DRE DRE_P_SP --DR_Net VGG11 --Sampling RS --lambda_DRE 0.01 --batch_size_DRE 256 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn Softplus --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 100 --IS_batch_size 100 --samp_round 1 --comp_ISFID
#
# echo "DCGAN DRE-P-SP VGG11 Softplus 0.05"
# python3 main.py --GAN DCGAN --DRE DRE_P_SP --DR_Net VGG11 --Sampling RS --lambda_DRE 0.05 --batch_size_DRE 256 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn Softplus --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 100 --IS_batch_size 100 #--comp_ISFID
#
# echo "DCGAN DRE-P-SP VGG13 Softplus 0.01"
# python3 main.py --GAN DCGAN --DRE DRE_P_SP --DR_Net VGG13 --Sampling RS --lambda_DRE 0.01 --batch_size_DRE 256 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn Softplus --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 100 --IS_batch_size 100 --samp_round 1 --comp_ISFID
#
# echo "DCGAN DRE-P-SP VGG16 Softplus 0.01"
# python3 main.py --GAN DCGAN --DRE DRE_P_SP --DR_Net VGG16 --Sampling RS --lambda_DRE 0.01 --batch_size_DRE 256 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn Softplus --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 100 --IS_batch_size 100 --samp_round 1 --comp_ISFID
#
# echo "DCGAN DRE-P-SP VGG19 Softplus 0.01"
# python3 main.py --GAN DCGAN --DRE DRE_P_SP --DR_Net VGG19 --Sampling RS --lambda_DRE 0.01 --batch_size_DRE 256 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn Softplus --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 100 --IS_batch_size 100 --samp_round 1 --comp_ISFID

# echo "-------------------------------------------------------------------------------------------------"
# echo "compare different loss functions when DR models and sampelrs are fixed as MLP5 and RS"
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-uLSIF+RS ReLU"
# python3 main.py --GAN DCGAN --DRE DRE_F_uLSIF --DR_Net MLP5 --Sampling RS --lambda_DRE 0 --batch_size_DRE 512 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 5000 --resumeTrain_DRE 0 --FID_batch_size 500 --IS_batch_size 500 --samp_round 3 --base_lr_DRE 1e-5 --not_decay_lr_DRE #--comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-DSKL+RS ReLU"
# python3 main.py --GAN DCGAN --DRE DRE_F_DSKL --DR_Net MLP5 --Sampling RS --lambda_DRE 0 --batch_size_DRE 512 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 200 --IS_batch_size 200 --samp_round 3 --base_lr_DRE 1e-5 --not_decay_lr_DRE #--comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-BARR+RS ReLU lambda 10"
# python3 main.py --GAN DCGAN --DRE DRE_F_BARR --DR_Net MLP5 --Sampling RS --lambda_DRE 10 --batch_size_DRE 512 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 200 --IS_batch_size 200 --samp_round 3 --base_lr_DRE 1e-5 --not_decay_lr_DRE #--comp_ISFID

# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-BARR+RS ReLU lambda 0"
# python3 main.py --GAN DCGAN --DRE DRE_F_BARR --DR_Net MLP5 --Sampling RS --lambda_DRE 0 --batch_size_DRE 512 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 200 --IS_batch_size 200 --samp_round 3 --base_lr_DRE 1e-5 --not_decay_lr_DRE #--comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-BARR+RS ReLU lambda 0.005"
# python3 main.py --GAN DCGAN --DRE DRE_F_BARR --DR_Net MLP5 --Sampling RS --lambda_DRE 0.005 --batch_size_DRE 512 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 200 --IS_batch_size 200 --samp_round 3 --base_lr_DRE 1e-5 --not_decay_lr_DRE #--comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-BARR+RS ReLU lambda 0.01"
# python3 main.py --GAN DCGAN --DRE DRE_F_BARR --DR_Net MLP5 --Sampling RS --lambda_DRE 0.01 --batch_size_DRE 512 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 200 --IS_batch_size 200 --samp_round 3 --base_lr_DRE 1e-5 --not_decay_lr_DRE #--comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-BARR+RS ReLU lambda 0.05"
# python3 main.py --GAN DCGAN --DRE DRE_F_BARR --DR_Net MLP5 --Sampling RS --lambda_DRE 0.05 --batch_size_DRE 512 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 200 --IS_batch_size 200 --samp_round 3 --base_lr_DRE 1e-5 --not_decay_lr_DRE #--comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-BARR+RS ReLU lambda 0.1"
# python3 main.py --GAN DCGAN --DRE DRE_F_BARR --DR_Net MLP5 --Sampling RS --lambda_DRE 0.1 --batch_size_DRE 512 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 200 --IS_batch_size 200 --samp_round 3 --base_lr_DRE 1e-5 --not_decay_lr_DRE #--comp_ISFID






# echo "-------------------------------------------------------------------------------------------------"
# echo "compare convergence under different loss and lambda"
# echo "DCGAN DRE-F-uLSIF+RS ReLU lambda 0"
# python3 main.py --GAN DCGAN --DRE DRE_F_uLSIF --DR_Net MLP5 --Sampling RS --lambda_DRE 0 --batch_size_DRE 512 --epoch_DRE 500 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 500 --IS_batch_size 500 --samp_round 0  --base_lr_DRE 1e-5 --not_decay_lr_DRE
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-uLSIF+RS ReLU lambda 0.1"
# python3 main.py --GAN DCGAN --DRE DRE_F_uLSIF --DR_Net MLP5 --Sampling RS --lambda_DRE 0.1 --batch_size_DRE 512 --epoch_DRE 500 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 500 --IS_batch_size 500 --samp_round 0  --base_lr_DRE 1e-5 --not_decay_lr_DRE
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-uLSIF+RS ReLU lambda 1"
# python3 main.py --GAN DCGAN --DRE DRE_F_uLSIF --DR_Net MLP5 --Sampling RS --lambda_DRE 1 --batch_size_DRE 512 --epoch_DRE 500 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 500 --IS_batch_size 500 --samp_round 0  --base_lr_DRE 1e-5 --not_decay_lr_DRE
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-uLSIF+RS ReLU lambda 10"
# python3 main.py --GAN DCGAN --DRE DRE_F_uLSIF --DR_Net MLP5 --Sampling RS --lambda_DRE 10 --batch_size_DRE 512 --epoch_DRE 500 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 500 --IS_batch_size 500 --samp_round 0  --base_lr_DRE 1e-5 --not_decay_lr_DRE

# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS ReLU lambda 0"
# python3 main.py --GAN DCGAN --DRE DRE_F_SP --DR_Net MLP5 --Sampling RS --lambda_DRE 0 --batch_size_DRE 512 --epoch_DRE 500 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 500 --IS_batch_size 500 --samp_round 0  --base_lr_DRE 1e-5 --not_decay_lr_DRE
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS ReLU lambda 0.1"
# python3 main.py --GAN DCGAN --DRE DRE_F_SP --DR_Net MLP5 --Sampling RS --lambda_DRE 0.1 --batch_size_DRE 512 --epoch_DRE 500 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 500 --IS_batch_size 500 --samp_round 0  --base_lr_DRE 1e-5 --not_decay_lr_DRE
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS ReLU lambda 1"
# python3 main.py --GAN DCGAN --DRE DRE_F_SP --DR_Net MLP5 --Sampling RS --lambda_DRE 1 --batch_size_DRE 512 --epoch_DRE 500 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 500 --IS_batch_size 500 --samp_round 0  --base_lr_DRE 1e-5 --not_decay_lr_DRE
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN DRE-F-SP+RS ReLU lambda 10"
# python3 main.py --GAN DCGAN --DRE DRE_F_SP --DR_Net MLP5 --Sampling RS --lambda_DRE 10 --batch_size_DRE 512 --epoch_DRE 500 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 500 --IS_batch_size 500 --samp_round 0  --base_lr_DRE 1e-5 --not_decay_lr_DRE

























# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# python3 main.py --GAN WGANGP --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# python3 main.py --GAN WGANGP --DRE disc_MHcal --Sampling RS --samp_nfake 50000 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# python3 main.py --GAN WGANGP --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID

# baseline DREs
# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRS+DRE_P_uLSIF"
# python3 main.py --GAN WGANGP --DRE DRE_P_uLSIF --DR_Net 2layersCNN --lambda_DRE 0 --Sampling RS --samp_nfake 50000 --base_lr_DRE 1e-4 --not_decay_lr_DRE --batch_size_DRE 512 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRS+DRE_F_DSKL"
# python3 main.py --GAN WGANGP --DRE DRE_P_DSKL --DR_Net 6layersCNN --lambda_DRE 0 --Sampling RS --samp_nfake 50000 --base_lr_DRE 1e-4 --batch_size_DRE 512 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRS+DRE_P_BARR lambda 10"
# python3 main.py --GAN WGANGP --DRE DRE_P_BARR --DR_Net 6layersCNN --lambda_DRE 10 --Sampling RS --samp_nfake 50000 --base_lr_DRE 1e-4 --not_decay_lr_DRE --batch_size_DRE 512 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRS+BayesClass"
# python3 main.py --GAN WGANGP --DRE BayesClass --Sampling RS --samp_nfake 50000 --comp_ISFID


# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRE-F-SP+RS 0 ReLU"
# python3 main.py --GAN WGANGP --DRE DRE_F_SP --Sampling RS --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRE-F-SP+RS 0.005 ReLU"
# python3 main.py --GAN WGANGP --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.005 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRE-F-SP+RS 0.01 ReLU"
# python3 main.py --GAN WGANGP --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.01 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 #--comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRE-F-SP+RS 0.05 ReLU"
# python3 main.py --GAN WGANGP --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.05 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRE-F-SP+RS 0.1 ReLU"
# python3 main.py --GAN WGANGP --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.1 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRE-F-SP+RS 0.5 ReLU"
# python3 main.py --GAN WGANGP --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.5 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRE-F-SP+RS 1 ReLU"
# python3 main.py --GAN WGANGP --DRE DRE_F_SP --Sampling RS --lambda_DRE 1 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4

# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRE-F-SP+SIR 0.005 ReLU"
# python3 main.py --GAN WGANGP --DRE DRE_F_SP --Sampling SIR --lambda_DRE 0.005 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --comp_ISFID

# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRE-F-SP+MH 0.005 ReLU"
# python3 main.py --GAN WGANGP --DRE DRE_F_SP --Sampling MH --lambda_DRE 0.005 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --comp_ISFID --samp_round 3



# echo "-------------------------------------------------------------------------------------------------"
# echo "compare different loss functions when DR models and sampelrs are fixed as MLP5 and RS"
# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRE-F-uLSIF+RS ReLU"
# python3 main.py --GAN WGANGP --DRE DRE_F_uLSIF --DR_Net MLP5 --Sampling RS --lambda_DRE 0 --batch_size_DRE 512 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 200 --IS_batch_size 200 --samp_round 3  --base_lr_DRE 1e-5 --not_decay_lr_DRE #--comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRE-F-DSKL+RS ReLU"
# python3 main.py --GAN WGANGP --DRE DRE_F_DSKL --DR_Net MLP5 --Sampling RS --lambda_DRE 0 --batch_size_DRE 512 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 200 --IS_batch_size 200 --samp_round 3  --base_lr_DRE 1e-5 --not_decay_lr_DRE #--comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP DRE-F-BARR+RS ReLU lambda 10"
# python3 main.py --GAN WGANGP --DRE DRE_F_BARR --DR_Net MLP5 --Sampling RS --lambda_DRE 10 --batch_size_DRE 512 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 200 --IS_batch_size 200 --samp_round 3 --base_lr_DRE 1e-5 --not_decay_lr_DRE #--comp_ISFID






























# echo "-------------------------------------------------------------------------------------------------"
# echo "MMD-GAN"
# echo "-------------------------------------------------------------------------------------------------"
# echo "Baseline"
# python3 main.py --GAN MMDGAN --DRE None --Sampling None --samp_nfake 50000 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRS"
# python3 main.py --GAN MMDGAN --DRE disc_MHcal --Sampling RS --samp_nfake 50000 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "MH-GAN"
# python3 main.py --GAN MMDGAN --DRE disc_MHcal --Sampling MH --samp_nfake 50000 --comp_ISFID

# baseline DREs
# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRS+DRE_P_uLSIF"
# python3 main.py --GAN MMDGAN --DRE DRE_P_uLSIF --DR_Net 2layersCNN --lambda_DRE 0 --Sampling RS --samp_nfake 50000 --base_lr_DRE 1e-4 --not_decay_lr_DRE --batch_size_DRE 512 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRS+DRE_F_DSKL"
# python3 main.py --GAN MMDGAN --DRE DRE_P_DSKL --DR_Net 6layersCNN --lambda_DRE 0 --Sampling RS --samp_nfake 50000 --base_lr_DRE 1e-3 --batch_size_DRE 512 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRS+DRE_P_BARR lambda 10"
# python3 main.py --GAN MMDGAN --DRE DRE_P_BARR --DR_Net 6layersCNN --lambda_DRE 10 --Sampling RS --samp_nfake 50000 --base_lr_DRE 1e-4 --not_decay_lr_DRE --batch_size_DRE 512 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRS+BayesCl0ass"
# python3 main.py --GAN MMDGAN --DRE BayesClass --Sampling RS --samp_nfake 50000 --comp_ISFID


# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRE-F-SP+RS 0 ReLU"
# python3 main.py --GAN MMDGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 0 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 #--comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRE-F-SP+RS 0.005 ReLU"
# python3 main.py --GAN MMDGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.005 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 #--comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRE-F-SP+RS 0.006 ReLU"
# python3 main.py --GAN MMDGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.006 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRE-F-SP+RS 0.008 ReLU"
# python3 main.py --GAN MMDGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.008 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4 --comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRE-F-SP+RS 0.01 ReLU"
# python3 main.py --GAN MMDGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.01 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRE-F-SP+RS 0.05 ReLU"
# python3 main.py --GAN MMDGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.05 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRE-F-SP+RS 0.1 ReLU not run"
# python3 main.py --GAN MMDGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.1 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRE-F-SP+RS 0.5 ReLU not run"
# python3 main.py --GAN MMDGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.5 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRE-F-SP+RS 1 ReLU not run"
# python3 main.py --GAN MMDGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 1 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4




# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRE-F-SP+SIR 0.006 ReLU"
# python3 main.py --GAN MMDGAN --DRE DRE_F_SP --Sampling SIR --lambda_DRE 0.006 --samp_nfake 50000 --DR_final_ActFn ReLU --comp_ISFID

# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRE-F-SP+MH 0.006 ReLU"
# python3 main.py --GAN MMDGAN --DRE DRE_F_SP --Sampling MH --lambda_DRE 0.006 --samp_nfake 50000 --DR_final_ActFn ReLU --comp_ISFID




# echo "-------------------------------------------------------------------------------------------------"
# echo "compare different loss functions when DR models and sampelrs are fixed as MLP5 and RS"
# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRE-F-uLSIF+RS ReLU"
# python3 main.py --GAN MMDGAN --DRE DRE_F_uLSIF --DR_Net MLP5 --Sampling RS --lambda_DRE 0 --batch_size_DRE 512 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 200 --IS_batch_size 200 --samp_round 3  --base_lr_DRE 1e-5 --not_decay_lr_DRE
# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRE-F-DSKL+RS ReLU"
# python3 main.py --GAN MMDGAN --DRE DRE_F_DSKL --DR_Net MLP5 --Sampling RS --lambda_DRE 0 --batch_size_DRE 512 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 200 --IS_batch_size 200 --samp_round 3  --base_lr_DRE 1e-5 --not_decay_lr_DRE #--comp_ISFID
# echo "-------------------------------------------------------------------------------------------------"
# echo "MMDGAN DRE-F-BARR+RS ReLU lambda 10"
# python3 main.py --GAN MMDGAN --DRE DRE_F_BARR --DR_Net MLP5 --Sampling RS --lambda_DRE 10 --batch_size_DRE 512 --epoch_DRE 200 --samp_nfake 50000 --DR_final_ActFn ReLU --samp_batch_size 10000 --resumeTrain_DRE 0 --FID_batch_size 200 --IS_batch_size 200 --samp_round 3 --base_lr_DRE 1e-5 --not_decay_lr_DRE #--comp_ISFID
