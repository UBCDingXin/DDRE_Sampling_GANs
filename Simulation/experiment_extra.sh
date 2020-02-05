#!/bin/bash




######################################################################################
# Extra experiment 1
# Replace MLP5 with a CNN

# for lambda_DRE in 0 0.005 0.01 0.05 0.1
# do
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "DRE-SP+RS lambda $lambda_DRE"
#   python3 main.py --NSIM 3 --DRE DRE_SP --DR_Net CNN5 --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE $lambda_DRE --DR_final_ActFn ReLU --batch_size_DRE 512
# done
#
# for lambda_DRE in 0.01 0.1
# do
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "DRE-SP+MH lambda $lambda_DRE"
#   python3 main.py --NSIM 3 --DRE DRE_SP --DR_Net CNN5 --Sampling MH --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE $lambda_DRE --DR_final_ActFn ReLU --batch_size_DRE 512
# done
#
# for lambda_DRE in 0.01 0.1
# do
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "DRE-SP+SIR lambda $lambda_DRE"
#   python3 main.py --NSIM 3 --DRE DRE_SP --DR_Net CNN5 --Sampling SIR --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE $lambda_DRE --DR_final_ActFn ReLU --batch_size_DRE 512
# done



######################################################################################
# Extra experiment 2
# relying on the assumption of optimality may hurt the final performance

NSIM=5
MAX_EPOCH_KT=100

# python3 main.py --NSIM $NSIM --DRE None --Sampling None --epoch_gan 50 --gmm_nfake 100000 --gmm_ncomp_nsim 165 265 295 190 215 --show_visualization

#-----------------------------------------------------------------------------
# fit GMM
## select the optimal n_components: The optimal gmm_ncomp is 280 with BIC 14466.126049
# python3 main.py --NSIM $NSIM --DRE GT --Sampling SIR --epoch_gan 50 --gmm_nfake 100000 --gmm_ncomp_grid_lb 50 --gmm_ncomp_grid_ub 300 --gmm_ncomp_grid_step 5

## Ground truth DR + SIR/RS/MH
# python3 main.py --NSIM $NSIM --DRE GT --Sampling SIR --epoch_gan 50 --gmm_nfake 100000 --gmm_ncomp_nsim 165 265 295 190 215 --show_visualization
# python3 main.py --NSIM $NSIM --DRE GT --Sampling RS --epoch_gan 50 --gmm_nfake 100000 --gmm_ncomp_nsim 165 265 295 190 215 --show_visualization
# python3 main.py --NSIM $NSIM --DRE GT --Sampling MH --epoch_gan 50 --gmm_nfake 100000 --gmm_ncomp_nsim 165 265 295 190 215 --show_visualization


# for epoch_KT in {0..100..5}
# for epoch_KT in 0 2 5 8 10 13 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
# do
#   if [ $epoch_KT -eq 0 ];then
#     echo "-------------------------------------------------------------------------------------------------"
#     echo "epoch_gan=50, SIR, no keep train"
#     python3 main.py --NSIM $NSIM --DRE disc --Sampling SIR --epoch_gan 50 --gmm_ncomp_nsim 165 265 295 190 215 --compute_disc_err --compute_dre_err --gmm_ncomp_grid_lb 100 --gmm_ncomp_grid_ub 300 --gmm_ncomp_grid_step 5
#   else
#     echo "-------------------------------------------------------------------------------------------------"
#     echo "epoch_gan=50, SIR, keep train $epoch_KT epoch"
#     python3 main.py --NSIM $NSIM --DRE disc_KeepTrain --Sampling SIR --epoch_gan 50 --epoch_KeepTrain $epoch_KT --gmm_ncomp_nsim 165 265 295 190 215 --compute_disc_err --compute_dre_err --gmm_ncomp_grid_lb 100 --gmm_ncomp_grid_ub 300 --gmm_ncomp_grid_step 5
#   fi
# done


######################################################################################
# Extra experiment 3

NSIM=1
LAMBDA_DRE=0.05
EPOCH_DRE_MAX=5000

# echo "##################################################################################################"
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE-SP+SIR lambda $LAMBDA_DRE"
# python3 main.py --NSIM $NSIM --DRE DRE_SP --Sampling SIR --epoch_DRE $EPOCH_DRE_MAX --base_lr_DRE 1e-3 --decay_lr_DRE --lr_decay_epochs_DRE 1000 --lambda_DRE $LAMBDA_DRE --DR_final_ActFn ReLU --gmm_ncomp_nsim 165 265 295 190 215 --batch_size_DRE 512 --DRE_save_at_epoch 20 50 100 150 200 300 400 600 800 1000 1200 1500 2000 2500 3000 4000 5000

# for epoch_DRE in 20 50 100 150 200 300 400 600 800 1000 1200 1500 2000 2500 3000 4000 5000
# do
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "DRE-SP+SIR lambda $LAMBDA_DRE epoch = $epoch_DRE"
#   python3 main.py --NSIM $NSIM --DRE DRE_SP --Sampling SIR --epoch_DRE $epoch_DRE --base_lr_DRE 1e-3 --decay_lr_DRE --lr_decay_epochs_DRE 1000 --lambda_DRE $LAMBDA_DRE --DR_final_ActFn ReLU --batch_size_DRE 512 --gmm_ncomp_nsim 165 265 295 190 215 --gmm_nfake 100000 --compute_dre_err --gmm_ncomp_grid_lb 100 --gmm_ncomp_grid_ub 300 --gmm_ncomp_grid_step 5
# done


# echo "##################################################################################################"
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE-uLSIF+SIR lambda $LAMBDA_DRE"
# python3 main.py --NSIM $NSIM --DRE DRE_uLSIF --Sampling SIR --epoch_DRE $EPOCH_DRE_MAX --base_lr_DRE 1e-3 --decay_lr_DRE --lr_decay_epochs_DRE 1000 --lambda_DRE $LAMBDA_DRE --DR_final_ActFn ReLU --batch_size_DRE 512 --gmm_ncomp_nsim 165 265 295 190 215 --DRE_save_at_epoch 20 50 100 150 200 300 400 600 800 1000 1200 1500 2000 2500 3000 4000 5000


for epoch_DRE in 20 50 100 150 200 300 400 600 800 1000 1200 1500 2000 2500 3000 4000 5000
do
  echo "-------------------------------------------------------------------------------------------------"
  echo "DRE-uLSIF+SIR lambda $LAMBDA_DRE epoch = $epoch_DRE"
  python3 main.py --NSIM $NSIM --DRE DRE_uLSIF --Sampling SIR --epoch_DRE $epoch_DRE --base_lr_DRE 1e-3 --decay_lr_DRE --lr_decay_epochs_DRE 1000 --lambda_DRE $LAMBDA_DRE --DR_final_ActFn ReLU --batch_size_DRE 512 --gmm_ncomp_nsim 165 265 295 190 215 --gmm_nfake 100000 --compute_dre_err --gmm_ncomp_grid_lb 100 --gmm_ncomp_grid_ub 300 --gmm_ncomp_grid_step 5
done
