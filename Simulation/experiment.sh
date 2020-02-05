#!/bin/bash


# python3 main.py --NSIM 3 --DRE None --Sampling None --epoch_gan 50 --lr_gan 1e-3 --show_visualization

# python3 main.py --NSIM 3 --DRE disc --Sampling RS --epoch_gan 50 --show_visualization
# python3 main.py --NSIM 3 --DRE disc_MHcal --Sampling RS --epoch_gan 50 --show_visualization
# python3 main.py --NSIM 3 --DRE disc_MHcal --Sampling MH --epoch_gan 50 #--show_visualization
# python3 main.py --NSIM 3 --DRE disc_KeepTrain --Sampling RS --epoch_gan 50 --show_visualization

# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE-uLSIF"
# python3 main.py --NSIM 3 --DRE DRE_uLSIF --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-5 --lambda_DRE 0 --DR_final_ActFn ReLU --batch_size_DRE 512 #--show_visualization
# python3 main.py --NSIM 3 --DRE DRE_uLSIF --Sampling MH --epoch_DRE 400 --base_lr_DRE 1e-5 --lambda_DRE 0 --DR_final_ActFn ReLU --batch_size_DRE 512 --show_visualization
# python3 main.py --NSIM 3 --DRE DRE_uLSIF --Sampling SIR --epoch_DRE 400 --base_lr_DRE 1e-5 --lambda_DRE 0 --DR_final_ActFn ReLU --batch_size_DRE 512 --show_visualization
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE-DSKL"
# python3 main.py --NSIM 3 --DRE DRE_DSKL --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-5 --lambda_DRE 0 --DR_final_ActFn ReLU --batch_size_DRE 512 #--show_visualization
# python3 main.py --NSIM 3 --DRE DRE_DSKL --Sampling MH --epoch_DRE 400 --base_lr_DRE 1e-5 --lambda_DRE 0 --DR_final_ActFn ReLU --batch_size_DRE 512 --show_visualization
# python3 main.py --NSIM 3 --DRE DRE_DSKL --Sampling SIR --epoch_DRE 400 --base_lr_DRE 1e-5 --lambda_DRE 0 --DR_final_ActFn ReLU --batch_size_DRE 512 --show_visualization
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE-BARR+RS"
# python3 main.py --NSIM 3 --DRE DRE_BARR --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-5 --lambda_DRE 10 --DR_final_ActFn ReLU --batch_size_DRE 512 #--show_visualization
# python3 main.py --NSIM 3 --DRE DRE_BARR --Sampling MH --epoch_DRE 400 --base_lr_DRE 1e-5 --lambda_DRE 10 --DR_final_ActFn ReLU --batch_size_DRE 512 --show_visualization
# python3 main.py --NSIM 3 --DRE DRE_BARR --Sampling SIR --epoch_DRE 400 --base_lr_DRE 1e-5 --lambda_DRE 10 --DR_final_ActFn ReLU --batch_size_DRE 512 --show_visualization


# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE-SP+RS with different lambda's"
# echo "DRE-SP+RS lambda 0"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE-SP+RS lambda 0.005"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.005 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE-SP+RS lambda 0.01"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.01 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE-SP+RS lambda 0.05"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.05 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE-SP+RS lambda 0.1"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.1 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE-SP+RS lambda 0.5"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.5 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE-SP+RS lambda 1"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 1 --DR_final_ActFn ReLU --batch_size_DRE 512

# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE-SP+MH with different lambda's"
# echo "DRE-SP+MH lambda 0"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling MH --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "DRE-SP+MH lambda 0.005"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling MH --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.005 --DR_final_ActFn ReLU --batch_size_DRE 512 --show_visualization
# echo "DRE-SP+MH lambda 0.01"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling MH --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.01 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "DRE-SP+MH lambda 0.05"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling MH --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.05 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "DRE-SP+MH lambda 0.1"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling MH --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.1 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "DRE-SP+MH lambda 0.5"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling MH --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.5 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "DRE-SP+MH lambda 1"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling MH --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 1 --DR_final_ActFn ReLU --batch_size_DRE 512
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE-SP+SIR with different lambda's"
# echo "DRE-SP+SIR lambda 0"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling SIR --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "DRE-SP+SIR lambda 0.005"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling SIR --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.005 --DR_final_ActFn ReLU --batch_size_DRE 512 --show_visualization
# echo "DRE-SP+SIR lambda 0.01"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling SIR --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.01 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "DRE-SP+SIR lambda 0.05"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling SIR --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.05 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "DRE-SP+SIR lambda 0.1"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling SIR --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.1 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "DRE-SP+SIR lambda 0.5"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling SIR --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.5 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "DRE-SP+SIR lambda 1"
# python3 main.py --NSIM 3 --DRE DRE_SP --Sampling SIR --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 1 --DR_final_ActFn ReLU --batch_size_DRE 512


# echo "-------------------------------------------------------------------------------------------------"
# echo "Compare different loss functions: uLSIF vs SP with penalty term; only one round; only need their training loss"
# ### DR on valid set: 1.762, 7.222, 11.644; prop. good samples: 88.9 (0.0)
# python3 main.py --NSIM 1 --DRE DRE_uLSIF --Sampling RS --epoch_DRE 1000 --base_lr_DRE 1e-5 --lambda_DRE 0 --DR_final_ActFn ReLU --batch_size_DRE 512
# ### DR on valid set: 1.380, 7.467, 12.063; prop. good samples: 89.4 (0.0)
# python3 main.py --NSIM 1 --DRE DRE_uLSIF --Sampling RS --epoch_DRE 1000 --base_lr_DRE 1e-5 --lambda_DRE 0.1 --DR_final_ActFn ReLU --batch_size_DRE 512
# ### DR on valid set: 1.459, 7.592, 12.063; prop. good samples: 89.7 (0.0)
# python3 main.py --NSIM 1 --DRE DRE_uLSIF --Sampling RS --epoch_DRE 1000 --base_lr_DRE 1e-5 --lambda_DRE 1 --DR_final_ActFn ReLU --batch_size_DRE 512
# python3 main.py --NSIM 1 --DRE DRE_uLSIF --Sampling RS --epoch_DRE 1000 --base_lr_DRE 1e-5 --lambda_DRE 3 --DR_final_ActFn ReLU --batch_size_DRE 512
### DR on valid set: ; prop. good samples: 87.9 (0.0)
# python3 main.py --NSIM 1 --DRE DRE_uLSIF --Sampling RS --epoch_DRE 1000 --base_lr_DRE 1e-5 --lambda_DRE 5 --DR_final_ActFn ReLU --batch_size_DRE 512
# ### DR on valid set: 1.103, 5.688, 7.822; prop. good samples: 86.0 (0.0)
# python3 main.py --NSIM 1 --DRE DRE_uLSIF --Sampling RS --epoch_DRE 1000 --base_lr_DRE 1e-5 --lambda_DRE 10 --DR_final_ActFn ReLU --batch_size_DRE 512

### DR on valid set: 11.324, 10.593, 7.185; prop. good samples: 99.5 (0.0)
# python3 main.py --NSIM 1 --DRE DRE_SP --Sampling RS --epoch_DRE 1000 --base_lr_DRE 1e-3 --decay_lr_DRE --lambda_DRE 0 --DR_final_ActFn ReLU --batch_size_DRE 512
### DR on valid set: 13.640, 10.290, 5.711; prop. good samples: 99.5 (0.0)
# python3 main.py --NSIM 1 --DRE DRE_SP --Sampling RS --epoch_DRE 1000 --base_lr_DRE 1e-3 --decay_lr_DRE --lambda_DRE 0.1 --DR_final_ActFn ReLU --batch_size_DRE 512
### DR on valid set: 8.012, 6.177, 3.249; prop. good samples: 94.0 (0.0)
# python3 main.py --NSIM 1 --DRE DRE_SP --Sampling RS --epoch_DRE 1000 --base_lr_DRE 1e-3 --decay_lr_DRE --lambda_DRE 1 --DR_final_ActFn ReLU --batch_size_DRE 512
# python3 main.py --NSIM 1 --DRE DRE_SP --Sampling RS --epoch_DRE 1000 --base_lr_DRE 1e-3 --decay_lr_DRE --lambda_DRE 3 --DR_final_ActFn ReLU --batch_size_DRE 512
## prop. good samples: 82.5 (0.0)
# python3 main.py --NSIM 1 --DRE DRE_SP --Sampling RS --epoch_DRE 1000 --base_lr_DRE 1e-3 --decay_lr_DRE --lambda_DRE 5 --DR_final_ActFn ReLU --batch_size_DRE 512
### DR on valid set: 2.829, 2.646, 1.498; prop. good samples: 83.1 (0.0)
# python3 main.py --NSIM 1 --DRE DRE_SP --Sampling RS --epoch_DRE 1000 --base_lr_DRE 1e-3 --decay_lr_DRE --lambda_DRE 10 --DR_final_ActFn ReLU --batch_size_DRE 512




# echo "-------------------------------------------------------------------------------------------------"
# echo "DRE-BARR with different lambdas"
# echo "DRE-BARR+RS lambda 0"
# python3 main.py --NSIM 1 --DRE DRE_BARR --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "DRE-BARR+RS lambda 0.01"
# python3 main.py --NSIM 1 --DRE DRE_BARR --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.01 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "DRE-BARR+RS lambda 0.1"
# python3 main.py --NSIM 1 --DRE DRE_BARR --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 0.1 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "DRE-BARR+RS lambda 1"
# python3 main.py --NSIM 1 --DRE DRE_BARR --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 1 --DR_final_ActFn ReLU --batch_size_DRE 512
# echo "DRE-BARR+RS lambda 10"
# python3 main.py --NSIM 1 --DRE DRE_BARR --Sampling RS --epoch_DRE 400 --base_lr_DRE 1e-3 --lambda_DRE 10 --DR_final_ActFn ReLU --batch_size_DRE 512
