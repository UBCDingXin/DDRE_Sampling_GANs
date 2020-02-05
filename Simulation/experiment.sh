#!/bin/bash


python3 main.py --NSIM 3 --DRE DRE_SP --Sampling RS --epoch_DRE 5 --base_lr_DRE 1e-3 --lambda_DRE 0 --DR_final_ActFn ReLU --batch_size_DRE 512 --epoch_gan 5
