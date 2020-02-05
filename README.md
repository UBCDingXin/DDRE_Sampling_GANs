# Codes for the experiments in "[Subsampling Generative Adversarial Networks: Density Ratio Estimation in Feature Space with Softplus Loss](https://arxiv.org/abs/1909.10670)"
## 1. To do list
- [x] The 25 2-D Gaussians Simulation
- [x] The CIFAR-10 dataset


## 2. Sample Usage
### Simulation
```
python3 main.py --NSIM 3 --DRE DRE_SP --Sampling RS --epoch_DRE 5 --base_lr_DRE 1e-3 --lambda_DRE 0 --DR_final_ActFn ReLU --batch_size_DRE 512 --epoch_gan 5
```

### CIFAR-10
```
#pre-train a ResNet34 for feature extraction
python3 PreTrainCNN.py --CNN ResNet34 --isometric_map --transform

#pre-train an InceptionV3 for evaluation
python3 PreTrainCNN.py --CNN InceptionV3 --transform

#DCGAN, DRE-F-SP
python3 main.py --GAN DCGAN --DRE DRE_F_SP --Sampling RS --lambda_DRE 0.01 --samp_nfake 50000 --DR_final_ActFn ReLU --epoch_DRE 200 --base_lr_DRE 1e-4
```
