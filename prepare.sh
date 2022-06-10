#!/bin/bash
# you will need wget and anaconda

# download weights
cd data
wget "https://github.com/hendrycks/pre-training/raw/master/uncertainty/CIFAR/snapshots/imagenet/cifar10_excluded/imagenet_wrn_baseline_epoch_99.pt"
cd -

# setup environment
conda env create --name mchad2 -f environment.yaml
conda activate mchad2
pip install aiohttp==3.7 async-timeout==3.0.1 pytorch-ood=0.0.6 tensorboardX==2.5.1
