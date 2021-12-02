#!/bin/bash
# Run some missing experiments
datasets="cifar10 svhn cifar100"
log_dir="logs/multiruns/complete/missing/$(date +"%D-%T")/${now:%Y-%m-%d_%H-%M-%S}/"

options="$@"

# python run.py -m experiment="cifar10-mchad-o" seed="range(1,7,1)" trainer.gpus=1 hydra.sweep.dir="${log_dir}/${ds}/mchad-o/" ${options}

for ds in $datasets
do
  python run.py -m experiment="${ds}-softmax" seed="range(1,7,1)" trainer.gpus=1 hydra.sweep.dir="${log_dir}/${ds}/softmax/" ${options}
done
