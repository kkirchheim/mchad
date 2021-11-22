#!/bin/bash
# Run several experiments
datasets="cifar10 svhn cifar100"
methods="mchad mchad-o softmax center cac ii"
log_dir="logs/multiruns/complete/$(date +"%D-%T")/${now:%Y-%m-%d_%H-%M-%S}/"

options="$@"

for ds in $datasets
do
  for method in $methods
  do
    d="${log_dir}/${ds}/${method}/"
    python run.py -m experiment="${ds}-${method}" seed="range(1,13,1)" trainer.gpus=1 hydra.sweep.dir=${d}'/${now:%Y-%m-%d_%H-%M-%S}/' ${options}
  done
done

