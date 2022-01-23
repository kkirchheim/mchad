#!/bin/bash
# Run missing experiments
datasets="cifar10 svhn cifar100"
methods="gcenter gcac"
log_dir="logs/multiruns/missing/$(date +"%D-%T")/"

options="$@"

for ds in $datasets
do
  for method in $methods
  do
    # run 6 seed replicates of each experiment
    python run.py -m experiment="${ds}-${method}" seed="range(1,7,1)" trainer.gpus=1 hydra.sweep.dir="${log_dir}/${ds}/${method}/" ${options}
  done
done
