#!/bin/bash
# Run all experiments
datasets=("cifar10" "svhn" "cifar100")
methods=("mchad" "gmchad" "gcenter" "gcac" "center" "cac" "ii")
log_dir="logs/multiruns/complete/$(date +"%D-%T")/"

options="$@"

for ds in ${datasets[*]}
do
  for method in ${methods[*]}
  do
    python run.py -m experiment="${ds}-${method}" seed="range(1,22,1)" trainer.gpus=1 hydra.sweep.dir="${log_dir}/${ds}/${method}/" "${options}"
  done
done
