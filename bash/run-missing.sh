#!/bin/bash
# Run all experiments
datasets=("svhn")
methods=("mchad" "gcac")
log_dir="logs/multiruns/complete/$(date +"%D-%T")/"

options="$@"

for ds in ${datasets[*]}
do
  for method in ${methods[*]}
  do
    python run.py -m experiment="${ds}-${method}" seed="range(1,22)" trainer.gpus=1 hydra.sweep.dir="${log_dir}/${ds}/${method}/" "${options}"
  done
done
