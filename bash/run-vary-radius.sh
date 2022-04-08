#!/bin/bash
# Run several experiments
log_dir="logs/multiruns/ablation/$(date +'%D-%T')/"
options="$@ +trainer.val_check_interval=0.10" # check 4 times each epoch

python run.py -m experiment="cifar10-mchad" +model.radius="range(0,2.1,0.1)" trainer.gpus=1 hydra.sweep.dir="${log_dir}/vary_radius" ${options}
python run.py -m experiment="cifar10-gmchad" +model.radius="range(0,2.1,0.1)" trainer.gpus=1 hydra.sweep.dir="${log_dir}/vary_radius" ${options}
