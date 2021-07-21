#!/bin/bash
# Run several experiments
log_dir="logs/multiruns/ablation/$(date +'%D-%T')/"
options="$@ +trainer.val_check_interval=0.10" # check 10 times each epoch

python run.py -m experiment="cifar10-gmchad" model.weight_oe="range(0,0.000011,0.000001)" trainer.min_epochs=100 trainer.gpus=1 hydra.sweep.dir="${log_dir}/vary_oe" dataset_dir="/data_fast/kirchheim/datasets/" ${options}
python run.py -m experiment="cifar10-gmchad" model.weight_ce="range(0,1.1,0.1)" trainer.min_epochs=100 trainer.gpus=1 hydra.sweep.dir="${log_dir}/vary_ce" dataset_dir="/data_fast/kirchheim/datasets/" ${options}
python run.py -m experiment="cifar10-gmchad" model.weight_center="range(0,1.1,0.1)" trainer.min_epochs=100  trainer.gpus=1 hydra.sweep.dir="${log_dir}/vary_center" dataset_dir="/data_fast/kirchheim/datasets/" ${options}
