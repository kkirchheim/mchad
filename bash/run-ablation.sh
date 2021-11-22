#!/bin/bash
# Run several experiments
log_dir="logs/multiruns/ablation/$(date +"%D-%T")/${now:%Y-%m-%d_%H-%M-%S}/"
options="$@"

python run.py -m experiment="cifar10-mchad" model.weight_oe="range(0,1.1,0.1)" trainer.gpus=1 hydra.sweep.dir=${log_dir} ${options}
python run.py -m experiment="cifar10-mchad" model.weight_ce="range(0,1.1,0.1)" trainer.gpus=1 hydra.sweep.dir=${log_dir} ${options}
python run.py -m experiment="cifar10-mchad" model.weight_center="range(0,1.1,0.1)" trainer.gpus=1 hydra.sweep.dir=${log_dir} ${options}
