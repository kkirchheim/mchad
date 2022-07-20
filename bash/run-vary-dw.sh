#!/bin/bash
# Ablation study
# Vary the depth and width on the network
log_dir="logs/multiruns/experiment/$(date +'%D-%T')/"
options="$@"


n_gpus=21

options="++hydra.launcher.ray.remote.num_gpus=1 ++hydra.launcher.ray.remote.num_cpus=2 ++hydra.launcher.ray.init.num_cpus=40 ++hydra.launcher.ray.init.num_gpus=${n_gpus} ${@}"
log_dir="logs/multiruns/experiment/vary-dw/$(date +"%D-%T")/"

python run.py -m experiment="svhn-mchad-myresnet" seed="1234" model.backbone.depth="range(1,32)" model.backbone.width="range(1,32)" trainer.gpus=1 hydra.sweep.dir="${log_dir}/" ${options}
