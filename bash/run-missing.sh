#!/bin/bash
# Run some missing experiments
datasets="cifar10 svhn cifar100"
log_dir="logs/multiruns/complete/missing/$(date +"%D-%T")/${now:%Y-%m-%d_%H-%M-%S}/"

options="$@"

#for ds in $datasets
#do
#  python run.py -m experiment="${ds}-mchad" seed="range(1,13,1)" trainer.gpus=1 hydra.sweep.dir="${log_dir}/${ds}-mchad/" ${options}
#  python run.py -m experiment="${ds}-mchad-o" seed="range(1,13,1)" trainer.gpus=1 hydra.sweep.dir="${log_dir}/${ds}-mchad-o/" ${options}
#  python run.py -m experiment="${ds}-center" seed="range(1,13,1)" trainer.gpus=1 hydra.sweep.dir="${log_dir}/${ds}-center/" ${options}
#done

python run.py -m experiment="cifar100-softmax" seed="range(1,13,1)" trainer.gpus=1 hydra.sweep.dir="${log_dir}/${now:%Y-%m-%d_%H-%M-%S}/" ${options}
