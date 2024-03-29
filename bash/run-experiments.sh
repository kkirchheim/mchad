#!/bin/bash
# Run all experiments
datasets=("cifar10" "svhn" "cifar100")
methods=("gmchad" "gcenter" "gcac" "mchad" "center" "cac" "ii")
log_dir="logs/multiruns/complete/$(date +"%D-%T")/"

for ds in ${datasets[*]}
do
  for method in ${methods[*]}
  do
    echo "run ${method} ${ds}"
    python run.py -m experiment="${ds}-${method}" seed="range(1,22)" trainer.gpus=1 dataset_dir="/data_fast/kirchheim/datasets/" ++model.pretrained_checkpoint='${data_dir}/imagenet_wrn_baseline_epoch_99.pt' hydra.sweep.dir="${log_dir}/${ds}/${method}/" ${@}
    # kill remaining jobs
    kill  $(ps -ae -o pid,args | grep "ray::_run_job()" | cut -f 1 -d " ")
  done
done
