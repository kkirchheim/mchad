#!/bin/bash
#  helper script for scheduling jobs our slurm cluster
#SBATCH -J mchad  # please replase by short unique self speaking identifier
###SBATCH -N 1         # number of nodes, we only have one
#SBATCH --gres=gpu:v100:8     # type:number of GPU cards
#SBATCH --mem-per-cpu=6000    # main MB per task? max. 500GB/80=6GB
#SBATCH --ntasks-per-node 1   # bigger for mpi-tasks
#SBATCH --cpus-per-task 40    # 10 CPU-threads needed (physcores*2)
#SBATCH --time 24:00:00        # set 0h59min walltime

. /usr/local/bin/slurmProlog.sh  # output slurm settings, debugging
# module load cuda     # latest cuda, or use cuda/10.0 cuda/10.1 cuda/11.2
echo "debug: CUDA_ROOT=$CUDA_ROOT"

log_dir="logs/multiruns/ablation/$(date +'%D-%T')/"
options="++hydra.launcher.ray.remote.num_gpus=0.5 ++hydra.launcher.ray.remote.num_cpus=2 ++hydra.launcher.ray.init.num_cpus=80 ++hydra.launcher.ray.init.num_gpus=8 +trainer.val_check_interval=0.10 $@"


srun python run.py -m experiment="cifar10-gmchad" model.weight_oe="range(0,0.0011,0.0001)" trainer.gpus=1 hydra.sweep.dir="${log_dir}/vary_oe" ${options}


. /usr/local/bin/slurmEpilog.sh   # cleanup
