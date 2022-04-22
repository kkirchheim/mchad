#!/bin/bash
#  helper script for scheduling jobs our slurm cluster
#SBATCH -J tune-mchad  # please replase by short unique self speaking identifier
###SBATCH -N 1         # number of nodes, we only have one
#SBATCH --gres=gpu:v100:1     # type:number of GPU cards
#SBATCH --mem-per-cpu=6000    # main MB per task? max. 500GB/80=6GB
#SBATCH --ntasks-per-node 1   # bigger for mpi-tasks
#SBATCH --cpus-per-task 10    # 10 CPU-threads needed (physcores*2)
#SBATCH --time 48:00:00        # set 0h59min walltime

. /usr/local/bin/slurmProlog.sh  # output slurm settings, debugging
# module load cuda     # latest cuda, or use cuda/10.0 cuda/10.1 cuda/11.2
echo "debug: CUDA_ROOT=$CUDA_ROOT"

options="++hydra.launcher.ray.remote.num_gpus=1.0 ++hydra.launcher.ray.remote.num_cpus=2 ++hydra.launcher.ray.init.num_cpus=40 ++hydra.launcher.ray.init.num_gpus=1 $@"


srun python run.py experiment="cifar100-cac" seed="13" trainer.gpus=1 hydra.sweep.dir="${log_dir}/${ds}/${method}/" ${options}


. /usr/local/bin/slurmEpilog.sh   # cleanup
