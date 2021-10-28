#!/bin/bash
#   last-update: 2019-11-05, 2021-02-03
#SBATCH -J tune-mchad  # please replase by short unique self speaking identifier
###SBATCH -N 1         # number of nodes, we only have one
#SBATCH --gres=gpu:v100:8     # type:number of GPU cards
#SBATCH --mem-per-cpu=6000    # main MB per task? max. 500GB/80=6GB
#SBATCH --ntasks-per-node 1   # bigger for mpi-tasks
#SBATCH --cpus-per-task 40    # 10 CPU-threads needed (physcores*2)
#SBATCH --time 24:00:00        # set 0h59min walltime

. /usr/local/bin/slurmProlog.sh  # output slurm settings, debugging
# module load cuda     # latest cuda, or use cuda/10.0 cuda/10.1 cuda/11.2
echo "debug: CUDA_ROOT=$CUDA_ROOT"
# conda activate myenv

datasets="cifar10"
methods="softmax"
options="++hydra.launcher.ray.remote.num_gpus=0.5 ++hydra.launcher.ray.remote.num_cpus=2 ++hydra.launcher.ray.init.num_cpus=80 ++hydra.launcher.ray.init.num_gpus=8"


for ds in $datasets
do
  for meth in $methods
  do
    srun python run.py -m experiment="${ds}-${meth}" seed="range(1000,16000,1000)" trainer.gpus=1 ${options}
  done
done


conda deactivate  # workoff :)
. /usr/local/bin/slurmEpilog.sh   # cleanup

