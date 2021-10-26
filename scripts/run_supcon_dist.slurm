#!/bin/bash

#SBATCH -J supcon_baseline_ddp
#SBATCH -o supcon_baseline_ddp%j.out
#SBATCH -e supcon_baseline_ddp%j.err
#SBATCH --mail-user=swhan@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=1T
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --exclusive

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=pytorch1.7
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes 
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " NGPUs per node:= " $SLURM_GPUS_PER_NODE 
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG
echo " Running on multiple nodes and GPU devices"
echo ""
echo " Run started at:- "
date

python \
      $HOME2/SupContrast/main_supcon_dist.py \
      --data /nobackup/projects/public/imagenet21k_resized/imagenet21k_train/ \
      --workers 16 \
      --epochs 800 \
      --batch_size 256 \
      --learning_rate 1.2 \
      --temp 0.1 \
      --dataset imagenet \
      --print-freq 50 \
      --checkpoint-dir $HOME2/SupContrast/results/ \
      --name supcon-baseline-imagenet

echo "Run completed at:- "
date

