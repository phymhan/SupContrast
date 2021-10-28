#!/bin/bash

#SBATCH -J supcon_baseline
#SBATCH -o supcon_baseline%j.out
#SBATCH -e supcon_baseline%j.err
#SBATCH --mail-user=swhan@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:32g:6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=500g
#SBATCH --time=6:00:00
#SBATCH --exclusive

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSsgwh/
PYTHON_VIRTUAL_ENVIRONMENT=pytorch1.7
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

# SLURM_NPROCS and SLURM_NTASK_PER_NODE env variables are set by sbatch Slurm commands based on the SBATCH directives above
# or options specified on the command line.
if [ "x$SLURM_NPROCS" = "x" ]
then
  if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
  then
    SLURM_NTASKS_PER_NODE=1
  fi
  SLURM_NPROCS=`expr $SLURM_JOB_NUM_NODES \* $SLURM_NTASKS_PER_NODE`
else
  if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
  then
    SLURM_NTASKS_PER_NODE=`expr $SLURM_NPROCS / $SLURM_JOB_NUM_NODES`
  fi
fi

# Get the host name of the allocated compute node(s) and generate the host list file.
srun hostname -s | sort -u > ~/tmp/hosts.$SLURM_JOBID
awk "{ print \$0 \"-ib slots=$SLURM_NTASKS_PER_NODE\"; }" ~/tmp/hosts.$SLURM_JOBID >~/tmp/tmp.$SLURM_JOBID
mv ~/tmp/tmp.$SLURM_JOBID ~/tmp/hosts.$SLURM_JOBID

python \
      $HOME2/scratch/SupContrast/main_dist.py \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/train/ \
      --workers 16 \
      --epochs 800 \
      --batch-size 768 \
      --learning-rate 1.2 \
      --temp 0.1 \
      --print-freq 50 \
      --checkpoint-dir $HOME2/scratch/SupContrast/results/ \
      --log-dir $HOME2/scratch/SupContrast/logs/ \
      --name supcon-baseline-imagenet

echo "Run completed at:- "
date

