#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSsgwh
PYTHON_VIRTUAL_ENVIRONMENT=pytorch1.7
CONDA_ROOT=$HOME2/barn/miniconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

python \
      $HOME2/scratch/test/SupContrast/main_dist_linear_eval_multi.py \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/train/ \
      --test-data /gpfs/u/locker/200/CADS/datasets/ImageNet/val/ \
      --workers 16 \
      --nodes 1 \
      --ngpus 4 \
      --epochs 100 \
      --batch-size 512 \
      --learning-rate 5.0 \
      --momentum 0.9 \
      --weight-decay 0.0 \
      --print-freq 50 \
      --checkpoint-dir $HOME2/scratch/test/SupContrast/results/ \
      --checkpoint-path $HOME2/scratch/top5/SupContrast/results/supcon-baseline-imagenet/checkpoint_799 \
      --log-dir $HOME2/scratch/test/SupContrast/logs/ \
      --name supcon-baseline-linear-eval

echo "Run completed at:- "
date

