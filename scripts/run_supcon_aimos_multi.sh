#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSsgwh
PYTHON_VIRTUAL_ENVIRONMENT=pytorch1.7
CONDA_ROOT=$HOME2/scratch/miniconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

python \
      $HOME2/scratch/baseline/SupContrast/main_dist_multi.py \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/train/ \
      --workers 24 \
      --nodes 4 \
      --ngpus 6 \
      --epochs 800 \
      --batch-size 1536 \
      --learning-rate 1.2 \
      --temp 0.1 \
      --print-freq 50 \
      --checkpoint-dir $HOME2/scratch/baseline/SupContrast/results/ \
      --log-dir $HOME2/scratch/baseline/SupContrast/logs/ \
      --name supcon-baseline-imagenet-multi

echo "Run completed at:- "
date

