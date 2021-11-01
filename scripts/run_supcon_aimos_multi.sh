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
      $HOME2/scratch/top5/SupContrast/main_dist_multi.py \
      --data /gpfs/u/locker/200/CADS/datasets/ImageNet/train/ \
      --workers 16 \
      --nodes 2 \
      --ngpus 6 \
      --epochs 800 \
      --batch-size 768 \
      --learning-rate 1.2 \
      --temp 0.1 \
      --print-freq 50 \
      --checkpoint-dir $HOME2/scratch/top5/SupContrast/results/ \
      --log-dir $HOME2/scratch/SupContrast/logs/ \
      --top5-path $HOME2/scratch/top5/SupContrast/imagenet_top5.pkl \
      --name supcon-negboost-imagenet-multi

echo "Run completed at:- "
date

