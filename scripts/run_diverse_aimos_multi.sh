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
      $HOME2/scratch/diverse/SupContrast/main_dist_multi.py \
      --data $HOME2/scratch/diverse/SupContrast/data/ \
      --workers 16 \
      --nodes 1 \
      --ngpus 6 \
      --arch mlp \
      --epochs 100 \
      --batch-size 384 \
      --learning-rate 0.5 \
      --temp 0.1 \
      --color-std 0.0 \
      --num-colors 10 \
      --lamb 0.001 \
      --train_stage1 \
      --train_stage2 \
      --train_stage3 \
      --print-freq 50 \
      --checkpoint-dir $HOME2/scratch/diverse/SupContrast/results/ \
      --log-dir $HOME2/scratch/diverse/SupContrast/logs/ \
      --name infonce-diverse-sequential-randomcolor-epoch100-lambda0.001

echo "Run completed at:- "
date

