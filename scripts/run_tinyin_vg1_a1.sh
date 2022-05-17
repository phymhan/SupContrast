#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSsgwh
PYTHON_VIRTUAL_ENVIRONMENT=pytorch1.7
CONDA_ROOT=$HOME2/scratch/miniconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

python $HOME2/scratch/SupContrast-Ligong/launcher_main4.py \
    --batch_size 512 --epochs 1000 --size 64 \
    --wandb_project simclr-t64 \
    --dataset path --data_folder "/gpfs/u/home/BNSS/BNSSsgwh/scratch/data/tiny-imagenet-200-1" \
    --mean "(0.480, 0.448, 0.398)" \
    --std "(0.277, 0.269, 0.282)" \
    --log_dir ./logs/0515_main-t64_VG1+A1_crop=small \
    --pos_view_paths "/gpfs/u/home/BNSS/BNSSsgwh/scratch/data/data_t64/t64_eps1=0.2_eps2=0.31_uint8.pkl" --uint8 \
    --model resnet18 --syncBN \
    --learning_rate 0.5 --temp 0.5 --alpha 0.5 \
    --cosine --method 'simclr' --setting 'v1=expert,v2=gan' \
    --save_freq 1 \
    --nodes 1 \
    --num_workers 32 \
    --add_randomcrop

echo "Run completed at:- "
date

