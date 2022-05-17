#!/bin/bash

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSsgwh
PYTHON_VIRTUAL_ENVIRONMENT=pytorch1.7
CONDA_ROOT=$HOME2/scratch/miniconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

python $HOME2/scratch/SupContrast-Ligong/launcher_main2_supcon.py \
    --batch_size 512 --epochs 1000 --size 64 \
    --wandb_project simclr-t64 \
    --dataset path --data_folder "/gpfs/u/home/BNSS/BNSSsgwh/scratch/data/tiny-imagenet-200" \
    --mean "(0.480, 0.448, 0.398)" \
    --std "(0.277, 0.269, 0.282)" \
    --log_dir ./logs/0515_main-t64_VG2+A2_crop=small_2 \
    --alpha 0.5 --append_view \
    --pos_view_paths "/gpfs/u/home/BNSS/BNSSsgwh/scratch/data/data_t64/t64_noise=0.2.pkl" \
    --model resnet18 \
    --learning_rate 0.5 \
    --temp 0.5 --syncBN \
    --save_freq 1 \
    --nodes 1 \
    --num_workers 32 \
    --cosine --method 'essl+pos_app'

echo "Run completed at:- "
date

