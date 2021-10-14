#!/bin/bash

python main_supcon.py --batch_size 128   --learning_rate 0.5   --temp 0.1 --dataset cifar100 --trial 1 > supcon_randneg_bs128.txt