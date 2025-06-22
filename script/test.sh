#!/bin/bash

# SIM-simulation beads finetune test
path="./ckpt/finetune/beads/lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.75--lrc=32--s2"

CUDA_VISIBLE_DEVICES="0" python infer.py \
    --data_dir="./data/SIM-simulation/beads/standard/test/*.tif" \
    --resume_path=${path} \
    --save_dir="./result/SIM-simulation/beads"