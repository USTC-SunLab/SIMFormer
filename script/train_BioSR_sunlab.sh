#!/bin/bash
# BioSR multi-stage training

# GPU setup
gpu="2,1,0"

# Stage 1: Initial training
lr=1e-4
epochs=70
initial_lrc=512
lrc=$initial_lrc

CUDA_VISIBLE_DEVICES=${gpu} python3 train.py \
        --trainset="./data/BioSR/*/2D/gt/*.tif" \
        --testset="./data/BioSR/*/2D/gt/*.tif" \
        --batchsize=96 \
        --lr=$lr \
        --min_datasize=50000 \
        --epoch=$epochs \
        --mask_ratio=0.75 \
        --add_noise=1.0 \
        --resume_pretrain \
        --save_dir="./ckpt/adapter/2D_BioSR/comb_3x" \
        --patch_size 3 16 16 \
        --rescale 3 3 \
        --crop_size 80 80 \
        --psf_size 49 49 \
        --lrc=$lrc

# Stages 2-5: Progressive LRC reduction
epochs=30
s_value=1
end_lrc=32

while [ $lrc -gt $end_lrc ]; do
    next_lrc=$((lrc / 2))
    
    if [ $s_value -eq 1 ]; then
        mask_ratio=0.75
    else
        mask_ratio=0.25
    fi
    
    CUDA_VISIBLE_DEVICES=${gpu} python3 train.py \
            --trainset="./data/BioSR/*/2D/gt/*.tif" \
            --testset="./data/BioSR/*/2D/gt/*.tif" \
            --batchsize=64 \
            --lr=$lr \
            --min_datasize=50000 \
            --epoch=$epochs \
            --mask_ratio=0.25 \
            --add_noise=1.0 \
            --resume_pretrain \
            --resume_s1_path="./ckpt/adapter/2D_BioSR/comb_3x/mask_ratio=${mask_ratio}--add_noise=1.0--lr=0.0001--lrc=${lrc}--lp_tv=0.001--s${s_value}" \
            --save_dir="./ckpt/adapter/2D_BioSR/comb_3x" \
            --patch_size 3 16 16 \
            --rescale 3 3 \
            --crop_size 80 80 \
            --psf_size 49 49 \
            --lrc=$next_lrc \
            --not_resume_s1_opt \
            --resume
    
    lrc=$next_lrc
    s_value=$((s_value + 1))
done

# Stage 6: Extended training
CUDA_VISIBLE_DEVICES=${gpu} python3 train.py \
        --trainset="./data/BioSR/*/2D/gt/*.tif" \
        --testset="./data/BioSR/*/2D/gt/*.tif" \
        --batchsize=18 \
        --lr=$lr \
        --min_datasize=20000 \
        --epoch=2000 \
        --mask_ratio=0.75 \
        --add_noise=1.0 \
        --resume_pretrain \
        --resume_s1_path="./ckpt/adapter/2D_BioSR/comb_3x/mask_ratio=0.25--add_noise=1.0--lr=0.0001--lrc=${lrc}--lp_tv=0.001--s5" \
        --save_dir="./ckpt/adapter/2D_BioSR/comb_3x" \
        --patch_size 3 16 16 \
        --rescale 3 3 \
        --crop_size 80 80 \
        --psf_size 49 49 \
        --lrc=$next_lrc \
        --not_resume_s1_opt
