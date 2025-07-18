#!/bin/bash
# SIMFormer distillation training for BioSR dataset
# Uses SIMFormer outputs as pseudo ground-truth

# GPU setup
gpu="0,1,2,3,4,5,6,7"

# Training parameters
lr=1e-4
epochs=1000
lrc=32

# Paths to input data and SIMFormer predictions
trainset="./data/BioSR/*/2D/train/*.tif"
testset="./data/BioSR/*/2D/test/*.tif"

# Path to SIMFormer inference results (must contain test/ subdirectory with predictions)
simformer_infer_save_dir="./results/BioSR"

# Path to pre-trained SIMFormer checkpoint for initialization
resume_s1_path="./ckpt/adapter/2D_BioSR/comb_3x/mask_ratio=0.75--add_noise=1.0--lr=0.0001--lrc=32--lp_tv=0.001--s6"

# Distillation training
CUDA_VISIBLE_DEVICES=${gpu} python train_distillation.py \
    --trainset="${trainset}" \
    --testset="${testset}" \
    --simformer_infer_save_dir="${simformer_infer_save_dir}" \
    --batchsize=36 \
    --lr=$lr \
    --min_datasize=10000 \
    --epoch=$epochs \
    --mask_ratio=0.0 \
    --add_gaussian_noise=1.0 \
    --shot_noise_scale=10.0 \
    --emitter_lasso=0 \
    --resume_s1_path="${resume_s1_path}" \
    --not_resume_s1_opt \
    --save_dir="./ckpt/distillation/BioSR" \
    --patch_size 3 16 16 \
    --rescale 3 3 \
    --crop_size 80 80 \
    --lrc=$lrc