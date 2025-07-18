#!/bin/bash
# SIMFormer distillation training for simulated dataset
# Uses SIMFormer outputs as pseudo ground-truth

# GPU setup
gpu="0,1,2,3,4,5,6,7"

# Training parameters
lr=1e-4
epochs=500
lrc=32

# Paths to simulated data
trainset="./data/SIM-simulation/*/*/*/train/*.tif"
testset="./data/SIM-simulation/*/*/*/test/*.tif"

# Path to SIMFormer inference results
simformer_infer_save_dir="./results/SIM-simulation"

# Path to pre-trained SIMFormer checkpoint
resume_s1_path="./ckpt/adapter/SIM-simulation/comb_3x/mask_ratio=0.25--add_noise=1.0--lr=0.0001--lrc=32--lp_tv=0.001--s7"

# Single-stage distillation for simulated data
echo "Distillation training on simulated data"
CUDA_VISIBLE_DEVICES=${gpu} python train_distillation.py \
    --trainset="${trainset}" \
    --testset="${testset}" \
    --simformer_infer_save_dir="${simformer_infer_save_dir}" \
    --batchsize=48 \
    --lr=$lr \
    --min_datasize=10000 \
    --epoch=$epochs \
    --mask_ratio=0.0 \
    --add_gaussian_noise=5.0 \
    --shot_noise_scale=0.5 \
    --emitter_lasso=0 \
    --resume_s1_path="${resume_s1_path}" \
    --not_resume_s1_opt \
    --save_dir="./ckpt/distillation/SIM-simulation" \
    --patch_size 3 16 16 \
    --rescale 3 3 \
    --crop_size 80 80 \
    --lrc=$lrc \
    --tag="simulate"