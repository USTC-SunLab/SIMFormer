#!/bin/bash
# SIMFormer distillation inference script
# Process data using distilled models

# GPU setup
gpu="0"

# Example 1: BioSR test data
echo "Testing distillation model on BioSR data"
CUDA_VISIBLE_DEVICES=${gpu} python test_distillation.py \
    --data_dir="./data/BioSR/CCPs/2D/test/*.tif" \
    --resume_path="./ckpt/distillation/BioSR/lr=0.0001--add_gaussian_noise=1.0--shot_noise_scale=1.0--lrc=32" \
    --save_dir="./results/distillation/BioSR/CCPs" \
    --batchsize=1 \
    --crop_size 256 256 \
    --patch_size 3 16 16 \
    --rescale 3 3 \
    --save_format tif

# Example 2: Simulated test data with specific noise level
echo "Testing on simulated data with specific noise level"
CUDA_VISIBLE_DEVICES=${gpu} python test_distillation.py \
    --data_dir="./data/SIM-simulation/curve/ave_photon/10/test/*.tif" \
    --resume_path="./ckpt/distillation/SIM-simulation/lr=0.0001--add_gaussian_noise=0.5--shot_noise_scale=1.0--lrc=32--simulate" \
    --save_dir="./results/distillation/SIM-simulation/curve/ave_photon/10" \
    --batchsize=1 \
    --crop_size 256 256 \
    --patch_size 3 16 16 \
    --rescale 3 3 \
    --save_format tif

# Example 3: Process custom microscopy data
echo "Testing on custom microscopy data"
CUDA_VISIBLE_DEVICES=${gpu} python test_distillation.py \
    --data_dir="./data/custom/*.tif" \
    --resume_path="./ckpt/distillation/BioSR/lr=0.0001--add_gaussian_noise=2.0--shot_noise_scale=10.0--lrc=32" \
    --resume_iter=5000 \
    --save_dir="./results/distillation/custom" \
    --batchsize=1 \
    --crop_size 512 512 \
    --patch_size 3 16 16 \
    --rescale 3 3 \
    --overlap 64 \
    --save_format png

# Example 4: Sequential processing for time-lapse data
echo "Processing time-lapse data sequentially"
CUDA_VISIBLE_DEVICES=${gpu} python test_distillation.py \
    --data_dir="./data/timelapse/cell_*.tif" \
    --resume_path="./ckpt/distillation/BioSR/lr=0.0001--add_gaussian_noise=1.0--shot_noise_scale=5.0--mask_ratio=0.25--lrc=32" \
    --save_dir="./results/distillation/timelapse" \
    --batchsize=1 \
    --crop_size 256 256 \
    --patch_size 3 16 16 \
    --rescale 3 3 \
    --sequential_mode \
    --save_format tif