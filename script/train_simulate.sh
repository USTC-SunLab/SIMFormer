gpu="0,1,2,3,4,5,6,7"

lr=1e-4
epochs=70
initial_lrc=512
lrc=$initial_lrc

CUDA_VISIBLE_DEVICES=${gpu} python train.py \
        --trainset="./data/SIM-simulation/*/*/*/train/*.tif" \
        --testset="./data/SIM-simulation/*/*/*/test/*.tif" \
        --batchsize=36 \
        --lr=$lr \
        --min_datasize=10000 \
        --epoch=$epochs \
        --mask_ratio=0.75 \
        --add_noise=1.0 \
        --resume_pretrain \
        --save_dir="./ckpt/adapter/SIM-simulation/comb_3x" \
        --patch_size 3 16 16 \
        --rescale 3 3 \
        --crop_size 80 80 \
        --psf_size 49 49 \
        --lrc=$lrc \
        --not_resume_s1_opt \
        --use_gt


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
    
    CUDA_VISIBLE_DEVICES=${gpu} python train.py \
            --trainset="./data/SIM-simulation/*/*/*/train/*.tif" \
            --testset="./data/SIM-simulation/*/*/*/test/*.tif" \
            --batchsize=18 \
            --lr=$lr \
            --min_datasize=10000 \
            --epoch=$epochs \
            --mask_ratio=0.25 \
            --add_noise=1.0 \
            --resume_pretrain \
            --resume_s1_path="./ckpt/adapter/SIM-simulation/comb_3x/lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=${mask_ratio}--lrc=${lrc}--s${s_value}" \
            --save_dir="./ckpt/adapter/SIM-simulation/comb_3x" \
            --patch_size 3 16 16 \
            --rescale 3 3 \
            --crop_size 80 80 \
            --psf_size 49 49 \
            --lrc=$next_lrc \
            --use_gt \
            --not_resume_s1_opt \
            --resume

    lrc=$next_lrc
    s_value=$((s_value + 1))

done

# Stage 6: Extended training
CUDA_VISIBLE_DEVICES=${gpu} python train.py \
        --trainset="./data/SIM-simulation/*/*/*/train/*.tif" \
        --testset="./data/SIM-simulation/*/*/*/test/*.tif" \
        --batchsize=9 \
        --lr=$lr \
        --min_datasize=10000 \
        --epoch=2000 \
        --mask_ratio=0.75 \
        --add_noise=1.0 \
        --resume_pretrain \
        --resume_s1_path="./ckpt/adapter/SIM-simulation/comb_3x/lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.25--lrc=${lrc}--s5" \
        --save_dir="./ckpt/adapter/SIM-simulation/comb_3x" \
        --patch_size 3 16 16 \
        --rescale 3 3 \
        --crop_size 80 80 \
        --psf_size 49 49 \
        --lrc=$next_lrc \
        --use_gt \
        --not_resume_s1_opt

# Stage 7: Additional training with low mask ratio
CUDA_VISIBLE_DEVICES=${gpu} python train.py \
        --trainset="./data/SIM-simulation/*/*/*/train/*.tif" \
        --testset="./data/SIM-simulation/*/*/*/test/*.tif" \
        --batchsize=9 \
        --lr=$lr \
        --min_datasize=10000 \
        --epoch=300 \
        --mask_ratio=0.25 \
        --add_noise=1.0 \
        --resume_pretrain \
        --resume_s1_path="./ckpt/adapter/SIM-simulation/comb_3x/lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.75--lrc=${lrc}--s6" \
        --save_dir="./ckpt/adapter/SIM-simulation/comb_3x" \
        --patch_size 3 16 16 \
        --rescale 3 3 \
        --crop_size 80 80 \
        --psf_size 49 49 \
        --lrc=$next_lrc \
        --use_gt \
        --not_resume_s1_opt
