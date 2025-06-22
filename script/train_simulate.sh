gpu="0,1,2,3,4,5,6,7"

lr=1e-4
epochs=70
initial_lrc=512
lrc=$initial_lrc

CUDA_VISIBLE_DEVICES=${gpu} python train.py \
        --trainset="/data_nas/nas/Research/Datasets/SIM_simulation/*/*/*/train/*.tif" \
        --testset="/data_nas/nas/Research/Datasets/SIM_simulation/*/*/*/test/*.tif" \
        --batchsize=36 \
        --lr=$lr \
        --min_datasize=10000 \
        --epoch=$epochs \
        --mask_ratio=0.75 \
        --add_noise=1.0 \
        --resume_pretrain \
        --save_dir="./ckpt/adapter/SIM_simulation/comb_3x" \
        --patch_size 3 16 16 \
        --rescale 3 3 \
        --crop_size 80 80 \
        --psf_size 49 49 \
        --lrc=$lrc \
        --accumulation_step=2 \
        --not_resume_s1_opt \
        --use_gt \
        --use_mp \
        --use_remat


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
    
    # CUDA_VISIBLE_DEVICES=${gpu} python train.py \
    #         --trainset="/data_nas/nas/Research/Datasets/SIM_simulation/*/*/*/train/*.tif" \
    #         --testset="/data_nas/nas/Research/Datasets/SIM_simulation/*/*/*/test/*.tif" \
    #         --batchsize=18 \
    #         --lr=$lr \
    #         --min_datasize=10000 \
    #         --epoch=$epochs \
    #         --mask_ratio=0.25 \
    #         --add_noise=1.0 \
    #         --resume_pretrain \
    #         --resume_s1_path="./ckpt/adapter/SIM_simulation/comb_3x/lr=0.0001--add_noise=1.0--accumulation_step=4--lp_tv=0.001--mask_ratio=${mask_ratio}--lrc=${lrc}--s${s_value}" \
    #         --save_dir="./ckpt/adapter/SIM_simulation/comb_3x" \
    #         --patch_size 3 16 16 \
    #         --rescale 3 3 \
    #         --crop_size 80 80 \
    #         --psf_size 49 49 \
    #         --lrc=$next_lrc \
    #         --accumulation_step=2 \
    #         --use_gt \
    #         --not_resume_s1_opt \
    #         --use_mp \
    #         --use_remat

    lrc=$next_lrc
    s_value=$((s_value + 1))

done

# CUDA_VISIBLE_DEVICES=${gpu} python train.py \
#         --trainset="/data_nas/nas/Research/Datasets/SIM_simulation/*/*/*/train/*.tif" \
#         --testset="/data_nas/nas/Research/Datasets/SIM_simulation/*/*/*/test/*.tif" \
#         --batchsize=9 \
#         --lr=$lr \
#         --min_datasize=10000 \
#         --epoch=200 \
#         --mask_ratio=0.25 \
#         --add_noise=1.0 \
#         --resume_pretrain \
#         --resume_s1_path="./ckpt/adapter/SIM_simulation/comb_3x/lr=0.0001--add_noise=1.0--accumulation_step=4--lp_tv=0.001--mask_ratio=0.25--lrc=${lrc}--s${s_value}" \
#         --save_dir="./ckpt/adapter/SIM_simulation/comb_3x" \
#         --patch_size 3 16 16 \
#         --rescale 3 3 \
#         --crop_size 80 80 \
#         --psf_size 49 49 \
#         --lrc=$next_lrc \
#         --accumulation_step=4 \
#         --use_gt \
#         --not_resume_s1_opt

# CUDA_VISIBLE_DEVICES=${gpu} python train.py \
#         --trainset="/data_nas/nas/Research/Datasets/SIM_simulation_std_noise/*/*/*/train/*.tif" \
#         --testset="/data_nas/nas/Research/Datasets/SIM_simulation_std_noise/*/*/*/test/*.tif" \
#         --batchsize=12 \
#         --lr=$lr \
#         --min_datasize=10000 \
#         --epoch=1000 \
#         --mask_ratio=0.75 \
#         --add_noise=1.0 \
#         --resume_pretrain \
#         --resume_s1_path="./ckpt/adapter/SIM_simulation/comb_3x/lr=0.0001--add_noise=1.0--accumulation_step=4--lp_tv=0.001--mask_ratio=0.25--lrc=${lrc}--s$((s_value+1))" \
#         --save_dir="./ckpt/adapter/SIM_simulation_std_noise/comb_3x" \
#         --patch_size 3 16 16 \
#         --rescale 3 3 \
#         --crop_size 80 80 \
#         --psf_size 49 49 \
#         --lrc=$next_lrc \
#         --accumulation_step=4 \
#         --use_gt \
#         --not_resume_s1_opt

# CUDA_VISIBLE_DEVICES=${gpu} python train.py \
#         --trainset="/data_nas/nas/Research/Datasets/SIM_simulation_std_noise/*/*/*/train/*.tif" \
#         --testset="/data_nas/nas/Research/Datasets/SIM_simulation_std_noise/*/*/*/test/*.tif" \
#         --batchsize=9 \
#         --lr=$lr \
#         --min_datasize=10000 \
#         --epoch=300 \
#         --mask_ratio=0.25 \
#         --add_noise=1.0 \
#         --resume_pretrain \
#         --resume_s1_path="./ckpt/adapter/SIM_simulation_std_noise/comb_3x/lr=0.0001--add_noise=1.0--accumulation_step=4--lp_tv=0.001--mask_ratio=0.75--lrc=32--s7" \
#         --save_dir="./ckpt/adapter/SIM_simulation_std_noise/comb_3x" \
#         --patch_size 3 16 16 \
#         --rescale 3 3 \
#         --crop_size 80 80 \
#         --psf_size 49 49 \
#         --lrc=$next_lrc \
#         --accumulation_step=4 \
#         --use_gt \
#         --not_resume_s1_opt

# CUDA_VISIBLE_DEVICES=${gpu} python train.py \
#         --trainset="/data_nas/nas/Research/Datasets/SIM_simulation_v4/*/*/*/train/*.tif" \
#         --testset="/data_nas/nas/Research/Datasets/SIM_simulation_v4/*/*/*/test/*.tif" \
#         --batchsize=16 \
#         --lr=$lr \
#         --min_datasize=10000 \
#         --epoch=500 \
#         --mask_ratio=0.25 \
#         --add_noise=1.0 \
#         --resume_pretrain \
#         --resume_s1_path="./ckpt/adapter/SIM_simulation_std_noise/comb_3x/lr=0.0001--add_noise=1.0--accumulation_step=4--lp_tv=0.001--mask_ratio=0.25--lrc=32--s8" \
#         --save_dir="./ckpt/adapter/SIM_simulation_v4/comb_3x" \
#         --patch_size 3 16 16 \
#         --rescale 3 3 \
#         --crop_size 80 80 \
#         --psf_size 49 49 \
#         --lrc=$next_lrc \
#         --accumulation_step=2 \
#         --not_resume_s1_opt \
#         --use_gt \
#         --use_mp \
#         --use_remat