##################
## BioSR
##################


# path="./ckpt/adapter/2D_BioSR/comb_3x/lr=0.0001--add_noise=1.0--accumulation_step=4--lp_tv=0.001--mask_ratio=0.75--lrc=32--s8"

# chown -R root:root ${path}

# CUDA_VISIBLE_DEVICES="0" python infer.py \
#     --data_dir="<path to BioSR dataset>/*/2D/train/*.tif" \
#     --resume_path=${path} \
#     --save_dir=./result/BioSR_v2/train

# CUDA_VISIBLE_DEVICES="0" python infer.py \
#     --data_dir="<path to BioSR dataset>/*/2D/test/*.tif" \
#     --resume_path=${path} \
#     --save_dir=./result/BioSR_v2/test

# CUDA_VISIBLE_DEVICES="0" python infer.py \
#     --data_dir="<path to BioSR dataset>/*/2D/gt/*.tif" \
#     --resume_path=${path} \
#     --save_dir=./result/BioSR_v2/gt




##################
## BioSR noise
##################

# path="lr=0.0001--add_noise=1.0--accumulation_step=4--lp_tv=0.001--mask_ratio=0.75--lrc=32--s8"

# CUDA_VISIBLE_DEVICES="1" python infer.py \
#     --data_dir="<path to BioSR dataset>_noisy_test_set3/*/*/*.tif" \
#     --resume_path="./ckpt/adapter/2D_BioSR/comb_3x/${path}" \
#     --save_dir=./result/BioSR_noisy_test_set3/


##################
## simulation
##################


# path="./ckpt/adapter/SIM_simulation_v4/comb_3x/lr=0.0001--add_noise=1.0--accumulation_step=4--lp_tv=0.001--mask_ratio=0.25--lrc=32--s9"

# CUDA_VISIBLE_DEVICES="0" python infer.py \
#     --data_dir="<path to dataset>/SIM_simulation_v4/*/*/*/test/*.tif" \
#     --resume_path=${path} \
#     --save_dir=./result/SIM_simulation_v4/test/


# CUDA_VISIBLE_DEVICES="0" python infer.py \
#     --data_dir="<path to dataset>/SIM_simulation_v4/*/*/*/train/*.tif" \
#     --resume_path=${path} \
#     --save_dir=./result/SIM_simulation_v4/train/


# CUDA_VISIBLE_DEVICES="1" python infer.py \
#     --data_dir="<path to dataset>/SIM_simulation_v4/ccps/ave_photon/1/test/*.tif" \
#     --resume_path=${path} \
#     --save_dir=./result/SIM_simulation_v4/test/ccps/ave_photon/1


# CUDA_VISIBLE_DEVICES="1" python infer.py \
#     --data_dir="<path to dataset>/SIM_simulation_v4/ccps/ave_photon/1/train/*.tif" \
#     --resume_path=${path} \
#     --save_dir=./result/SIM_simulation_v4/train/ccps/ave_photon/1

#################
# BioSR procedure
#################

# task_names=(
#   "CCPs"
#   "Microtubules"
# )
# path_list=(
#     "lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.75--lrc=512--s1"
#     "lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.25--lrc=256--s2"
#     "lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.25--lrc=128--s3"
#     "lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.25--lrc=64--s4"
#     "lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.25--lrc=32--s5"
#     "lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.25--lrc=32--s6"
#     "lr=0.0001--add_noise=1.0--accumulation_step=4--lp_tv=0.001--mask_ratio=0.25--lrc=32--s7"
#     "lr=0.0001--add_noise=1.0--accumulation_step=4--lp_tv=0.001--mask_ratio=0.75--lrc=32--s8"
# )
# for task_name in "${task_names[@]}"
# do
#     for path in "${path_list[@]}"
#     do
#         CUDA_VISIBLE_DEVICES="0" python infer.py \
#             --data_dir="<path to BioSR dataset>/${task_name}/2D/gt/Cell_002.tif" \
#             --resume_path="./ckpt/adapter/2D_BioSR/comb_3x/${path}" \
#             --save_dir=./result/procedure/BioSR/${task_name}/${path}
#     done
# done

######################
# finetune
######################
# path="./ckpt/adapter/finetune/beads/lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.75--lrc=32--s9"

# CUDA_VISIBLE_DEVICES="7" python infer.py \
#     --data_dir="<path to dataset>/SIM-simulation-onestack/beads/density/standard/train/*.tif" \
#     --resume_path=${path} \
#     --save_dir=./result/SIM-simulation-onestack/beads


# path="./ckpt/adapter/finetune/microtubules/lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.75--lrc=32--s9"

# CUDA_VISIBLE_DEVICES="2" python infer.py \
#     --data_dir="<path to dataset>/BioTISR/BioTISR_Microtubules/Cell_001/*.tif" \
#     --resume_path=${path} \
#     --save_dir="./result/BioTISR_Microtubules/Cell_001"

# path="./ckpt/adapter/finetune/mitochondria/lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.75--lrc=32--s9"

# CUDA_VISIBLE_DEVICES="2" python infer.py \
#     --data_dir="<path to dataset>/BioTISR/BioTISR_Mitochondria/*/*.tif" \
#     --resume_path=${path} \
#     --save_dir="./result/BioTISR_Mitochondria_lasso"

# path="./ckpt/adapter/finetune/beads/lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.75--lrc=32--s9"

# CUDA_VISIBLE_DEVICES="4" python infer.py \
#     --data_dir="<path to dataset>/SIM-simulation/beads/density/standard/train/*.tif" \
#     --resume_path=${path} \
#     --save_dir="./result/simulation_beads"

# path="./ckpt/adapter/finetune/100nm_beads_l2/lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.75--lrc=32--s9"

# CUDA_VISIBLE_DEVICES="4" python infer.py \
#     --data_dir="<path to dataset>/100nm_beads/*.tif" \
#     --resume_path=${path} \
#     --save_dir="./result/100nm_beads_l2"

path="./ckpt/adapter/finetune/Open-3DSIM/lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.75--lrc=32--s10"

CUDA_VISIBLE_DEVICES="7" python infer.py \
    --data_dir="<path to dataset>/Open-3DSIM/*.tif"\
    --resume_path=${path} \
    --adapt_z_dimension \
    --target_z_frames=9 \
    --save_dir="./result/Open-3DSIM"


path="./ckpt/adapter/finetune/Mito_Zeiss/lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.75--lrc=32--s10"

CUDA_VISIBLE_DEVICES="7" python infer.py \
    --data_dir="<path to dataset>/SIMtoolbox/*.tif" \
    --resume_path=${path} \
    --adapt_z_dimension \
    --target_z_frames=9 \
    --save_dir="./result/SIMtoolbox"