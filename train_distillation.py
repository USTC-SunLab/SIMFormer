# SIMFormer distillation training script
# This script trains a simplified model using SIMFormer outputs as pseudo ground-truth

import os
import glob
import argparse
from shutil import copyfile
import numpy as np
import torch.utils.tensorboard as tb
from model_distillation import pipeline

# Arguments
parser = argparse.ArgumentParser(description='SIMFormer self-distillation training')

# Dataset
parser.add_argument('--crop_size', nargs='+', type=int, default=[112, 112], help="y, x")
parser.add_argument('--trainset', type=str, default="../data/3D")
parser.add_argument('--testset', type=str, default="../data/3D")
parser.add_argument('--min_datasize', type=int, default=18000)

# SIMFormer Infer Integration - Key for distillation
parser.add_argument('--simformer_infer_save_dir', type=str, required=True,
                    help='SIMFormer inference output directory containing test/ subfolder with emitter predictions')

# Training settings
parser.add_argument('--batchsize', type=int, default=18)
parser.add_argument('--epoch', type=int, default=101)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--add_gaussian_noise', type=float, default=0.1)
parser.add_argument('--shot_noise_scale', type=float, default=None)

# Resume
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume_pretrain', action='store_true')
parser.add_argument('--resume_s1_path', type=str, default=None)
parser.add_argument('--resume_s1_iter', type=str, default=None)
parser.add_argument('--not_resume_s1_opt', action='store_true')

# Distillation-specific losses
parser.add_argument('--emitter_lasso', type=float, default=0,
                    help='L1 regularization on emitter output')

# MAE
parser.add_argument('--mask_ratio', type=float, default=0.0,
                    help='Masking ratio for MAE (typically 0 for distillation)')
parser.add_argument('--patch_size', nargs='+', type=int, default=[3, 16, 16], help="z, y, x")

# Output rescaling (matching SIMFormer's super-resolution factor)
parser.add_argument('--rescale', nargs='+', type=int, default=[3, 3], help="y, x")

# Low-rank coding dimension (simplified for distillation)
parser.add_argument('--lrc', type=int, default=32)

# Logging
parser.add_argument('--save_dir', type=str, default="./ckpt/distillation")
parser.add_argument('--tag', type=str, default=None)

def train(args):
    args = parser.parse_args()
    
    # Build configuration string for save directory
    to_log = ["mask_ratio", "add_gaussian_noise", "lr", "lrc"]
    if args.shot_noise_scale is not None:
        to_log.append("shot_noise_scale")
    if args.emitter_lasso > 0:
        to_log.append("emitter_lasso")
    
    cfg_str = '--'.join(["{}={}".format(k, v) for k, v in vars(args).items() if k in to_log])
    if args.tag is not None:
        cfg_str += '--{}'.format(args.tag)
    
    args.save_dir = os.path.join(args.save_dir, cfg_str)
    print(f"\033[93mSave directory: {args.save_dir}\033[0m")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
    
    # Setup tensorboard
    tensorboard_dir = os.path.join(args.save_dir, 'runs')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = tb.SummaryWriter(tensorboard_dir)

    # Copy source files for reproducibility
    file_names = glob.glob("./*.py") + glob.glob("./script/*.sh")
    save_file_tmp_path = os.path.join(args.save_dir, 'src')
    os.makedirs(save_file_tmp_path, exist_ok=True)
    for file_name in file_names:
        if os.path.isfile(file_name):
            copyfile(file_name, os.path.join(save_file_tmp_path, os.path.basename(file_name)))

    # Run training pipeline
    _ = pipeline(args, writer)
    writer.close()

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)