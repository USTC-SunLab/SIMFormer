# SIMFormer training script

import os
import glob
import argparse
from shutil import copyfile
import re
import numpy as np
import torch.utils.tensorboard as tb
from model import pipeline

# Arguments
parser = argparse.ArgumentParser(description='Super-resolved microscopy via physics-informed masked autoencoder')

# Dataset
parser.add_argument('--crop_size', nargs='+', type=int, default=[112, 112])
parser.add_argument('--trainset', type=str, default="../data/3D")
parser.add_argument('--testset', type=str, default="../data/3D")
parser.add_argument('--min_datasize', type=int, default=18000)
parser.add_argument('--sampling_rate', type=float, default=1.0)
parser.add_argument('--adapt_pattern_dimension', action='store_true',
                    help='Adapt pattern dimension for model compatibility when data has different pattern dimension than training (9 frames)')
parser.add_argument('--target_pattern_frames', type=int, default=9,
                    help='Target pattern dimension size (default: 9 for standard SIM with 3 angles × 3 phases)')
parser.add_argument('--random_pattern_sampling', action='store_true',
                    help='Use random sampling instead of uniform for pattern dimension adaptation')

# Training
parser.add_argument('--batchsize', type=int, default=18)
parser.add_argument('--epoch', type=int, default=101)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--add_noise', type=float, default=0.1)
parser.add_argument('--use_gt', action='store_true')
parser.add_argument('--accumulation_step', type=int, default=None)

# Resume
parser.add_argument('--resume_pretrain', action='store_true')
parser.add_argument('--resume_s1_path', type=str, default=None)
parser.add_argument('--resume_s1_iter', type=str, default=None)
parser.add_argument('--not_resume_s1_opt', action='store_true')
parser.add_argument('--resume', action='store_true')

# Loss weights
parser.add_argument('--tv_loss', type=float, default=1e-3)
parser.add_argument('--lp_tv', type=float, default=1e-3)
parser.add_argument('--psfc_loss', type=float, default=1e-1)

# MAE
parser.add_argument('--mask_ratio', type=float, default=0.75)
parser.add_argument('--patch_size', nargs='+', type=int, default=[3, 16, 16])

# Physics
parser.add_argument('--psf_size', nargs='+', type=int, default=[49, 49])
parser.add_argument('--rescale', nargs='+', type=int, default=[2, 2])
parser.add_argument('--lrc', type=int, default=1)

# Logging
parser.add_argument('--save_dir', type=str, default="./ckpt")
parser.add_argument('--tag', type=str, default=None)

def build_experiment_name(args):
    # Build experiment name from parameters
    parameters_to_log = ["mask_ratio", "add_noise", "lr", "lrc"]
    
    if args.lp_tv > 0:
        parameters_to_log.append("lp_tv")
    if args.accumulation_step is not None:
        parameters_to_log.append("accumulation_step")
    
    config_string = '--'.join([f"{k}={getattr(args, k)}" for k in parameters_to_log])
    
    # Stage number
    if args.resume_s1_path is not None:
        stage_match = re.search(r'--s(\d+)', args.resume_s1_path)
        if stage_match:
            previous_stage = int(stage_match.group(1))
            config_string += f'--s{previous_stage + 1}'
        else:
            # If no stage number found, assume it's stage 0 (initial checkpoint)
            config_string += '--s1'
    else:
        config_string += '--s1'
    
    if args.tag is not None:
        config_string += f'--{args.tag}'
    
    return config_string

def setup_directories(args):
    # Setup experiment directories
    experiment_name = build_experiment_name(args)
    args.save_dir = os.path.join(args.save_dir, experiment_name)
    
    print(f"\033[93mExperiment directory: {args.save_dir}\033[0m")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
    
    tensorboard_dir = os.path.join(args.save_dir, 'runs')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = tb.SummaryWriter(tensorboard_dir)
    
    return writer

def backup_source_code(save_dir):
    # Backup source code
    source_files = glob.glob("./*.py") + glob.glob("./*.sh")
    
    source_code_dir = os.path.join(save_dir, 'src')
    os.makedirs(source_code_dir, exist_ok=True)
    
    for file_path in source_files:
        file_name = os.path.basename(file_path)
        copyfile(file_path, os.path.join(source_code_dir, file_name))
    
    print(f"Source code backed up to: {source_code_dir}")

def train(args):
    # Main training function
    args = parser.parse_args()
    
    writer = setup_directories(args)
    backup_source_code(args.save_dir)
    
    print("\nStarting training pipeline...")
    pipeline(args, writer)
    
    writer.close()
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)