import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'False'
import argparse
import numpy as np
from model import pipeline_infer
import re

parser = argparse.ArgumentParser(description='Super-resolved microscopy via physics-informed masked autoencoder')
##################################### Dataset #################################################
parser.add_argument('--crop_size', nargs='+', type=int, default=[80, 80])
parser.add_argument('--data_dir', type=str, default="../data/")
parser.add_argument('--gt', action='store_true')
parser.add_argument('--adapt_pattern_dimension', action='store_true', 
                    help="Adapt pattern dimension for model compatibility when data has different pattern dimension than training (9 frames)")
parser.add_argument('--target_pattern_frames', type=int, default=9, 
                    help="Target pattern dimension size (default: 9 for standard SIM with 3 angles × 3 phases)")
parser.add_argument('--random_pattern_sampling', action='store_true', 
                    help="Use random sampling instead of uniform for pattern dimension adaptation")
##################################### Inferring setting #################################################
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--resume_path', type=str, default=None)
parser.add_argument('--resume_s1_path', type=str, default=None)
parser.add_argument('--resume_s1_iter', type=int, default=None)
parser.add_argument('--add_noise', type=float, default=0)
##################################### MAE ############################################
parser.add_argument('--patch_size', nargs='+', type=int, default=[3, 16, 16])
parser.add_argument('--mask_ratio', type=float, default=0.0)
##################################### Physics ############################################
parser.add_argument('--num_p', type=int, default=9)
parser.add_argument('--psf_size', nargs='+', type=int, default=[49, 49])
parser.add_argument('--rescale', nargs='+', type=int, default=[3, 3])
parser.add_argument('--lrc', type=int, default=None)
##################################### Log ############################################
parser.add_argument('--save_dir', type=str, default="./test")


def infer(args):
    args = parser.parse_args()
    if args.lrc is None:
        match = re.search(r"lrc=(\d+)", args.resume_path)
        if match:
            args.lrc = int(match.group(1))
    
    os.makedirs(args.save_dir, exist_ok=True)
    print(args.save_dir)
    _ = pipeline_infer(args)


    

if __name__ == "__main__":
    args = parser.parse_args()
    infer(args)
    