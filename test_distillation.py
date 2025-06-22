# SIMFormer distillation inference script
# Processes data using distilled models (emitter-only output)

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'False'
import argparse
import numpy as np
from model_distillation import pipeline_infer_distillation
import re

parser = argparse.ArgumentParser(description='SIMFormer distillation inference')

# Dataset
parser.add_argument('--crop_size', nargs='+', type=int, default=[80, 80], help="y, x")
parser.add_argument('--data_dir', type=str, required=True, 
                    help="Path to input data (can use wildcards like '*.tif')")

# Inference settings
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--resume_path', type=str, required=True,
                    help="Path to distillation checkpoint directory")
parser.add_argument('--resume_iter', type=int, default=None,
                    help="Specific iteration to load (default: latest)")

# Model configuration
parser.add_argument('--patch_size', nargs='+', type=int, default=[3, 16, 16], help="z, y, x")
parser.add_argument('--rescale', nargs='+', type=int, default=[3, 3], help="y, x - super-resolution factor")
parser.add_argument('--lrc', type=int, default=None, help="Low-rank coding dimension (auto-detected from path)")

# Output
parser.add_argument('--save_dir', type=str, default="./results/distillation",
                    help="Directory to save inference results")
parser.add_argument('--save_format', type=str, default='tif', choices=['tif', 'png'],
                    help="Output image format")

# Processing options
parser.add_argument('--overlap', type=int, default=0,
                    help="Overlap for tiling large images (0 = no tiling)")
parser.add_argument('--sequential_mode', action='store_true',
                    help="Process sequential frames rather than batch processing")


def infer(args):
    args = parser.parse_args()
    
    # Auto-detect LRC from checkpoint path if not specified
    if args.lrc is None:
        match = re.search(r"lrc=(\d+)", args.resume_path)
        if match:
            args.lrc = int(match.group(1))
            print(f"Auto-detected LRC: {args.lrc}")
        else:
            args.lrc = 32  # Default value for distillation
            print(f"Using default LRC: {args.lrc}")
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Output directory: {args.save_dir}")
    
    # Save configuration
    with open(os.path.join(args.save_dir, 'inference_config.txt'), 'w') as f:
        f.write(str(args))
    
    # Run inference pipeline
    _ = pipeline_infer_distillation(args)
    
    print(f"Inference completed! Results saved to: {args.save_dir}")


if __name__ == "__main__":
    args = parser.parse_args()
    infer(args)