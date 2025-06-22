# SIMFormer: Transformer-based Self-supervised Structured Illumination Microscopy Reconstruction

## Abstract

SIMFormer is a Transformer-based self-supervised framework for super-resolution structured illumination microscopy (SIM) reconstruction. By leveraging masked autoencoding and physics-informed learning, SIMFormer achieves ~45 nm resolution (2.5× better than conventional SIM) without requiring ground truth high-resolution data. The framework performs blind reconstruction, simultaneously estimating emitters, illumination patterns, background, and point spread function (PSF) from raw SIM data alone.

## Key Features

- **Super-resolution beyond conventional limits**: Achieves ~45 nm resolution, approaching STORM-level detail
- **Self-supervised learning**: No ground truth high-resolution images required
- **Blind reconstruction**: Simultaneously estimates all imaging parameters
- **Noise robustness**: SIMFormer+ variant maintains quality at low photon counts
- **Adaptable**: One-stack fine-tuning for new microscopes or sample types
- **Fast inference**: Real-time reconstruction capability

## Installation

### Prerequisites

- Python 3.12
- CUDA 12.1 (for GPU support)
- JAX-compatible GPU

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SIMFormer.git
cd SIMFormer
```

2. Create a conda environment:
```bash
conda create -n simformer python=3.12
conda activate simformer
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For GPU support, install JAX with CUDA:
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Data Preparation

### BioSR Dataset

BioSR is a benchmark dataset for super-resolution microscopy reconstruction. To prepare BioSR data for training:

```bash
python BioSR_preprocess.py
```

This script processes raw BioSR microscopy data (MRC format) into TIFF files and creates the following structure:
```
/root/data/BioSR/
├── CCPs/
│   ├── 2D/
│   │   ├── level_01/     # Low noise
│   │   ├── level_02/
│   │   ├── ...
│   │   ├── level_09/     # High noise
│   │   ├── train/        # Highest noise level (Cell_001-035)
│   │   └── test/         # Highest noise level (Cell_036-054)
│   └── WF/               # Widefield (averaged frames)
├── Microtubules/         # Same structure as CCPs
├── F-actin/              # level_01 to level_12
└── ER/                   # level_01 to level_06
```

Key features:
- Converts MRC files to TIFF format
- Processes multiple noise levels (level_01 = low noise, highest level = high noise)
- Creates train/test splits for the highest noise level
- Generates widefield (WF) images by averaging SIM frames

### Simulated Data

For comprehensive testing and validation, use the main simulation tool:

```bash
python simulation_np.py --output_prefix ../data/SIM-simulation
```

This generates synthetic SIM datasets with train/test splits for three structure types:
- **Curves**: Bezier curve structures (filaments)
- **Tubes**: Tubular structures (ER-like)
- **Rings**: Ring-shaped structures (vesicle-like)

Each structure type is generated with variations in:
- Light pattern period (3, 4, 5, 7, 10 pixels)
- Average photon count (0.1 to 1000)
- Structure density/size parameters

Output structure:
```
../data/SIM-simulation/
├── curve/
│   ├── light_pattern_period/
│   ├── ave_photon/
│   └── sparsity/
├── tube/
└── ring/
```

For simple bead-only simulations without train/test split:
```bash
python simulation_beads.py
```

## Usage

### Getting Started

We recommend starting with one-stack fine-tuning experiments using simulated bead data:

1. **Generate simulated bead data**:
   ```bash
   python simulation_beads.py
   ```

2. **Fine-tune pre-trained model on one stack**:
   ```bash
   bash script/finetune.sh
   ```
   
This approach is recommended because:
- Requires minimal data (just one SIM stack)
- Provides quick validation of the framework
- Bead structures offer clear metrics for resolution assessment
- Allows testing on controlled, noise-free ground truth

After validating on simulated data, you can proceed to fine-tune on your experimental data or train from scratch on larger datasets.

### Training

#### Multi-stage Training

SIMFormer employs an 8-stage progressive training strategy. To train the model, use the provided script:

```bash
bash script/train_BioSR_sunlab.sh
```

This script automatically handles all 8 stages of training with the appropriate curriculum learning schedule, progressively refining the model from initial feature learning to final robustness training.

### Inference

To process SIM image stacks and generate super-resolved reconstructions, use the provided script:

```bash
bash script/test.sh
```

The script includes examples for processing various datasets including BioSR, simulated data, and custom microscopy data with both standard and sequential frame processing modes.

### Fine-tuning

SIMFormer can be adapted to new microscopes or sample types with minimal data through one-stack fine-tuning. Use the provided script:

```bash
bash script/finetune.sh
```

The script includes configurations for fine-tuning on various datasets including beads, specific organelles, and different microscope systems. It demonstrates both single-stage and multi-stage fine-tuning approaches.

### Z-dimension Adaptation

SIMFormer is trained on standard 9-frame SIM data (3 angles × 3 phases). When working with datasets that have different Z-dimensions, use the `--adapt_z_dimension` flag:

```bash
python train.py \
    --trainset /path/to/data \
    --adapt_z_dimension \
    --target_z_frames 9 \
    ...
```

This feature:
- **Adapts** any input Z-dimension to match the model's expected 9 frames
- **Preserves** model architecture compatibility without modification
- **Samples** frames uniformly (default) or randomly from the Z-stack

Common use cases:
- **Open-3DSIM dataset**: May have varying Z-dimensions
- **SIMtoolbox data**: Often contains different numbers of frames  
- **Custom microscopy data**: Adapt any Z-dimension to model requirements

## Model Architecture

### Core Components

1. **Masked Autoencoder (MAE)**
   - Vision Transformer (ViT-Base) with 3D patch embedding
   - 12 transformer layers with adapter modules
   - Random masking ratio: 25-75%

2. **Physics-informed Decoders**
   - **Emitter Decoder**: CNN upsampling with Softplus activation
   - **Illumination Pattern Decoder**: Low-rank coding (LRC) with Softmax
   - **Background Decoder**: Coarse pooling for spatial smoothness

3. **PSF Estimation**
   - Deep Image Prior approach with random noise input
   - Convolutional network outputting PSF kernel

### Training Strategy

The model employs curriculum learning with progressive stages:

| Stage | Mask Ratio | LRC Rank | Key Focus |
|-------|------------|----------|-----------| 
| 1-5   | 0.75→0.25  | 512→32   | Feature learning |
| 6-7   | 0.25       | 32       | Refinement |
| 8     | 0.75       | 32       | Robustness |

### Loss Functions

Total loss combines multiple components:
- **Reconstruction Loss**: 0.875×L1 + 0.125×MS-SSIM
- **Total Variation**: Smoothness on patterns and PSF
- **Hessian Loss**: Continuity prior for emitters
- **PSF Centering**: Prevents PSF drift

## Repository Structure

```
SIMFormer/
├── train.py              # Main training script
├── infer.py              # Inference script
├── model.py              # Training pipeline and loss functions
├── network.py            # Neural network architectures
├── simulation_np.py      # SIM simulation utilities
├── simulation_beads.py   # Bead simulation for testing
├── jax_mae/              # Masked autoencoder implementation
│   ├── mae.py
│   ├── vision_transformer.py
│   └── ...
├── script/               # Training and evaluation scripts
│   ├── train_BioSR_sunlab.sh
│   ├── finetune.sh
│   └── test.sh
├── utils_*.py            # Utility functions
├── requirements.txt      # Package dependencies
└── README.md            # This file
```