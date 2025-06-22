import os
import argparse
import numpy as np
from skimage.draw import disk
from skimage.transform import resize
from simulation_np import (
    new_psf_2d, 
    cosine_light_pattern,
    get_Ks,
    convolve_fft,
    percentile_norm,
    save_tiff_imagej_compatible,
    FFT_sim_recon,
    circular_blur
)

def generate_beads(num_points, W, radius=3):
    """Generate random beads - standalone implementation"""
    num_points = max(1, int(round(num_points)))
    W = int(W)
    radius = max(1, int(radius))
    
    img = np.zeros((W, W), dtype=np.float32)
    
    # Random positions
    x_positions = np.random.randint(radius, W-radius, size=num_points)
    y_positions = np.random.randint(radius, W-radius, size=num_points)
    
    # Draw circles
    for x, y in zip(x_positions, y_positions):
        rr, cc = disk((y, x), radius, shape=img.shape)
        img[rr, cc] = 1.0
    
    # Blur if small radius
    if radius <= 1:
        img = circular_blur(img, 2)
    
    return img

def generate_beads_sim(params):
    """Generate beads SIM dataset"""
    
    # Create directory structure
    train_path = os.path.join(params['output_dir'], 'train')
    train_gt_path = os.path.join(params['output_dir'], 'train_gt')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(train_gt_path, exist_ok=True)
    
    # Generate beads pattern
    beads = generate_beads(params['num_beads'], params['W'] * params['scale'], params['radius'])
    beads = (beads - beads.min()) / (beads.max() - beads.min()) if beads.max() > beads.min() else beads
    
    # Generate PSF
    psf = np.array(new_psf_2d(params['lambda'], 49 * params['scale'] // 3, 62.6 / params['scale']))[np.newaxis, ...]
    
    # Generate light patterns
    Ks = get_Ks(params['theta_start'], 3, params['period'] * params['scale'], beads.shape[-1])
    patterns = cosine_light_pattern(beads.shape, Ks, phases=params['phi'], M=params['magnitude'])
    
    # Apply patterns and convolve
    modified = beads[np.newaxis, ...] * patterns
    raw = convolve_fft(modified, psf)
    raw = percentile_norm(raw)
    
    # Resize to target size
    raw = resize(raw, (raw.shape[0], raw.shape[1], params['W'], params['W']))
    beads = resize(beads, (params['W'], params['W']))
    patterns = resize(patterns, (patterns.shape[0], patterns.shape[1], params['W'], params['W']))
    
    # Add noise
    raw_std = raw / raw.std()
    no_empty = raw_std > 1e-2
    raw = raw / raw[no_empty].mean() * params['ave_photon']
    raw = np.random.poisson(raw) + np.random.normal(0, params['noise'] * raw.std(), raw.shape)
    
    # Save outputs following simulation_np.py format
    save_tiff_imagej_compatible(f"{train_path}/0.tif", raw.astype(np.float32), "CZYX")
    save_tiff_imagej_compatible(f"{train_gt_path}/0.tif", beads.astype(np.float32), "YX")
    save_tiff_imagej_compatible(f"{train_gt_path}/0_lp.tif", patterns.astype(np.float32), "CZYX")
    
    # Save config
    import json
    config = {
        'num_beads': params['num_beads'],
        'radius': params['radius'],
        'image_size': params['W'],
        'period': params['period'],
        'noise': params['noise'],
        'ave_photon': params['ave_photon'],
        'M': params['magnitude'],
        'wavelength': params['lambda'],
        'theta_start': params['theta_start'],
        'phi': params['phi'].tolist()
    }
    with open(f"{train_gt_path}/0_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Generated beads dataset:")
    print(f"  - Number of beads: {params['num_beads']}")
    print(f"  - Bead radius: {params['radius']} pixels")
    print(f"  - Image size: {params['W']}x{params['W']}")
    print(f"  - Output: {params['output_dir']}/train/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate beads SIM dataset')
    parser.add_argument('--num_beads', type=int, default=100, help='Number of beads')
    parser.add_argument('--radius', type=int, default=3, help='Bead radius in pixels')
    parser.add_argument('--size', type=int, default=256, help='Image size')
    parser.add_argument('--output_dir', type=str, default='./data/SIM-simulation/beads', help='Output directory')
    parser.add_argument('--period', type=float, default=5, help='SIM pattern period')
    parser.add_argument('--noise', type=float, default=0.1, help='Noise level')
    parser.add_argument('--photons', type=int, default=1000, help='Average photon count')
    parser.add_argument('--magnitude', type=float, default=0.8, help='Pattern modulation depth')
    parser.add_argument('--wavelength', type=int, default=488, help='Wavelength in nm')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Set parameters
    params = {
        'num_beads': args.num_beads,
        'radius': args.radius,
        'W': args.size,
        'scale': 6,
        'output_dir': args.output_dir,
        'period': args.period,
        'noise': args.noise,
        'ave_photon': args.photons,
        'magnitude': args.magnitude,
        'lambda': args.wavelength,
        'theta_start': np.random.rand() * 360,
        'phi': np.random.rand(3) * 2 * np.pi
    }
    
    generate_beads_sim(params)