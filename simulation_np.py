from utils_imageJ import save_tiff_imagej_compatible
from skimage.transform import resize
from skimage.draw import disk
import numpy as np
import tqdm
import cv2
from scipy.special import comb
from scipy.signal import fftconvolve
import math
import multiprocessing
from scipy import ndimage
import argparse
import os
import json
from scipy.special import jv

def new_psf_2d(Lambda=488, size=49, step=None):
    # Generate 2D PSF
    _Nx = size
    _Ny = size
    _scale = 1.
    _step = step or 62.6 / _scale
    _x = np.linspace(-_Nx/2*_step, _Nx/2*_step, _Nx) - 0.0001
    _y = np.linspace(-_Ny/2*_step, _Ny/2*_step, _Ny)
    
    xx, yy = np.meshgrid(_x, _y)
    r = np.sqrt(xx**2 + yy**2)
    NA = 1.3
    mask = r > 1e-10
    psf = np.zeros_like(r)
    psf[mask] = (jv(1, 2 * np.pi * NA / Lambda * r[mask])**2) / (np.pi * r[mask]**2)
    central_value = (NA**2) / (Lambda**2)
    psf[~mask] = central_value
    psf = psf / psf.sum()
    
    return psf

def convolve_curve_with_psf(curve, psf):
    convolved = convolve_fft(curve, psf)
    return convolved


def gaussian_kernel_3d(size, sigma):
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    z = np.linspace(-size // 2, size // 2, size)
    xx, yy, zz = np.meshgrid(x, y, z)
    kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2 + zz ** 2) / sigma ** 2)
    return kernel / np.sum(kernel)


def catmull_rom_spline(t, control_points, alpha=0.5):
    n = len(control_points)
    t_scaled = t * (n - 3)
    t_floor = np.floor(t_scaled).astype(int)
    t_fraction = t_scaled - t_floor
    p0_indices = np.clip(t_floor - 1, 0, n - 1)
    p1_indices = np.clip(t_floor, 0, n - 1)
    p2_indices = np.clip(t_floor + 1, 0, n - 1)
    p3_indices = np.clip(t_floor + 2, 0, n - 1)
    p0 = control_points[p0_indices]
    p1 = control_points[p1_indices]
    p2 = control_points[p2_indices]
    p3 = control_points[p3_indices]
    t2 = t_fraction * t_fraction
    t3 = t2 * t_fraction

    a = 2 * t2 - t3 - t_fraction
    b = 3 * t3 - 5 * t2 + 2
    c = -3 * t3 + 4 * t2 + t_fraction
    d = t3 - t2

    return 0.5 * (a[:, np.newaxis] * p0 + b[:, np.newaxis] * p1 + c[:, np.newaxis] * p2 + d[:, np.newaxis] * p3)


def percentile_norm(img, p_low=0.1, p_high=99.9):
    p_low = np.percentile(img, p_low)
    p_high = np.percentile(img, p_high)
    img = (img - p_low) / (p_high - p_low)
    img = np.clip(img, 0, 1)
    return img

def std_norm(img):
    img = img - img.mean()
    img = img / img.std()
    return img


def generate_bezier_curves(n, L, W):
    result = np.zeros((L, W, W), dtype=np.uint8)
    for _ in range(n):
        num_points = np.random.randint(100 * L, 200 * L)
        t = np.linspace(0, 1, num_points)
        control_points = np.random.rand(6, 3)
        control_points = control_points * [[L, W, W]]
        curve = catmull_rom_spline(t, control_points).astype(int)
        curve = curve[np.all(curve < W, axis=-1)]
        curve = curve[np.all(curve >= 0, axis=-1)]
        curve = curve[curve[:, 0] < L]
        num_points = curve.shape[0]
        steps = np.random.choice([-2, -1, 0, 0, 0, 0, 1, 2], size=num_points)
        walk = np.cumsum(steps) + 100
        result[curve[:, 0], curve[:, 1], curve[:, 2]] = walk

    result = convolve_fft(result, gaussian_kernel_3d(5, 1))
    return result


def cosine_light_pattern(shape, Ks, phases, M):
    x = np.linspace(0, shape[-1] - 1, shape[-1])
    y = np.linspace(0, shape[-2] - 1, shape[-2])
    xx, yy = np.meshgrid(x, y, indexing='ij')

    fields = []
    for i, K in enumerate(Ks):
        phase = phases[i]
        for j in range(3):
            kx, ky = K[0], K[1]
            field = 1 + M * \
                np.cos(
                    2 * np.pi / shape[-1] * (kx * xx + ky * yy) + phase + j * 2 * np.pi / 3)
            fields.append(field[np.newaxis, ...])
    return np.stack(fields, axis=0)



def random_angles(initial_angle):
    return [initial_angle, initial_angle + 120, initial_angle + 240]




def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def bezier_curve(control_points, num_points=100):
    n = len(control_points)
    t = np.linspace(0, 1, num_points)
    curve_points = np.zeros((num_points, 2))
    for i in range(n):
        binomial_coeff = math.factorial(n - 1) / (math.factorial(i) * math.factorial(n - 1 - i))
        point = np.outer(binomial_coeff * (t ** i) * ((1 - t) ** (n - 1 - i)), control_points[i])
        curve_points += point
    return curve_points


def generate_bezier_curves_2d(m, n):
    # Generate 2D curves
    upsampling_factor = 4
    n = upsampling_factor * n
    img = np.zeros((n, n))
    for _ in range(m):
        num_points = np.random.randint(3, 6)
        control_points = np.random.rand(num_points, 2) * n
        curve_points = bezier_curve(control_points, num_points=n)
        
        xs, ys = curve_points[:, 0].astype(int), curve_points[:, 1].astype(int)
        for x, y in zip(np.diff(xs), np.diff(ys)):
            img[ys, xs] = np.random.randint(160, 255)
    
    img = cv2.GaussianBlur(img.astype(np.float32), (9, 9), 1)
    img = resize(img, (n//upsampling_factor, n//upsampling_factor), anti_aliasing=True)
    img /= 255
    return img








def circular_blur(image, ksize):
    # Circular blur
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    for i in range(ksize):
        for j in range(ksize):
            if (i - center) ** 2 + (j - center) ** 2 <= center ** 2:
                kernel[i, j] = 1
    kernel /= np.sum(kernel)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def generate_bezier_curves_tube(tube_width, n):
    # Generate tube curves
    upsampling_factor = 4
    n = upsampling_factor * n
    img = np.zeros((n, n))
    m = np.random.randint(15,25)
    for _ in range(m):
        num_points = np.random.randint(3, 6)
        control_points = np.random.rand(num_points, 2) * n
        curve_points = bezier_curve(control_points, num_points=n)
        
        xs, ys = curve_points[:, 0].astype(int), curve_points[:, 1].astype(int)
        for x,y in zip(xs,ys):
            rr, cc = disk((x, y), tube_width*upsampling_factor, shape=img.shape)
            img[rr, cc] = 255
        
    img /= 255
    sobel_x = ndimage.sobel(img, axis=0)
    sobel_y = ndimage.sobel(img, axis=1)
    edges = np.hypot(sobel_x, sobel_y)
    edges = resize(edges, (n//upsampling_factor, n//upsampling_factor), anti_aliasing=True)
    edges = (edges / edges.max() * 255).astype(np.uint8)

    img = circular_blur(edges.astype(np.float32), 2)
    return img


def generate_ring(m, W, radius=(3, 5)):
    # Generate ring
    img = np.zeros((W, W), dtype=np.uint8)
    circle_num = m
    for _ in range(circle_num):
        x1 = np.random.randint(radius[0], radius[1])
        x2 = np.random.randint(max(radius[0], x1-1), min(radius[1], x1+1))
        axes = (x1, x2)
        center = (np.random.randint(
            axes[0], img.shape[0] - axes[0]), np.random.randint(axes[1], img.shape[1] - axes[1]))
        angle = np.random.randint(0, 360)

        segments = []
        for start_angle in range(0, 360, 60):
            end_angle = start_angle + 60
            segment = np.zeros_like(img)
            cv2.ellipse(segment, center, axes, angle,
                        start_angle, end_angle, 255, 1)
            sigma = np.random.uniform(0.5, 1)
            segment = cv2.GaussianBlur(segment, (0, 0), sigma)
            segment = np.clip(segment, 0, 255)
            segments.append(segment)
        ellipse = np.maximum.reduce(segments)
        img = np.maximum(img, ellipse)
    return img/img.max()


def sim_raw_generator(period, m, W, psf, rng_seed, theta_start, phi, noise_level, magnitude, emitter_type):
    np.random.seed(rng_seed)
    if emitter_type == 'curve':
        curve = generate_bezier_curves_2d(m, W)
    if emitter_type == 'tube':
        tube_width = m
        curve = generate_bezier_curves_tube(tube_width, W)
    if emitter_type == 'ring':
        circle_num = np.random.randint(100, 200)
        radius = m
        curve = generate_ring(circle_num, W, radius)

    curve = curve - np.min(curve)
    curve = curve / np.max(curve)

    Ks = get_Ks(theta_start, 3, period, curve.shape[-1])
    cosine_patterns = cosine_light_pattern(
        curve.shape, Ks, phases=phi, M=magnitude)

    modified_curves = curve[np.newaxis, ...] * cosine_patterns
    res = convolve_curve_with_psf(modified_curves, psf)
    res = percentile_norm(res)
    return curve, modified_curves, res, cosine_patterns


def convolve_fft_batch(xin, k):
    # FFT convolution
    y = fftconvolve(xin, k, mode='same', axes=(-2, -1))
    return y

def convolve_fft(xin, k):
    x = xin.reshape([-1, xin.shape[-2], xin.shape[-1]])
    k = k.reshape([k.shape[-2], k.shape[-1]])
    y = np.array([convolve_fft_batch(xi, k) for xi in x])
    return y.reshape(xin.shape)



def rot2d(A, theta):
    rot = np.array([[np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])
    return np.matmul(rot, A)


def get_Ks(theta_start, n_angle, period, img_size):
    start_k = np.array([np.cos(np.pi / 180 * theta_start),
                        np.sin(np.pi / 180 * theta_start)])
    ks = np.stack([rot2d(start_k, i*2*np.pi/n_angle)
                   for i in range(n_angle)]) * img_size / period
    return ks

def estimate_phase(sim, x, y):
    N = sim.shape[0]
    Dc = []
    for i in range(N):
        Dc.append(sim[i:i+1, ...] * np.exp(-1j*2/N*np.pi*i))
    Dc_fft = np.fft.fftshift(np.fft.fft2(sum(Dc), axes=(-2, -1)))
    phi = np.angle(Dc_fft[..., round(x):round(x)+1, round(y):round(y)+1])
    return phi

def estimate_phase_sims(D, Ks):
    mid_x = (D.shape[-2] + 1) // 2
    mid_y = (D.shape[-1] + 1) // 2
    phi_1 = estimate_phase(D[0, 0:3, ...], Ks[0, 0]+mid_x, Ks[0, 1]+mid_y)
    phi_2 = estimate_phase(D[0, 3:6, ...], Ks[1, 0]+mid_x, Ks[1, 1]+mid_y)
    phi_3 = estimate_phase(D[0, 6:9, ...], Ks[2, 0]+mid_x, Ks[2, 1]+mid_y)
    phi = np.concatenate([phi_1, phi_2, phi_3], axis=0)
    phi = np.squeeze(phi)
    return phi

def FFT_sim_recon(D, phi_in, theta_start, period, M, OTF, wienner_w):
    M = np.array([M, M, M])
    n_angle = 3
    n_phase = 3
    Ks = get_Ks(theta_start, 3, period, D.shape[-1])
    phi = estimate_phase_sims(D, Ks)
    D -= D.min()
    D /= D.max()
    D_fft = np.fft.fftshift(np.fft.fft2(D), axes=(-2, -1))
    D_fft = D_fft.reshape(
        [D_fft.shape[0], n_angle, n_phase, D_fft.shape[2], D_fft.shape[3]])
    # Pad 2x
    pad_height = D_fft.shape[3] // 2
    pad_width = D_fft.shape[4] // 2
    D_fft = np.pad(D_fft, pad_width=((0, 0), (0, 0), (0, 0), (pad_height,
                    pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    DN = D_fft.mean(axis=2)
    phase_s = np.array([np.exp(1j*2*np.pi*i/n_phase)
                        for i in range(n_phase)])
    phase_s = phase_s.reshape([1, 1, n_phase, 1, 1])
    DS = D_fft * phase_s
    DS = DS.mean(axis=2)
    phase_p = np.array([np.exp(-1j*2*np.pi*i/n_phase)
                        for i in range(n_phase)])
    phase_p = phase_p.reshape([1, 1, n_phase, 1, 1])
    DP = D_fft * phase_p
    DP = DP.mean(axis=2)

    DS = 2 * DS * \
        np.exp(1j*phi.reshape([1, n_angle, 1, 1])) / \
        M.reshape([1, n_angle, 1, 1])
    DP = 2 * DP * \
        np.exp(-1j*phi.reshape([1, n_angle, 1, 1])) / \
        M.reshape([1, n_angle, 1, 1])

    DSList, DPList = [], []
    for a, k in enumerate(Ks):
        ky, kx = k
        DSList.append(np.roll(np.roll(
            DS[:, a, ...], +np.int32(np.round(ky)), axis=-2), +np.int32(np.round(kx)), axis=-1))
        DPList.append(np.roll(np.roll(
            DP[:, a, ...], -np.int32(np.round(ky)), axis=-2), -np.int32(np.round(kx)), axis=-1))
    DS = np.stack(DSList, axis=1)
    DP = np.stack(DPList, axis=1)

    # Expand OTF
    if len(OTF.shape) == 2:
        OTF = np.repeat(OTF[np.newaxis, np.newaxis, :, :], 3, axis=1)
    elif len(OTF.shape) == 3 and OTF.shape[0] == 3:
        OTF = OTF[np.newaxis, :, :, :]
    
    OTF_S_List, OTF_P_List = [], []
    for a, k in enumerate(Ks):
        ky, kx = k
        OTF_S_List.append(np.roll(np.roll(
            OTF[:, a, ...], +np.int32(np.round(ky)), axis=-2), +np.int32(np.round(kx)), axis=-1))
        OTF_P_List.append(np.roll(np.roll(
            OTF[:, a, ...], -np.int32(np.round(ky)), axis=-2), -np.int32(np.round(kx)), axis=-1))
        
    OTF_S = np.stack(OTF_S_List, axis=1)
    OTF_P = np.stack(OTF_P_List, axis=1)
    OTF_N = OTF

    DFinal = (DS * np.abs(OTF_S) + DP * np.abs(OTF_P) + DN * np.abs(OTF_N)).sum(axis=1, keepdims=True) / ((np.abs(OTF_S) ** 2 + np.abs(OTF_P) ** 2 + np.abs(OTF_N) ** 2).sum(axis=1, keepdims=True) + wienner_w)
    
    rec = np.abs(np.fft.ifft2(np.fft.ifftshift(DFinal, axes=(-2, -1))))
    
    return rec


def process_single_simulation(args):
    i, period, scale, m, W, psf, rng_seed, noise, magnitude, emitter_type, ave_photon, OTF, wienner_w, data_path, gt_path, is_train = args
    np.random.seed(rng_seed)
    
    phi = (np.random.rand(3) - 0.5) * 2 * np.pi if is_train else np.random.rand(3) * 2 * np.pi
    theta_start = np.random.rand() * 360
    curve, modified_curves, res, cosine_patterns = sim_raw_generator(
        period * scale, m, W * scale, psf, rng_seed, theta_start, phi, noise, magnitude, emitter_type)

    modified_curves = resize(modified_curves, (res.shape[0], res.shape[1], W, W))
    res = resize(res, (res.shape[0], res.shape[1], W, W))

    # Add noise
    res_std = res / res.std()
    no_empty_area = res_std > 1e-2
    res = res / res[no_empty_area].mean() * ave_photon
    res = np.random.poisson(res) + np.random.normal(0, noise * res.std(), res.shape)

    recon = FFT_sim_recon(res.transpose(1, 0, 2, 3), phi, theta_start, period, magnitude, OTF, wienner_w)

    # Save results
    save_tiff_imagej_compatible(f'{gt_path}/{i}_lp.tif', cosine_patterns.astype(np.float32), "CZYX")
    save_tiff_imagej_compatible(f'{gt_path}/{i}_no_empty.tif', no_empty_area.astype(np.float32), "CZYX")
    save_tiff_imagej_compatible(f'{gt_path}/{i}_recon.tif', recon.astype(np.float32), "CZYX")
    save_tiff_imagej_compatible(f'{gt_path}/{i}.tif', curve.astype(np.float32), "YX")
    save_tiff_imagej_compatible(f'{data_path}/{i}.tif', res.astype(np.float32), "CZYX")

    # Save config
    config = {
        'period': period,
        'num': m,
        'noise_level': noise,
        'M': magnitude,
        'theta_start': theta_start,
        'phi': phi.tolist(),
    }
    with open(f'{gt_path}/{i}_config.json', 'w') as f:
        json.dump(config, f)

    return i


def main_sim(period=5, noise=1., ave_photon=100, LAMBDA=500, dirname='line', sparsity=30, magnitude=0.3, emitter_type='ring', wienner_w=0.1, W=256, bs_train=10, bs_test=100):
    print(dirname)
    scale = 6
    if isinstance(sparsity, list):
        m = [sparsity[0] * scale / 3, sparsity[1] * scale / 3]
    else:
        m = sparsity if emitter_type == "curve" else sparsity * scale / 3

    rng = np.random.default_rng(42)
    psf = np.array(new_psf_2d(LAMBDA, 49 * scale // 3, 62.6 / scale))[np.newaxis, ...]
    print(f"PSF shape: {psf.shape}")
    
    # Generate OTF
    raw_psf_otf = new_psf_2d(LAMBDA, 2 * W, 62.6 / 2)
    OTF = np.fft.fftshift(np.fft.fft2(raw_psf_otf), axes=(-2, -1))
    print(f"OTF shape: {OTF.shape}")
    
    OTF = np.repeat(OTF[np.newaxis, np.newaxis, :, :], 3, axis=1)
    print(f"OTF final shape: {OTF.shape}")

    train_set_path = f'{dirname}/train'
    train_set_gt_path = f'{dirname}/train_gt'
    test_set_path = f'{dirname}/test'
    test_set_gt_path = f'{dirname}/test_gt'
    
    for path in [train_set_path, train_set_gt_path, test_set_path, test_set_gt_path]:
        os.makedirs(path, exist_ok=True)
    
    save_tiff_imagej_compatible(f'{dirname}/psf.tif', psf, "ZYX")

    def run_simulations(bs, data_path, gt_path, is_train=True):
        with multiprocessing.Pool(32) as pool:
            tasks = [(i, period, scale, m, W, psf, rng.integers(1e9), noise, magnitude, emitter_type, ave_photon, OTF, wienner_w, data_path, gt_path, is_train) for i in range(bs)]
            for _ in tqdm.tqdm(pool.imap_unordered(process_single_simulation, tasks), total=bs):
                pass

    run_simulations(bs_train, train_set_path, train_set_gt_path, is_train=True)
    run_simulations(bs_test, test_set_path, test_set_gt_path, is_train=False)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser(description='Generate SIM simulation data and save to local directories')
    parser.add_argument('--output_prefix', type=str, default='./data/SIM-simulation', help='Output directory prefix')
    
    args = parser.parse_args()
    
    LAMBDA = 488
    period = 5
    noise = 0.1
    ave_photon = 1000
    M = 0.8
    line_num = 20
    tube_width = 3
    radius = [3, 4]
    output_prefix = args.output_prefix
    W = 256
    bs_train = 1
    bs_test = 1
    
    # CURVE
    savedir = f'{output_prefix}/curve'
    os.makedirs(savedir, exist_ok=True)
    
    main_sim(period, noise, ave_photon, LAMBDA, f'{savedir}/light_pattern_period/standard', line_num, M, 'curve', W=W, bs_train=bs_train, bs_test=bs_test)

    for period1 in [3, 4, 7, 10]:
        main_sim(period1, noise, ave_photon, LAMBDA,
                 f'{savedir}/light_pattern_period/{period1}', line_num, M, 'curve', W=W, bs_train=bs_train, bs_test=bs_test)
        
    for ave_photon1 in [0.1, 0.5, 1, 10, 500, 1000]:
        main_sim(period, noise, ave_photon1, LAMBDA,
                 f'{savedir}/ave_photon/{ave_photon1}', line_num, M, 'curve', W=W, bs_train=bs_train, bs_test=bs_test)

    for line_num1 in [5, 10, 30, 50]:
        main_sim(period, noise, ave_photon, LAMBDA,
                 f'{savedir}/sparsity/{line_num1}', line_num1, M, 'curve', W=W, bs_train=bs_train, bs_test=bs_test)

    # TUBE
    savedir = f'{output_prefix}/tube'
    os.makedirs(savedir, exist_ok=True)
    
    main_sim(period, noise, ave_photon, LAMBDA, f'{savedir}/light_pattern_period/standard', tube_width, M, 'tube', W=W, bs_train=bs_train, bs_test=bs_test)

    for period1 in [3, 4, 7, 10]:
        main_sim(period1, noise, ave_photon, LAMBDA,
                 f'{savedir}/light_pattern_period/{period1}', tube_width, M, 'tube', W=W, bs_train=bs_train, bs_test=bs_test)
        
    for ave_photon1 in [0.1, 0.5, 1, 10, 500, 1000]:
        main_sim(period, noise, ave_photon1, LAMBDA,
                 f'{savedir}/ave_photon/{ave_photon1}', tube_width, M, 'tube', W=W, bs_train=bs_train, bs_test=bs_test)

    for tube_width1 in [1, 2, 4, 5]:
        main_sim(period, noise, ave_photon, LAMBDA,
                 f'{savedir}/tube_width/{tube_width1}', tube_width1, M, 'tube', W=W, bs_train=bs_train, bs_test=bs_test)
        
    # RING
    savedir = f'{output_prefix}/ring'
    os.makedirs(savedir, exist_ok=True)
    
    main_sim(period, noise, ave_photon, LAMBDA, f'{savedir}/light_pattern_period/standard', radius, M, 'ring', W=W, bs_train=bs_train, bs_test=bs_test)

    for period1 in [3, 4, 7, 10]:
        main_sim(period1, noise, ave_photon, LAMBDA,
                 f'{savedir}/light_pattern_period/{period1}', radius, M, 'ring', W=W, bs_train=bs_train, bs_test=bs_test)
        
    for ave_photon1 in [0.1, 0.5, 1, 10, 500, 1000]:
        main_sim(period, noise, ave_photon1, LAMBDA,
                 f'{savedir}/ave_photon/{ave_photon1}', radius, M, 'ring', W=W, bs_train=bs_train, bs_test=bs_test)

    for radius1 in [[1, 2], [2, 3], [4, 5], [5, 6]]:
        main_sim(period, noise, ave_photon, LAMBDA,
                 f'{savedir}/radius/{np.mean(radius1)}', radius1, M, 'ring', W=W, bs_train=bs_train, bs_test=bs_test)


    


