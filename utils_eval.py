import numpy as np
from scipy.optimize import minimize
from skimage.metrics import structural_similarity
from skimage.metrics import normalized_root_mse
import jax.numpy as jnp
import jax
import cv2
import pdb

def cross_correlation(img1, img2):
    # Cross-correlation via FFT
    f1 = np.fft.fft2(img1)
    f2 = np.conj(np.fft.fft2(img2))
    result = np.fft.ifft2(f1 * f2)
    return np.abs(result)




def percentile_norm(img, p_low=0.1, p_high=99.9):
    # Percentile normalization
    p_low = np.percentile(img, p_low)
    p_high = np.percentile(img, p_high)
    if p_high == p_low:
        return np.zeros_like(img)
    else:
        return (img - p_low) / (p_high - p_low)




def gaussian_kernel(sigma, psf_size):
    # Generate Gaussian kernel
    size = [psf_size[i] // 2 for i in range(2)]
    coords = np.mgrid[-size[0]:size[0] + 1, -size[1]:size[1] + 1]
    kernel = np.exp(-jnp.sum((coords / sigma) ** 2, axis=0) / 2)
    kernel /= np.sum(kernel)
    return jax.lax.stop_gradient(kernel)



def eval_nrmse(x, y):
    # Evaluate NRMSE
    x_f = np.array(x.reshape([-1, *x.shape[-2:]]))
    y_f = np.array(y.reshape([-1, *y.shape[-2:]]))
    nrmse_list = []
    for i in range(x_f.shape[0]):
        img1, img2 = x_f[i], y_f[i]
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)
        img1 = percentile_norm(img1)
        img2 = percentile_norm(img2)
        nrmse = normalized_root_mse(img1, img2)
        nrmse_list.append(nrmse)
    return np.mean(nrmse_list)


