import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'False'
import warnings
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence
import pdb

# Multi-scale structural similarity

def _fspecial_gauss_1d(size, sigma):
    # Generate 1D Gaussian filter
    coords = jnp.arange(size, dtype=jnp.float32)
    coords -= size // 2
    g = jnp.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...]

def gaussian_filter_3d(input, win):
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    assert len(input.shape) == 5

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = jax.lax.conv_general_dilated(out, win.swapaxes(2 + i, -1), window_strides=(1, 1, 1), padding='VALID', feature_group_count=C)
        else:
            pass

    return out

def _ssim_3d(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):
    K1, K2 = K
    compensation = 1.
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = gaussian_filter_3d(X, win)
    mu2 = gaussian_filter_3d(Y, win)

    mu1_sq = jnp.square(mu1)
    mu2_sq = jnp.square(mu2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter_3d(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter_3d(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter_3d(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = ssim_map.mean((2, 3, 4))
    cs = cs_map.mean((2, 3, 4))
    return ssim_per_channel, cs

def ms_ssim_3d(
    X, Y, data_range=1.0, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)
):

    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    if len(X.shape) == 5:
        avg_pool = nn.avg_pool
    else:
        raise ValueError(f"Input images should be 5-d  tensors, but got {X.shape}")

    if win is not None:
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = jnp.array(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win[jnp.newaxis, ...]
        win = jnp.repeat(win, X.shape[1], axis=0)

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim_3d(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(nn.relu(cs))
            padding = [(0, 0)] + [(1, 1) for _ in X.shape[3:]]  # Modified to not pad z direction
            X = X.transpose((0, 2, 3, 4, 1))
            Y = Y.transpose((0, 2, 3, 4, 1))
            X = avg_pool(X, window_shape=(1, 2, 2), strides=(1, 2, 2), padding=padding)
            Y = avg_pool(Y, window_shape=(1, 2, 2), strides=(1, 2, 2), padding=padding)
            X = X.transpose((0, 4, 1, 2, 3))
            Y = Y.transpose((0, 4, 1, 2, 3))

    ssim_per_channel = nn.relu(ssim_per_channel)
    mcs_and_ssim = jnp.stack(mcs + [ssim_per_channel], axis=0)
    ms_ssim_val = jnp.prod(mcs_and_ssim ** weights.reshape(-1, 1, 1), axis=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)



if __name__ == "__main__":
    import jax.random as jr
    key = jr.PRNGKey(0)
    X = jr.uniform(key, shape=(2, 1, 16, 144, 144))
    Y = jr.uniform(jr.split(key)[0], shape=(2, 1, 16, 144, 144))
    data_range = 1.0
    size_average = True
    win_size = 5
    win_sigma = 1.5
    result = ms_ssim_3d(X, X+0.1*Y, data_range=data_range, size_average=size_average, win_size=win_size, win_sigma=win_sigma)
    print(f"MS-SSIM Value: {result}")
