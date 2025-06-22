# Simplified network for SIMFormer distillation
# Only outputs emitter predictions without physics simulation components

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax_mae import mae_vit_base_ada_patch16 as mae
from jax_mae.vision_transformer import Block
from utils_net import FCN
from typing import Callable
import functools

class Decoder(nn.Module):
    """Simplified decoder for emitter prediction only"""
    features: int = 64
    patch_size: tuple[int, int, int] = (1, 16, 16)
    out_p: int = 1
    kernel_init: Callable = nn.initializers.kaiming_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    
    @nn.compact
    def __call__(self, x):
        assert self.patch_size[1] == self.patch_size[2]
        up_scale_times = int(np.log2(self.patch_size[1]))
        b, d, h, w, c = x.shape
        
        # Reshape for spatial processing
        f = jnp.einsum('bdhwc->bhwdc', x).reshape((b, h, w, d*c))
        
        # Progressive upsampling with convolutions
        for t in range(up_scale_times):
            f = jax.image.resize(f, shape=(f.shape[0], f.shape[1]*2, f.shape[2]*2, f.shape[3]), method='linear')
            features_num = 2 ** (up_scale_times - t - 1) * self.features
            f = nn.Conv(features_num, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(f)
            f = nn.gelu(f)
            f = nn.Conv(features_num, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(f)
            f = nn.gelu(f)
            f = nn.Conv(features_num, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(f)
            f = nn.gelu(f)
        
        # Final projection to output
        f = nn.Conv(self.out_p, kernel_size=(1, 1), kernel_init=self.kernel_init, bias_init=self.bias_init)(f)
        y = jnp.einsum('bhwd->bdhw', f)[..., None]
        return y


class ViT_CNN_Distillation(nn.Module):
    """Simplified ViT-CNN for distillation - emitter output only"""
    img_size: tuple[int, int, int] = (1, 192, 192)
    patch_size: tuple[int, int, int] = (1, 16, 16)
    
    def setup(self):
        self.MAE = mae(img_size=self.img_size, PatchEmbed_type="mae3d", patch_size=self.patch_size)
        self.emitter_decoder = Decoder(patch_size=self.patch_size, out_p=1)
        # No background, light pattern decoders, or low-rank coding needed for distillation
    
    def unpatchify_feature(self, x):
        """Convert patches back to spatial features
        x: (N, L, C)
        f: (N, D, H, W, C)
        """
        p = self.patch_size
        d, h, w = (s // p[i] for i, s in enumerate(self.img_size))
        f = x.reshape((x.shape[0], d, h, w, -1))
        return f

    def __call__(self, x, args, training, mask_ratio):
        # Input: batch, C, Z, Y, X
        img_t = x.transpose([0, 2, 3, 4, 1])
        
        # ViT encoder with optional masking
        rng = self.make_rng("random_masking")
        
        if training:
            mask_ratio = float(mask_ratio)
        else:
            mask_ratio = 0.0
            
        latent, mask, ids_restore = self.MAE.forward_encoder(img_t, mask_ratio=mask_ratio, train=training, rng=rng)
        latent_embed_to_blk = self.MAE.forward_decoder_embed(latent, ids_restore)
        Features = self.MAE.forward_decoder_blks(latent_embed_to_blk, train=training)[:, 1:, :]
        
        # Process mask
        mask = jnp.tile(jnp.expand_dims(mask, -1), (1, 1, x.shape[1]*self.patch_size[0]*self.patch_size[1]*self.patch_size[2]))
        mask = self.MAE.unpatchify(mask)
        mask = mask.transpose([0, 4, 1, 2, 3])

        # Decode to emitter only
        f = self.unpatchify_feature(Features)
        emitter = self.emitter_decoder(f).transpose([0, 4, 1, 2, 3])
        
        return emitter, mask


class PiMAE_Distillation(nn.Module):
    """Main distillation model - simplified without physics simulation"""
    img_size: tuple[int, int] = (9, 224, 224)
    patch_size: tuple[int, int, int] = (3, 16, 16)
    
    def setup(self):
        self.pt_predictor = ViT_CNN_Distillation(self.img_size, self.patch_size)
        
    def __call__(self, x_clean, args, training):
        rng = self.make_rng("random_masking")
        rng_noise_1, rng_noise_2, rng_noise_3, rng_noise_4 = jax.random.split(rng, 4)

        # Input processing with optional noise augmentation
        x = x_clean
        x = jax.image.resize(x, shape=(x.shape[0], x.shape[1], x.shape[2], 
                                      x.shape[3]*args.rescale[0], x.shape[4]*args.rescale[1]), 
                           method='linear')
        
        if training:
            x_mean = jnp.mean(x, axis=(1, 2, 3, 4), keepdims=True)
            noise = x_mean * jax.lax.stop_gradient(
                jax.random.normal(rng_noise_1, x.shape) * 
                jax.random.uniform(rng_noise_2, (x.shape[0], 1, 1, 1, 1), 
                                 minval=0, maxval=args.add_gaussian_noise))
            
            if args.shot_noise_scale is None:
                x = x + noise
            else:
                scale = jax.lax.stop_gradient(
                    jax.random.uniform(rng_noise_3, (x.shape[0], 1, 1, 1, 1), 
                                     minval=args.shot_noise_scale, 
                                     maxval=args.shot_noise_scale*10))
                x_p = jax.random.poisson(rng_noise_4, x * scale) / scale
                x = x_p + noise
        
        # Normalize input
        x_min = jnp.percentile(x, 0.01, axis=(1, 2, 3, 4), keepdims=True)
        x = x - x_min
        
        # Get emitter prediction
        emitter, mask = self.pt_predictor(x, args, training, args.mask_ratio)
        emitter = jax.nn.softplus(emitter)
        
        # Downsample mask to match output resolution
        mask_real = nn.avg_pool(mask.transpose([0, 2, 3, 4, 1]), 
                               (1, args.rescale[0], args.rescale[1]), 
                               (1, args.rescale[0], args.rescale[1]), 
                               padding="VALID").transpose([0, 4, 1, 2, 3])
        
        return {
            "x_real": x_clean,
            "x_up": x * (1 - mask), 
            "deconv": emitter,
            "mask": mask_real
        }