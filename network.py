from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax_mae import mae_vit_base_ada_patch16 as mae
from jax_mae.vision_transformer import Block
from typing import Callable
import functools
import pdb


class FCN(nn.Module):
    out: int = 1
    @nn.compact
    def __call__(self, x, training):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = nn.Conv(features=48, kernel_size=[3, 3, 3], kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros_init(), padding='same')(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.activation.leaky_relu(x)
        x = nn.Conv(features=96, kernel_size=[3, 3, 3], kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros_init(), padding='same')(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.activation.leaky_relu(x)
        x = nn.Conv(features=96, kernel_size=[3, 3, 3], kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros_init(), padding='same')(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.activation.leaky_relu(x)
        x = nn.Conv(features=48, kernel_size=[3, 3, 3], kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros_init(), padding='same')(x)
        x = nn.activation.leaky_relu(x)
        x = nn.Conv(features=self.out, kernel_size=[3, 3, 3], kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros_init(), padding='same')(x)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        return x


class low_rank_coding(nn.Module):
    rank: int = 512
    kernel_init: Callable = nn.initializers.kaiming_normal()
    @nn.compact
    def __call__(self, x):
        y = Block(x.shape[-1], 4, 4, qkv_bias=True, name="decoder_block_lrc_0")(x)
        y = nn.Dense(512, name="lrc_layer_0", use_bias=False, kernel_init=self.kernel_init)(y)
        y = nn.avg_pool(y, tuple([1, 512 // self.rank]), padding="SAME")
        y = nn.Dense(x.shape[-1], name="lrc_layer_1", use_bias=False, kernel_init=self.kernel_init)(y)
        y = Block(x.shape[-1], 4, 4, qkv_bias=True, name="decoder_block_lrc_1")(y)
        return y
    

    
class Decoder(nn.Module):
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
        # f = jnp.einsum('bdhwc->bhwdc', x).reshape((b*h*w, d*c))
        # f = nn.Dense(features=self.out_p*c)(f).reshape((b, h, w, self.out_p*c))
        f = jnp.einsum('bdhwc->bhwdc', x).reshape((b, h, w, d*c))
        for t in range(up_scale_times):
            f = jax.image.resize(f, shape=(f.shape[0], f.shape[1]*2, f.shape[2]*2, f.shape[3]), method='linear')
            features_num = 2 ** (up_scale_times - t - 1) * self.features
            f = nn.Conv(features_num, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(f)
            f = nn.gelu(f)
            f = nn.Conv(features_num, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(f)
            f = nn.gelu(f)
            f = nn.Conv(features_num, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(f)
            f = nn.gelu(f)
        f = nn.Conv(self.out_p, kernel_size=(1, 1), kernel_init=self.kernel_init, bias_init=self.bias_init)(f)
        y = jnp.einsum('bhwd->bdhw', f)[..., None]
        return y


class ViT_CNN(nn.Module):
    img_size: tuple[int, int, int] = (1, 192, 192)
    patch_size: tuple[int, int, int] = (1, 16, 16)
    rank: int = 1
    
    def setup(self):
        self.MAE = mae(img_size=self.img_size, PatchEmbed_type="mae3d", patch_size=self.patch_size)
        self.emitter_decoder = Decoder(patch_size=self.patch_size, out_p=1)
        self.bg_decoder = Decoder(patch_size=self.patch_size, out_p=1)
        self.lf_decoder = Decoder(patch_size=self.patch_size, out_p=self.img_size[0])
        self.lorm = low_rank_coding(self.rank)
        

    def unpatchify_feature(self, x):
        """
        x: (N, L, C)
        f: (N, D, H, W, C)
        """
        p = self.patch_size
        d, h, w = (s // p[i] for i, s in enumerate(self.img_size))
        f = x.reshape((x.shape[0], d, h, w, -1))
        return f

    def __call__(self, x, args, training, mask_ratio):
        # batch, C, Z, Y, X
        img_t = x.transpose([0, 2, 3, 4, 1])
        # ViT encoder
        rng = self.make_rng("random_masking")
        
        if training:
            mask_ratio = float(mask_ratio)
        else:
            mask_ratio = 0.0
        latent, mask, ids_restore = self.MAE.forward_encoder(img_t, mask_ratio=mask_ratio, train=training, rng=rng)
        laten_embed_to_blk = self.MAE.forward_decoder_embed(latent, ids_restore)
        Features = self.MAE.forward_decoder_blks(laten_embed_to_blk, train=training)[:, 1:, :]
        

        mask = jnp.tile(jnp.expand_dims(mask, -1), (1, 1, x.shape[1]*self.patch_size[0]*self.patch_size[1]*self.patch_size[2]))
        mask = self.MAE.unpatchify(mask)
        mask = mask.transpose([0, 4, 1, 2, 3])

        f = self.unpatchify_feature(Features)
        # emitter
        emitter = self.emitter_decoder(f).transpose([0, 4, 1, 2, 3])
        # background
        bg = self.bg_decoder(f)
        bg = jax.nn.softplus(bg)
        bg = nn.avg_pool(bg, tuple([self.patch_size[0], self.patch_size[1]*4, self.patch_size[2]*4]), padding="SAME")
        bg = bg.transpose([0, 4, 1, 2, 3])

        F = self.lorm(Features)
        f = self.unpatchify_feature(F)

        
        light_pattern = self.lf_decoder(f).transpose([0, 4, 1, 2, 3])

        return emitter, bg, light_pattern, mask



class PiMAE(nn.Module):
    img_size: tuple[int, int] = (9, 224, 224)
    patch_size: tuple[int, int, int] = (3, 16, 16)
    psf_size: tuple[int, int, int] = (64, 64)
    rank: int = 1

    def setup(self):
        # emitter
        self.pt_predictor = ViT_CNN(self.img_size, self.patch_size, self.rank)
        # PSF
        self.psf_seed = self.param("psf_seed", nn.initializers.normal(1), [1, 32, 1, *self.psf_size])
        self.PSF_predictor = FCN(1)

        
    def __call__(self, x_clean, args, training):
        rng = self.make_rng("random_masking")
        rng_noise_1, rng_noise_2, rng_noise_3 = jax.random.split(rng, 3)

        # x : [batch, channel, z, y, x]
        x = x_clean
        x = jax.image.resize(x, shape=(x.shape[0], x.shape[1], x.shape[2], x.shape[3]*args.rescale[0], x.shape[4]*args.rescale[1]), method='linear')
        
        if training:
            x_mean = jnp.mean(x, axis=[1, 2, 3, 4], keepdims=True)
            noise = x_mean * jax.lax.stop_gradient(jax.random.normal(rng_noise_1, x.shape) \
                            * jax.random.uniform(rng_noise_2, (x.shape[0], 1, 1, 1, 1), minval=0.0, maxval=args.add_noise))
            x = x + noise
        
        emitter, bg, light_pattern, mask = self.pt_predictor(x, args, training, args.mask_ratio)
        emitter = jax.nn.softplus(emitter)
        
        light_pattern = jax.nn.softmax(light_pattern, axis=2) * light_pattern.shape[2]

        S = light_pattern * emitter
        
        # PSF
        if args.resume_s1_path is not None:
            # psf = jax.lax.stop_gradient(self.PSF_predictor(jax.lax.stop_gradient(self.psf_seed), training))
            psf = self.PSF_predictor(jax.lax.stop_gradient(self.psf_seed), training)
            # bg = jax.lax.stop_gradient(bg)
        else:
            psf = self.PSF_predictor(jax.lax.stop_gradient(self.psf_seed), training)
        psf = jax.nn.softmax(psf, axis=(2, 3, 4))
        # rec = convolve(S, psf) + bg
        S_f = S.reshape([-1, *S.shape[3:]])
        psf_f = psf.reshape([1, *psf.shape[3:]])
        rec_f = convolve_fft(S_f, psf_f)
        rec = rec_f.reshape(S.shape) + bg
        rec_real = nn.avg_pool(rec.transpose([0, 2, 3, 4, 1]), 
                            (1, args.rescale[0], args.rescale[1]), 
                            (1, args.rescale[0], args.rescale[1]), 
                            padding="VALID").transpose([0, 4, 1, 2, 3])
        mask_real = nn.avg_pool(mask.transpose([0, 2, 3, 4, 1]), 
                            (1, args.rescale[0], args.rescale[1]), 
                            (1, args.rescale[0], args.rescale[1]), 
                            padding="VALID").transpose([0, 4, 1, 2, 3])
        return {
            "x_real": x_clean,
            "x_up": x*(1-mask), 
            "deconv": emitter,
            "light_pattern": light_pattern,
            "background": bg,
            "rec_real": rec_real,
            "rec_up": rec,
            "psf": psf,
            "mask": mask_real
        }


@functools.partial(jax.vmap, in_axes=(0, None))
def convolve(xin, k):
    x = xin.reshape([-1, 1, *xin.shape[2:]])
    k = k.reshape([1, 1, k.shape[3], k.shape[4]])
    dn = jax.lax.conv_dimension_numbers(x.shape, k.shape,('NCHW', 'IOHW', 'NCHW'))
    y = jax.lax.conv_general_dilated(x, k, window_strides =(1, 1), dimension_numbers=dn, padding='SAME', precision='highest')
    return y.reshape(xin.shape)

def convolve_fft(xin, k):
    y = jax.scipy.signal.fftconvolve(xin, k, mode='same')
    return y