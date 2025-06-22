# MaskedAutoencoder implementation
import jax
from functools import partial
import jax.numpy as jnp
import flax.linen as nn
from .patch_embed import PatchEmbed, PatchEmbed3d
from .vision_transformer import Block
from .pos_embed import get_2d_sincos_pos_embed
from .utils import constant_init
from typing import Optional, Callable, Any, Union
from .utils_adapter import Extractor, Injector, SpatialPriorModule
import math


def unbatched_gather(x, ids_keep):
    return x[ids_keep, Ellipsis]


batched_gather = jax.vmap(unbatched_gather)


class MaskedAutoencoderViT(nn.Module):
    img_size: Optional[Union[tuple, int]] = 224
    patch_size: Optional[Union[tuple, int]] = 16
    out_chans: int = 1
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    mlp_ratio: float = 4.
    norm_layer: Optional[Callable] = nn.LayerNorm
    dtype: Any = jnp.float32
    PatchEmbed_type: str = "mae3d"
    adapter: bool = False

    def setup(self):
        if self.PatchEmbed_type == "mae":
            self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.embed_dim)
        elif self.PatchEmbed_type == "mae3d":
            self.patch_embed = PatchEmbed3d(self.img_size, self.patch_size, self.embed_dim)
        else:
            NotImplementedError

        self.cls_token = self.param("cls_token", nn.initializers.normal(0.02),
                               [1, 1, self.embed_dim])

        self.pos_embed_N = self.param("pos_embed_N", nn.initializers.normal(0.02),
                            [1, self.patch_embed.num_patches_N, self.embed_dim])

        self.pos_embed_Z = self.param("pos_embed_Z", nn.initializers.normal(0.02), 
                            [1, self.patch_embed.num_patches_Z, self.embed_dim])

        self.pos_embed_cls = self.param("pos_embed_cls", nn.initializers.normal(0.02),
                            [1, 1, self.embed_dim])
        
        self.blocks = [Block(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer, name="encoder_block_{:02d}".format(i)) for i in range(self.depth)]

        self.encoder_norm = self.norm_layer(name="encoder_norm")

        self.decoder_embed = nn.Dense(self.decoder_embed_dim, use_bias=True)
        self.mask_token = self.param("mask_token", nn.initializers.normal(0.02),
                                    [1, 1, self.decoder_embed_dim])

        self.decoder_blocks = [Block(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio, qkv_bias=True,
                                     norm_layer=self.norm_layer, name="decoder_block_{:02d}".format(i))
                               for i in range(self.decoder_depth)]
        self.decoder_pred = nn.Dense(self.patch_size[0]*self.patch_size[1]*self.patch_size[2]*self.out_chans, use_bias=True)
        self.decoder_norm = self.norm_layer(name="decoder_norm")


    def patchify(self, imgs):
        # Convert images to patches
        C = imgs.shape[-1]
        p = self.patch_embed.patch_size
        assert imgs.shape[1:4] == self.img_size and all(s % p[i] == 0 for i, s in enumerate(self.img_size))
        d, h, w = (s // p[i] for i, s in enumerate(self.img_size))
        x = imgs.reshape((imgs.shape[0], d, p[0], h, p[1], w, p[2], C))
        x = jnp.einsum('ndphqwrc->ndhwcpqr', x)
        x = x.reshape((imgs.shape[0], d, h * w, C, p[0], p[1], p[2]))
        return x
    

    def unpatchify(self, x):
        # Convert patches back to images
        p = self.patch_embed.patch_size
        d, h, w = (s // p[i] for i, s in enumerate(self.img_size))
        x = x.reshape((x.shape[0], d, h, w, -1, p[0], p[1], p[2]))
        x = jnp.einsum('ndhwcpqr->ndphqwrc', x)
        imgs = x.reshape((x.shape[0], self.img_size[0], self.img_size[1], self.img_size[2], -1))
        return imgs


    def random_masking(self, x, mask_ratio, rng):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = jax.random.uniform(rng, (N, L))

        ids_shuffle = jnp.argsort(noise, axis=1)
        ids_restore = jnp.argsort(ids_shuffle, axis=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = batched_gather(x, ids_keep)

        mask = jnp.ones((N, L))
        mask = mask.at[:, :len_keep].set(0)

        mask = batched_gather(mask, ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, train: bool = True, rng=None):
        x = self.patch_embed(x)
        B, Z, N, C = x.shape
        x = x.reshape([B, Z*N, C])
        
        pos_embed = jnp.tile(self.pos_embed_N, (1, Z, 1)) + jnp.repeat(self.pos_embed_Z, repeats=N, axis=1)
        pos_embed = jnp.concatenate([jnp.broadcast_to(self.pos_embed_cls, (pos_embed.shape[0], self.pos_embed_cls.shape[1], self.pos_embed_cls.shape[2])),
                                     pos_embed], axis=1)
        x = x + pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio, rng)
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = jnp.broadcast_to(cls_token, (x.shape[:1] + cls_token.shape[1:]))

        x = jnp.concatenate([cls_tokens, x], axis=1)
        for blk in self.blocks:
            x = blk(x, train=train)

        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, train: bool = True):
        x = self.decoder_embed(x)
        mask_tokens = jnp.broadcast_to(self.mask_token,
                                       (x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], x.shape[-1]))
        x_ = jnp.concatenate([x[:, 1:, :], mask_tokens], axis=1)
        x_ = batched_gather(x_, ids_restore)
        x = jnp.concatenate([x[:, :1, :], x_], axis=1)

        for blk in self.decoder_blocks:
            x = blk(x, train=train)
        x = self.decoder_norm(x)
        return x
    
    def forward_decoder_embed(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = jnp.broadcast_to(self.mask_token,
                                       (x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], x.shape[-1]))
        x_ = jnp.concatenate([x[:, 1:, :], mask_tokens], axis=1)
        x_ = batched_gather(x_, ids_restore)
        x = jnp.concatenate([x[:, :1, :], x_], axis=1)
        return x
    
    def forward_decoder_blks(self, x, train: bool = True):
        for blk in self.decoder_blocks:
            x = blk(x, train=train)
        x = self.decoder_norm(x)
        return x
    
    def forward_decoder_pred(self, x):
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        x = self.unpatchify(x)
        return x
            

    def __call__(self, imgs, mask_ratio: float = 0.75, train: bool = True, rng=None):
        if rng is None:
            rng = self.make_rng("random_masking")
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, train=train, rng=rng)
        feature = self.forward_decoder(latent, ids_restore, train=train)
        pred = self.decoder_pred(feature)[:, 1:, :]

        mask = jnp.tile(jnp.expand_dims(mask, -1), (1, 1, pred.shape[2]))
        return self.unpatchify(pred), self.unpatchify(mask)
    

class ViTAdapter(MaskedAutoencoderViT):
    img_size: Optional[Union[tuple, int]] = 224
    patch_size: Optional[Union[tuple, int]] = 16
    conv_inplanes: int = 64
    init_values: float = 0.
    with_cffn: bool = True
    cffn_ratio: float = 0.25
    add_vit_feature: bool = True

    def setup(self):
        super().setup()
        self.spm = SpatialPriorModule(inplanes=self.conv_inplanes, embed_dim=self.embed_dim, patch_z=self.patch_size[0])
        self.Extractors = [Extractor(self.embed_dim, self.num_heads, self.with_cffn, self.cffn_ratio, name="encoder_extractor_{:02d}".format(i)) for i in range(self.depth)]
        self.Injector = [Injector(self.embed_dim, self.num_heads, self.init_values, name="encoder_injector_{:02d}".format(i)) for i in range(self.depth)]

    def forward_encoder(self, x, mask_ratio, train: bool = True, rng=None):
        c = self.spm(x)

        x = self.patch_embed(x)
        B, Z, N, C = x.shape
        x = x.reshape([B, Z*N, C])
        
        pos_embed = jnp.tile(self.pos_embed_N, (1, Z, 1)) + jnp.repeat(self.pos_embed_Z, repeats=N, axis=1)
        pos_embed = jnp.concatenate([jnp.broadcast_to(self.pos_embed_cls, (pos_embed.shape[0], self.pos_embed_cls.shape[1], self.pos_embed_cls.shape[2])), pos_embed], axis=1)
        x = x + pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio, rng)
        c, _,    _           = self.random_masking(c, mask_ratio, rng)
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = jnp.broadcast_to(cls_token, (x.shape[:1] + cls_token.shape[1:]))
        x = jnp.concatenate([cls_tokens, x], axis=1)

        for i in range(self.depth):
            cls, x = x[:, :1, ], x[:, 1:, ]
            x = self.Injector[i](x, c, train=train)
            
            x = jnp.concatenate([cls, x], axis=1)
            x = self.blocks[i](x, train=train)
            
            cls, x = x[:, :1, ], x[:, 1:, ]
            c = self.Extractors[i](c, x, train=train)

            x = jnp.concatenate([cls, x], axis=1)


        x = self.encoder_norm(x)
        return x, mask, ids_restore

    




def mae_vit_small_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=768, depth=6, num_heads=8,
        decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model

def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=6, num_heads=8,
        decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model

def mae_vit_base_ada_patch16_dec512d8b(**kwargs):
    model = ViTAdapter(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model

def mae_vit_base_patch8_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


mae_vit_small = mae_vit_small_dec512d8b
mae_vit_small_patch16 = mae_vit_small_patch16_dec512d8b
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b
mae_vit_base_ada_patch16 = mae_vit_base_ada_patch16_dec512d8b
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b
mae_vit_base_patch8 = mae_vit_base_patch8_dec512d8b
