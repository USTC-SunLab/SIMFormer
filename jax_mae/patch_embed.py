# PatchEmbed layer implementation
import jax
import jax.numpy as jnp
import flax.linen as nn
from .utils import to_2tuple, to_3tuple
from typing import Optional, Callable, Union


class PatchEmbed(nn.Module):
    img_size: Optional[Union[tuple, int]] = 224
    patch_dim: Optional[Union[tuple, int]] = 16
    embed_dim: int = 768
    norm_layer: Optional[Callable] = None
    flatten: bool = True

    def setup(self):
        img_size = to_2tuple(self.img_size)
        patch_size = to_2tuple(self.patch_dim)
        grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_patches = grid_size[0] * grid_size[1]
        self.proj = nn.Conv(self.embed_dim, kernel_size=patch_size, strides=patch_size, padding='VALID',
                            kernel_init=nn.initializers.xavier_uniform())

    def __call__(self, inputs, train: bool = True):
        B, H, W, C = inputs.shape
        outputs = self.proj(inputs)
        if self.flatten:
            outputs = outputs.reshape(B, -1, self.embed_dim)
        if self.norm_layer is not None:
            outputs = self.norm_layer(outputs)
        return outputs


class PatchEmbed3d(nn.Module):
    img_size: Optional[Union[tuple, int]] = 224
    patch_dim: Optional[Union[tuple, int]] = 16
    embed_dim: int = 768
    norm_layer: Optional[Callable] = None
    flatten: bool = True

    def setup(self):
        img_size = to_3tuple(self.img_size)
        patch_size = to_3tuple(self.patch_dim)
        grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_patches_Z = grid_size[0]
        self.num_patches_N = grid_size[1] * grid_size[2]
        self.proj = nn.Conv(self.embed_dim, kernel_size=patch_size, strides=patch_size, padding='VALID',
                            kernel_init=nn.initializers.xavier_uniform())

    def __call__(self, inputs, train: bool = True):
        outputs = self.proj(inputs)
        B, Z, H, W, C = outputs.shape
        if self.flatten:
            outputs = outputs.reshape(B, Z, H*W, C)
        if self.norm_layer is not None:
            outputs = self.norm_layer(outputs)
        return outputs