import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Any, Union, Callable, Optional
from .drop import DropPath


class Extractor(nn.Module):
    dim: int
    num_heads: int = 6
    with_cffn: bool = True
    cffn_ratio: float = 0.25
    drop: float = 0.
    drop_path: float = 0.
    norm_layer: Union[Callable, nn.Module] = nn.LayerNorm

    def setup(self):
        self.query_norm = self.norm_layer()
        self.feat_norm = self.norm_layer()
        self.attn = Attention(dim=self.dim, num_heads=self.num_heads, qkv_bias=True)

        if self.with_cffn:
            self.ffn = ConvFFN(in_features=self.dim, hidden_features=int(self.dim * self.cffn_ratio), drop=self.drop)
            self.ffn_norm = self.norm_layer(self.dim)
            self.drop_path_layer = DropPath(self.drop_path) if self.drop_path > 0. else IdentityLayer()

    def __call__(self, q, f, train: bool = True):
        attn = self.attn(self.query_norm(q), self.feat_norm(f), train)
        q = q + attn

        if self.with_cffn:
            q = q + self.drop_path_layer(self.ffn(self.ffn_norm(q), train=train), train=train)
        return q
    
class Injector(nn.Module):
    dim: int
    num_heads: int = 6
    init_values: float = 0.
    norm_layer: Union[Callable, nn.Module] = nn.LayerNorm

    def setup(self):
        self.query_norm = self.norm_layer()
        self.feat_norm = self.norm_layer()
        self.attn = Attention(dim=self.dim, num_heads=self.num_heads, qkv_bias=True)
        self.gamma = self.param('gamma', nn.initializers.constant(self.init_values), (self.dim,))

    def __call__(self, q, f, train: bool = True):
        attn = self.attn(self.query_norm(q), self.feat_norm(f), train)
        return q + self.gamma * attn
    

class SpatialPriorModule(nn.Module):
    inplanes: int = 64
    embed_dim: int = 384
    kernel_init: Callable = nn.initializers.kaiming_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    patch_z: int = 1

    @nn.compact
    def __call__(self, x):
        x_shape = x.shape
        x = x.reshape((x_shape[0]*x_shape[1], x_shape[2], x_shape[3], x_shape[4]))
        x = nn.Conv(features=self.inplanes, kernel_size=(3, 3), strides=(2, 2), padding='SAME', kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.inplanes, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.inplanes, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        x = nn.Conv(features=self.inplanes*2, kernel_size=(3, 3), strides=(2, 2), padding='SAME', kernel_init=self.kernel_init, bias_init=self.bias_init)(x)

        x = nn.Conv(features=self.inplanes*4, kernel_size=(3, 3), strides=(2, 2), padding='SAME', kernel_init=self.kernel_init, bias_init=self.bias_init)(x)

        x = nn.Conv(features=self.embed_dim, kernel_size=(1, 1), strides=(1, 1), padding='SAME', kernel_init=self.kernel_init, bias_init=self.bias_init)(x)

        x = x.reshape([x_shape[0], x_shape[1], -1, x.shape[-1]])
        x = jax.image.resize(x, shape=(x.shape[0], x.shape[1]//self.patch_z, x.shape[2], x.shape[3]), method='linear')

        x = x.reshape([x_shape[0], -1, x.shape[-1]])
        return x
    

dense_kernel_init = nn.initializers.xavier_uniform()

class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    attn_drop: float = 0.
    proj_drop: float = 0.

    @nn.compact
    def __call__(self, x, f, train: bool = True):
        head_dim = self.dim // self.num_heads
        scale = head_dim ** -0.5
        
        q_layer = nn.Dense(self.dim, use_bias=self.qkv_bias, kernel_init=dense_kernel_init)
        k_layer = nn.Dense(self.dim, use_bias=self.qkv_bias, kernel_init=dense_kernel_init)
        v_layer = nn.Dense(self.dim, use_bias=self.qkv_bias, kernel_init=dense_kernel_init)
        proj_layer = nn.Dense(self.dim, kernel_init=dense_kernel_init)

        B, N, _ = x.shape
        q = q_layer(x).reshape(B, N, self.num_heads, head_dim).transpose((0, 2, 1, 3))
        k = k_layer(f).reshape(B, N, self.num_heads, head_dim).transpose((0, 2, 1, 3))
        v = v_layer(f).reshape(B, N, self.num_heads, head_dim).transpose((0, 2, 1, 3))

        attn = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attn = nn.softmax(attn, axis=-1)
        if self.attn_drop != 0:
            attn = nn.Dropout(self.attn_drop, deterministic=not train, name="attn_drop_layer")(attn)
        x = jnp.swapaxes((attn @ v), 1, 2).reshape(B, N, self.dim)
        x = proj_layer(x)
        if self.proj_drop != 0:
            x = nn.Dropout(self.proj_drop, deterministic=not train, name="proj_drop_layer")(x)
        return x

    

class ConvFFN(nn.Module):
    in_features: int
    hidden_features: int = None
    out_features: int = None
    drop: float = 0.
    kernel_init: Callable = nn.initializers.kaiming_normal()
    bias_init: Callable = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x, train: bool = True):
        hidden_features = self.hidden_features or self.in_features
        out_features = self.out_features or self.in_features

        x = nn.Dense(hidden_features, kernel_init=dense_kernel_init)(x)
        x = nn.Conv(features=self.hidden_features, kernel_size=(3, 3), padding='SAME', kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.drop)(x, deterministic=not train)
        x = nn.Dense(out_features, kernel_init=dense_kernel_init)(x)
        x = nn.Dropout(self.drop)(x, deterministic=not train)
        return x
    

class IdentityLayer(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool = True):
        return x
