# MLP layer implementation
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable


default_init = nn.initializers.lecun_normal()


class Mlp(nn.Module):
    # MLP for vision transformers
    hidden_features: int
    out_features: int = None
    drop: float = 0.
    activation: Callable = nn.gelu
    dtype: Any = jnp.float32
    kernel_init: Callable = default_init

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        out_features = self.out_features or self.hidden_features
        hidden_feature = self.hidden_features

        outputs = inputs.astype(self.dtype)
        outputs = nn.Dense(hidden_feature, kernel_init=self.kernel_init)(outputs)
        outputs = self.activation(outputs)
        outputs = nn.Dropout(self.drop)(outputs, deterministic=train)
        outputs = nn.Dense(out_features, kernel_init=self.kernel_init)(outputs)
        outputs = nn.Dropout(self.drop)(outputs, deterministic=train)
        return outputs
