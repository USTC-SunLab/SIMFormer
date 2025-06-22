import jax.numpy as jnp
from jax import dtypes
import collections.abc
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)


def constant_init(key, shape, dtype=jnp.float_, constant=0.04):
    return jnp.ones(shape, dtypes.canonicalize_dtype(dtype)) * constant
