import os
import numpy as np
import json
import collections
import platform
import random
import warnings
from tifffile import imwrite


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    # Percentile-based normalization

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x


def normalize_minmse(x, target):
    # Minimize MSE rescaling
    cov = np.cov(x.flatten(),target.flatten())
    alpha = cov[0,1] / (cov[0,0]+1e-10)
    beta = target.mean() - alpha*x.mean()
    return alpha*x + beta


###


def _raise(e):
    if isinstance(e, BaseException):
        raise e
    else:
        raise ValueError(e)


def consume(iterator):
    # Consume iterator
    collections.deque(iterator, maxlen=0)


def compose(*funcs):
    return lambda x: reduce(lambda f,g: g(f), funcs, x)



def axes_check_and_normalize(axes,length=None,disallowed=None,return_allowed=False):
    # Check and normalize axes string
    allowed = 'STCZYX'
    axes is not None or _raise(ValueError('axis cannot be None.'))
    axes = str(axes).upper()
    consume(a in allowed or _raise(ValueError("invalid axis '%s', must be one of %s."%(a,list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(ValueError("disallowed axis '%s'."%a)) for a in axes)
    consume(axes.count(a)==1 or _raise(ValueError("axis '%s' occurs more than once."%a)) for a in axes)
    length is None or len(axes)==length or _raise(ValueError('axes (%s) must be of length %d.' % (axes,length)))
    return (axes,allowed) if return_allowed else axes


def axes_dict(axes):
    # Convert axes string to dict
    axes, allowed = axes_check_and_normalize(axes,return_allowed=True)
    return { a: None if axes.find(a) == -1 else axes.find(a) for a in allowed }


def move_image_axes(x, fr, to, adjust_singletons=False):
    # Move image axes
    fr = axes_check_and_normalize(fr, length=x.ndim)
    to = axes_check_and_normalize(to)

    fr_initial = fr
    x_shape_initial = x.shape
    adjust_singletons = bool(adjust_singletons)
    if adjust_singletons:
        slices = [slice(None) for _ in x.shape]
        for i,a in enumerate(fr):
            if (a not in to) and (x.shape[i]==1):
                slices[i] = 0
                fr = fr.replace(a,'')
        x = x[tuple(slices)]
        for i,a in enumerate(to):
            if (a not in fr):
                x = np.expand_dims(x,-1)
                fr += a

    if set(fr) != set(to):
        _adjusted = '(adjusted to %s and %s) ' % (x.shape, fr) if adjust_singletons else ''
        raise ValueError(
            'image with shape %s and axes %s %snot compatible with target axes %s.'
            % (x_shape_initial, fr_initial, _adjusted, to)
        )

    ax_from, ax_to = axes_dict(fr), axes_dict(to)
    if fr == to:
        return x
    return np.moveaxis(x, [ax_from[a] for a in fr], [ax_to[a] for a in fr])


###


def choice(population, k=1, replace=True):
    ver = platform.sys.version_info
    if replace and (ver.major,ver.minor) in [(2,7),(3,5)]:
        return list(np.random.choice(population, k, replace=replace))
    else:
        try:
            state = random.getstate()
            random.seed(np.random.randint(np.iinfo(int).min, np.iinfo(int).max))
            if replace:
                return random.choices(population, k=k)
            else:
                return random.sample(population, k=k)
        finally:
            random.setstate(state)

def save_tiff_imagej_compatible(file, img, axes, **imsave_kwargs):
    # Save ImageJ-compatible TIFF
    img = img.astype(np.float32)
    axes = axes_check_and_normalize(axes,img.ndim,disallowed='S')

    t = img.dtype
    if   'float' in t.name: t_new = np.float32
    elif 'uint'  in t.name: t_new = np.uint16 if t.itemsize >= 2 else np.uint8
    elif 'int'   in t.name: t_new = np.int16
    else:                   t_new = t
    img = img.astype(t_new)
    if t != t_new:
        warnings.warn("Converting data type from '%s' to ImageJ-compatible '%s'." % (t, np.dtype(t_new)))

    img = move_image_axes(img, axes, 'TZCYX', True)

    imsave_kwargs['imagej'] = True
    imwrite(file, img, **imsave_kwargs)