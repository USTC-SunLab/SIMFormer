# Positional embeddings implementation
import jax
import jax.numpy as jnp


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, expand_first_dim=False,
                            dtype=jnp.float32):
    grid_h = jnp.arange(grid_size, dtype=dtype)
    grid_w = jnp.arange(grid_size, dtype=dtype)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = jnp.concatenate([jnp.zeros([1, embed_dim]), pos_embed], axis=0)
    if expand_first_dim:
        pos_embed = pos_embed[jnp.newaxis, Ellipsis]
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = jnp.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    # Generate 1D sincos positional embedding
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = jnp.einsum('m,d->md', pos, omega)

    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)

        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                            embedding_size).transpose((0, 3, 1, 2))
