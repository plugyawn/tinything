import numpy as np
from tinygrad.tensor import Tensor


def apply_rope(x):
    """
    x: (B, n_head, T, head_dim)
    """
    B, n_head, T, head_dim = x.shape
    half_dim = head_dim // 2
    theta = 10000 ** (-np.arange(0, half_dim, dtype=np.float32) / np.float32(half_dim))
    pos = Tensor.arange(T).reshape(T, 1)
    freqs = Tensor(theta).reshape(1, half_dim)
    angles = pos * freqs  # (T, half_dim)
    sin = angles.sin()
    cos = angles.cos()

    sin = sin.reshape(1, 1, T, half_dim)
    cos = cos.reshape(1, 1, T, half_dim)

    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    x_rot = Tensor.cat(x1 * cos - x2 * sin, x1 * sin + x2 * cos, dim=-1)
    return x_rot
