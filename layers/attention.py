import numpy as np
from tinygrad.tensor import Tensor

from layers.feedforward import Linear, SwiGLU
from utils.transformer_methods import apply_rope


class CausalSelfAttention:
    def __init__(self, n_embed, n_head, block_size):
        self.n_head = n_head
        self.block_size = block_size
        self.q_proj = Linear(n_embed, n_embed)
        self.k_proj = Linear(n_embed, n_embed)
        self.v_proj = Linear(n_embed, n_embed)
        self.out_proj = Linear(n_embed, n_embed)

    def parameters(self):
        yield from self.q_proj.parameters()
        yield from self.k_proj.parameters()
        yield from self.v_proj.parameters()
        yield from self.out_proj.parameters()

    def __call__(self, x, debug=False):
        if debug:
            import pdb

            pdb.set_trace()
        B, T, C = x.shape
        H = self.n_head
        assert C % H == 0, "Embedding dimension must be divisible by number of heads"
        head_size = C // H
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, T, H, head_size).permute(0, 2, 1, 3)  # split heads, permute
        k = k.reshape(B, T, H, head_size).permute(0, 2, 1, 3)
        q = apply_rope(q)
        k = apply_rope(k)
        k = k.permute(0, 1, 3, 2)
        v = v.reshape(B, T, H, head_size).permute(0, 2, 1, 3)  # permute for matmul

        att = (q @ k) / (C // self.n_head) ** 0.5  # scaled dot-product attention
        mask = Tensor.ones(T, T).tril()  # tri - triangular, l - lower
        att = att.masked_fill(mask == 0, float("-inf"))
        att = att.softmax(-1)
        y = att @ v  # attention output

        # reassemble all head outputs side by side
        y = y.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(y)
