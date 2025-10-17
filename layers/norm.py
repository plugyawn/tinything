import numpy as np
from tinygrad.tensor import Tensor


class LayerNorm:
    """
    Layer Normalization layer.
    Normalizes the input across the last dimension.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Tensor.ones(normalized_shape)
        self.bias = Tensor.zeros(normalized_shape)
        self.weight.requires_grad = True
        self.bias.requires_grad = True

    def parameters(self): 
        yield self.weight
        yield self.bias

    def __call__(self, x, debug = False):
        
        if debug:
            import pdb
            pdb.set_trace()

        mean = x.mean(axis=-1)
        mean = mean.unsqueeze(-1)
        variance = ((x - mean) ** 2).mean(axis=-1)
        variance = variance.unsqueeze(-1)
        x_normalized = (x - mean) / (variance + self.eps).sqrt()
        return x_normalized * self.weight + self.bias
