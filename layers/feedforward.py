import numpy as np
from tinygrad.tensor import Tensor


class Linear:
    """
    Basic linear layer.
    Does Ax + B; A is weights, B is bias.
    Set bias to False to disable bias.
    """

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor.randn(out_features, in_features) / np.sqrt(in_features)
        self.bias = Tensor.zeros(out_features) if bias else None

    def __call__(self, x):
        y = x.dot(self.weight.T)
        if self.bias is not None:
            y += self.bias
        return y


def swish(x):
    return x * x.sigmoid()


class SwiGLU:
    def __init__(self, dim, hidden_dim):
        self.w1 = Linear(dim, hidden_dim)
        self.w2 = Linear(dim, hidden_dim)
        self.w3 = Linear(hidden_dim, dim)

    def __call__(self, x):
        return self.w3(self.w1(x) * swish(self.w2(x)))
