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
        self.weight.requires_grad = True

        if bias:
            self.bias = Tensor.zeros(out_features) if bias else None
            self.bias.requires_grad = True

    def parameters(self, from_swiglu = False):
        yield self.weight # Not list because we dont want to eager array this. 
        if self.bias is not None:
            yield self.bias

    def __call__(self, x, debug = False):
        if debug:
            import pdb
            pdb.set_trace()

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
    
    def parameters(self):
        yield from self.w1.parameters(from_swiglu=True)
        yield from self.w2.parameters(from_swiglu=True)
        yield from self.w3.parameters(from_swiglu=True)

    def __call__(self, x):
        return self.w3(self.w1(x) * swish(self.w2(x)))
