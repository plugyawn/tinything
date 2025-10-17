import numpy as np
from tinygrad.tensor import Tensor


class SGDOptimizer:
    def __init__(self, parameters, lr=3e-4):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if hasattr(param, "grad") and param.grad is not None:
                param.data -= self.lr * param.grad.data

    def zero_grad(self):
        for param in self.parameters:
            if hasattr(param, "grad") and param.grad is not None:
                param.grad = Tensor.zeros_like(param)
