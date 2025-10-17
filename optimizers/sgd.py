import numpy as np
from tinygrad.tensor import Tensor


class SGDOptimizer:
    def __init__(self, parameters, lr=3e-4):
        self.parameters = [p for p in parameters if p.numel() > 0]
        self.lr = lr

    def step(self, debug = False):
        if debug:
            import pdb
            pdb.set_trace()

        # params_updated = 0
        # total_params = len(self.parameters)
        for param in self.parameters:
            if hasattr(param, "grad") and param.grad is not None:
                param.assign(param - self.lr * param.grad)
                # params_updated += 1
        # print(f"SGD Step: Updated {params_updated}/{total_params} parameters.")

    def zero_grad(self, debug = False):
        if debug:
            import pdb
            pdb.set_trace()

        for param in self.parameters:
            if hasattr(param, "grad"):
                param.grad = Tensor.zeros_like(param)
            else:
                print(f"Warning: Parameter {param} has no grad to zero.")
