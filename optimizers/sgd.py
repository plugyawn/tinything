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

class AdamOptimizer:
    def __init__(self, parameters, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay = 0.0, amsgrad = False, decoupled_wd = False):
        self.parameters = [p for p in parameters if p.numel() > 0]
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.amsgrad = bool(amsgrad)
        self.decoupled_wd = bool(decoupled_wd)
        self.moments1 = {id(p): Tensor.zeros_like(p) for p in self.parameters}
        self.moments2 = {id(p): Tensor.zeros_like(p) for p in self.parameters}
        self.timestep = 0

        self._state = {}
        self._t = 0
    
    def _ensure_state(self, p):
        pid = id(p)
        if pid not in self._state:
            m = Tensor.zeros_like(p)
            v = Tensor.zeros_like(p)
            state = {"m": m, "v": v}
            if self.amsgrad:
                state["vhat"] = Tensor.zeros_like(p)
            self._state[pid] = state

    def step(self, debug=False):
        if debug:
            import pdb; pdb.set_trace()

        any_updated = False
        self._t += 1  # increment step once per optimizer step

        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - (b1 ** self._t)  # bias correction denominators
        bc2 = 1.0 - (b2 ** self._t)

        for p in self.parameters:
            # skip if no grad this step
            if not hasattr(p, "grad") or p.grad is None:
                continue

            self._ensure_state(p)
            st = self._state[id(p)]

            # Gradient with (optional) L2 penalty:
            # - If decoupled_weight_decay=False -> classical Adam L2 (coupled): g = g + wd * p
            # - If decoupled_weight_decay=True  -> AdamW: apply weight decay directly to params
            g = p.grad
            if self.weight_decay > 0.0 and not self.decoupled_wd:
                g = g + self.weight_decay * p

            # Update first/second moments
            st["m"] = b1 * st["m"] + (1.0 - b1) * g
            st["v"] = b2 * st["v"] + (1.0 - b2) * (g * g)

            if self.amsgrad:
                # vhat = max(vhat, v)
                st["vhat"] = Tensor.maximum(st["vhat"], st["v"])
                v_used = st["vhat"]
            else:
                v_used = st["v"]

            # Bias-corrected moments
            m_hat = st["m"] / bc1
            v_hat = v_used / bc2

            # Parameter update
            denom = Tensor.sqrt(v_hat) + self.eps
            update = self.lr * (m_hat / denom)

            if self.decoupled_wd and self.weight_decay > 0.0:
                # AdamW-style decoupled decay
                p.assign(p - self.lr * self.weight_decay * p)

            p.assign(p - update)
            any_updated = True

        # (Optional) you could return whether anything was updated
        return any_updated

    def zero_grad(self, set_to_none=False, debug=False):
        if debug:
            import pdb; pdb.set_trace()

        for p in self.parameters:
            if hasattr(p, "grad") and p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad = Tensor.zeros_like(p)