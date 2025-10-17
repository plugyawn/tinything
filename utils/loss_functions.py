import numpy as np
from tinygrad.tensor import Tensor


def cross_entropy(logits, targets):
    """
    Compute the cross-entropy loss between logits and targets.
    Args:
        logits: Predicted logits from the model (batch_size, num_classes).
        targets: True class indices (batch_size,).
    """

    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    targets_flat = Tensor(targets.reshape(B * T))
    log_probs = logits_flat - logits_flat.logsumexp(axis=-1).unsqueeze(-1)
    loss = -log_probs[Tensor(np.arange(B * T)), targets_flat].mean()
    return loss
