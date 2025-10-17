import numpy as np
from tinygrad.tensor import Tensor


class Embedding:
    """
    Embedding layer.
    Maps discrete tokens to continuous vectors.
    """

    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor.randn(self.num_embeddings, self.embedding_dim) / np.sqrt(
            self.embedding_dim
        )
        self.weight.requires_grad = True
    
    def parameters(self):
        yield self.weight

    def __call__(self, x, debug = False):
        # x is a Tensor of indices, use Tensor indexing to get embeddings
        if debug:
            import pdb
            pdb.set_trace()

        if not isinstance(x, Tensor):
            x = Tensor(x)
        return self.weight[x]
