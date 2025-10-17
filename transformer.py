import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, Embedding, LayerNorm

from tqdm import tqdm

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

    def __call__(self, x):
        mean = x.mean(axis=-1)
        mean = mean.unsqueeze(-1)
        variance = ((x - mean) ** 2).mean(axis=-1)
        variance = variance.unsqueeze(-1)
        x_normalized = (x - mean) / (variance + self.eps).sqrt()
        return x_normalized * self.weight + self.bias
    
class CausalSelfAttention:
    def __init__(self, n_embed, n_head, block_size):
        self.n_head = n_head
        self.block_size = block_size
        self.q_proj = Linear(n_embed, n_embed)
        self.k_proj = Linear(n_embed, n_embed)
        self.v_proj = Linear(n_embed, n_embed)
        self.out_proj = Linear(n_embed, n_embed)
    
    def __call__(self, x, debug = False):
        if debug: import pdb; pdb.set_trace()
        B, T, C = x.shape
        H = self.n_head
        assert C % H == 0, "Embedding dimension must be divisible by number of heads"
        head_size = C // H 
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, T, H, head_size).permute(0, 2, 1, 3)  # split heads, permute
        k = k.reshape(B, T, H, head_size).permute(0, 2, 1, 3) # permute for matmul
        q = apply_rope(q)
        k = apply_rope(k)
        k = k.permute(0, 1, 3, 2)
        v = v.reshape(B, T, H, head_size).permute(0, 2, 1, 3)

        att = (q@k) / (C // self.n_head) ** 0.5  # scaled dot-product attention
        mask = Tensor.ones(T, T).tril() # tri - triangular, l - lower
        att = att.masked_fill(mask == 0, float('-inf'))
        att = att.softmax(-1)
        y = att @ v  # attention output

        # reassemble all head outputs side by side
        y = y.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(y)

class Embedding:
    """
    Embedding layer.
    Maps discrete tokens to continuous vectors.
    """
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor.randn(self.num_embeddings, self.embedding_dim) / np.sqrt(self.embedding_dim)

    def __call__(self, x):
        # x is a Tensor of indices, use Tensor indexing to get embeddings
        import pdb
        # pdb.set_trace()
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return self.weight[x]
        
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

def swish(x):
    return x * x.sigmoid()

class SwiGLU:
    def __init__(self, dim, hidden_dim):
        self.w1 = Linear(dim, hidden_dim)
        self.w2 = Linear(dim, hidden_dim)
        self.w3 = Linear(hidden_dim, dim)
    def __call__(self, x):
        return self.w3(self.w1(x) * swish(self.w2(x)))
    
class GPT:
    """
    GPT model class with simple pos embs.
    """
    def __init__(self, vocab_size, n_embed, n_head, n_layer, block_size):
        self.token_embedding = Embedding(vocab_size, n_embed)
        self.position_embedding = Embedding(block_size, n_embed)
        self.layers = n_layer
        # self.layers = [CausalSelfAttention(n_embed, n_head, block_size) for _ in range(n_layer)]
        self.ln_f = LayerNorm(n_embed)
        self.head = Linear(n_embed, vocab_size)

        self.blocks = [
            (
                LayerNorm(n_embed),
                CausalSelfAttention(n_embed, n_head, block_size),
                SwiGLU(n_embed, 4 * n_embed), # feed-forward expansion
                # SwiGLU(4 * n_embed, n_embed) # feed-forward projection
            )
            for _ in range(self.layers)
        ]

    def __call__(self, idx, debug = False):
        if debug:
            import pdb
            pdb.set_trace()
        B, T = idx.shape # batch size, sequence length
        tok_emb = self.token_embedding(idx) # token embeddings
        # pos = Tensor.arange(T) # simple, linear positions
        # pos_emb = self.position_embedding(pos) # position embeddings
        x = tok_emb # combine token and position embeddings

        for ln, attn, swiglu in self.blocks:
            x = x + attn(ln(x), debug = False) # attention block
            x = x + swiglu((ln(x))) # feed-forward block
        x = self.ln_f(x) # final layer norm
        logits = self.head(x) # output logits
        return logits
    
from utils import load_data
from loss_functions import cross_entropy

def train(model, train_data, get_batch, optimizer, epochs = 10, block_size=128, batch_size=32):
    print(generate(model, Tensor(np.array([[0]], dtype=np.int32)), max_new_tokens=10, block_size=block_size))
    for epoch in range(epochs):
        losses = []
        for _ in tqdm(range(len(train_data) // (batch_size * block_size)), desc = f"Epoch {epoch+1}/{epochs}"):
            x_batch, y_batch = get_batch(train_data, block_size, batch_size)
            logits = model(x_batch)
            loss = cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            losses.append(loss.item())
            # print(loss.item())
        print(f"Epoch {epoch+1} Loss: {np.mean(losses)}")
        print(generate(model, Tensor(np.array([[0]], dtype=np.int32)), max_new_tokens=10, block_size=block_size))

class SGDOptimizer:
    def __init__(self, parameters, lr=3e-4):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if hasattr(param, 'grad') and param.grad is not None:
                param.data -= self.lr * param.grad.data

    def zero_grad(self):
        for param in self.parameters:
            if hasattr(param, 'grad') and param.grad is not None:
                param.grad = Tensor.zeros_like(param)
    


def generate(model, idx, max_new_tokens, block_size):
    """
    Generate new tokens from the model given a starting sequence idx.
    Args:
        model: The trained GPT model.
        idx: Tensor of shape (1, T) containing the starting token indices.
        max_new_tokens: Number of tokens to generate.
        block_size: The context size of the model.
    """

    for _ in (range(max_new_tokens)):
        idx_cond = idx[:, -block_size:]  # crop context to block_size
        logits = model(idx_cond, debug = False)  # (1, T, vocab_size)
        logits = logits[:, -1, :]  # (1, vocab_size), get logits for the last token
        probs = logits.softmax(-1)  # apply softmax to get probabilities
        next_token = probs.multinomial(num_samples=1)  # sample from the distribution
        idx = Tensor(np.concatenate([idx.numpy(), next_token.numpy()], axis=1))  # append sampled token

        print(itos[int(next_token.numpy()[0,0])], end='', flush=True)

if __name__ == "__main__":
    block_size = 128
    batch_size = 32
    n_embed = 64
    n_head = 4
    n_layer = 3
    epochs = 10

    get_batch, train_data, val_data, stoi, itos, vocab_size = load_data("input.txt", block_size, batch_size)

    model = GPT(vocab_size, n_embed, n_head, n_layer, block_size)
    optimizer = SGDOptimizer([param for layer in model.blocks for param in layer] + 
                             [model.token_embedding.weight, model.position_embedding.weight,
                              model.ln_f.weight, model.ln_f.bias,
                              model.head.weight, model.head.bias], lr=1e-3)

    train(model, train_data, get_batch, optimizer, epochs, block_size, batch_size)
