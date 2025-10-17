from itertools import chain 

import numpy as np
from tinygrad.tensor import Tensor
from tqdm import tqdm

from layers.attention import CausalSelfAttention
from layers.feedforward import Linear, SwiGLU
from layers.lookup import Embedding
from layers.norm import LayerNorm
from optimizers.sgd import SGDOptimizer, AdamOptimizer
from utils.dataloader import load_data
from utils.loss_functions import cross_entropy


class GPT:
    """
    GPT model class with simple pos embs.
    """

    def __init__(self, vocab_size, n_embed, n_head, n_layer, block_size):
        self.token_embedding = Embedding(vocab_size, n_embed)
        self.position_embedding = Embedding(block_size, n_embed)
        self.layers = n_layer
        self.ln_f = LayerNorm(n_embed)
        self.output_head = Linear(n_embed, vocab_size)

        self.blocks = [
            (
                LayerNorm(n_embed),
                CausalSelfAttention(n_embed, n_head, block_size),
                SwiGLU(n_embed, 4 * n_embed),  # feed-forward expansion
            )
            for _ in range(self.layers)
        ]

    def parameters(self):
        for layer in self.blocks:
            for param in layer:
                yield from param.parameters()
        yield from self.token_embedding.parameters()
        yield from self.ln_f.parameters()
        yield from self.output_head.parameters()

    def __call__(self, idx, debug=False):
        if debug:
            import pdb
            pdb.set_trace()

        B, T = idx.shape  # batch size, sequence length
        tok_emb = self.token_embedding(idx)  # token embeddings
        x = tok_emb  # setup residual stream

        for ln, attn, swiglu in self.blocks:
            x = x + attn(ln(x), debug=False)  # attention block
            x = x + swiglu((ln(x)))  # feed-forward block
        x = self.ln_f(x)  # final layer norm
        logits = self.output_head(x)  # output logits
        return logits


def train(
    model, train_data, get_batch, optimizer, epochs=10, block_size=128, batch_size=32
):
    for epoch in range(epochs):
        losses = []

        for _ in tqdm(
            range(len(train_data) // (batch_size * block_size)),
            desc=f"Epoch {epoch+1}/{epochs}",
        ):
            
            x_batch, y_batch = get_batch(train_data, block_size, batch_size)
            logits = model(x_batch)
            loss = cross_entropy(logits, y_batch).mean()
            loss.backward()
            optimizer.step(debug = False)

            optimizer.zero_grad()

            losses.append(loss.item())
            # print(loss.item())
        print(f"Epoch {epoch+1} Loss: {np.mean(losses)}")
        print(
            generate(
                model,
                Tensor(np.array([[0]], dtype=np.int32)),
                max_new_tokens=10,
                block_size=block_size,
            )
        )


def generate(model, idx, max_new_tokens, block_size):
    """
    Generate new tokens from the model given a starting sequence idx.
    """

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]  # crop context to block_size
        logits = model(idx_cond, debug=False)  # (1, T, vocab_size)
        logits = logits[:, -1, :]  # (1, vocab_size), get logits for the last token
        probs = logits.softmax(-1)  # apply softmax to get probabilities
        next_token = probs.multinomial(num_samples=1)  # sample from the distribution
        idx = Tensor(
            np.concatenate([idx.numpy(), next_token.numpy()], axis=1)
        )  # append sampled token

        print(itos[int(next_token.numpy()[0, 0])], end="", flush=True)


if __name__ == "__main__":
    block_size = 128
    batch_size = 32
    n_embed = 32
    n_head = 4
    n_layer = 2
    epochs = 10

    get_batch, train_data, val_data, stoi, itos, vocab_size = load_data(
        "data/alice_in_wonderland.txt", block_size, batch_size
    )

    model = GPT(vocab_size, n_embed, n_head, n_layer, block_size)

    print(next(model.blocks[0][1].parameters()))
    tracked = []
    tracked_iter = []
    tracked += list(chain.from_iterable(model.parameters()))
    
    print("Length comparison:", len(tracked))
    
    import pdb;pdb.set_trace()
    # print(next(layer.parameters() for layer in model.blocks))
    # optimizer = SGDOptimizer(
    #     tracked,
    #     lr=1e-3,
    # )
    optimizer = AdamOptimizer(
        tracked,
        lr=1e-3,
    )

    train(model, train_data, get_batch, optimizer, epochs, block_size, batch_size)
