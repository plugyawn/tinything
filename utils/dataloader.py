# mostly data processing utilities
import numpy as np
from tqdm import tqdm


def load_data(filename, block_size, batch_size, split_ratio=0.9):
    try:
        with open(f"data/{filename}", "r") as file:
            data = file.read()
    except FileNotFoundError:
        try:
            with open(filename, "r") as file:
                data = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File {filename} not found in 'data/' directory or current directory."
            )

    chars = sorted(list(set(data)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encoded = [stoi[c] for c in data]

    split = int(0.9 * len(encoded))
    train_data = encoded[:split]
    val_data = encoded[split:]

    def get_batch(data, block_size, batch_size):
        ix = np.random.randint(0, len(data) - block_size, (batch_size,))
        x = np.array([data[i : i + block_size] for i in ix])
        y = np.array([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y

    vocab_size = len(chars)
    return get_batch, train_data, val_data, stoi, itos, vocab_size
