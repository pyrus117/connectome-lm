"""Token files and batch sampling.

Batch contents depend only on (seed, step), never on the arm or on where a
run resumed, so every arm at a given seed sees the same tokens in the same
order. The train file is cut into non-overlapping windows and read once, in
a seeded random order: no window is seen twice.
"""
from functools import lru_cache

import numpy as np
import torch


def open_tokens(path):
    return np.memmap(path, dtype=np.uint16, mode="r")


@lru_cache(maxsize=4)
def _order(windows, seed):
    return np.random.default_rng(seed).permutation(windows)


def batch_starts(n_tokens, ctx, batch, seed, step):
    order = _order((n_tokens - 1) // ctx, seed)
    sel = order[step * batch:(step + 1) * batch]
    if sel.size < batch:
        raise ValueError(f"step {step} runs past one epoch of the train file")
    return sel.astype(np.int64) * ctx


def get_batch(tokens, starts, ctx, device):
    x = np.stack([tokens[s:s + ctx] for s in starts]).astype(np.int64)
    y = np.stack([tokens[s + 1:s + ctx + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def eval_windows(n_tokens, ctx, n_eval):
    """Start offsets of the fixed, non-overlapping validation windows."""
    return np.arange(min(n_eval, n_tokens - 1) // ctx, dtype=np.int64) * ctx
