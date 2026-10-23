"""How alike two words are, judged from the training text alone.

Each pool word gets a vector of positive PMI with the tokens that sit within
`window` places of it; similarity is the cosine of those vectors. It uses no
model, so "look-alike" pairs are the same for every arm.
"""
from pathlib import Path

import numpy as np


def context_similarity(tokens, pool, vocab, window=2):
    """(len(pool), len(pool)) cosine similarity of PPMI context vectors."""
    slot = np.full(vocab, -1, np.int64)
    slot[pool] = np.arange(len(pool))
    where = slot[tokens]
    counts = np.zeros(len(pool) * vocab, np.int64)
    for off in [o for o in range(-window, window + 1) if o]:
        lo, hi = max(0, -off), len(tokens) - max(0, off)
        w = where[lo:hi]
        hit = w >= 0
        nbr = tokens[lo + off:hi + off][hit].astype(np.int64)
        counts += np.bincount(w[hit] * vocab + nbr, minlength=counts.size)
    c = counts.reshape(len(pool), vocab).astype(np.float64)
    total = c.sum()
    if total == 0:
        raise ValueError("no pool word occurs in the token stream")
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(c * total / (c.sum(1, keepdims=True) * c.sum(0, keepdims=True)))
    v = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0.0)
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    v = np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)
    return v @ v.T


def cached_similarity(tokens_path, pool, vocab, cache):
    """context_similarity over a token file, cached next to it. The cache is
    keyed by the pool so a different pool never reuses it."""
    cache = Path(cache)
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["pool"], pool):
            return z["sim"]
    sim = context_similarity(np.memmap(tokens_path, dtype=np.uint16, mode="r"), pool, vocab)
    np.savez(cache, pool=pool, sim=sim)
    return sim
