"""Planning statistics for an n-gram memory: distinct n-grams in the training
tokens, and how often a validation n-gram was seen in training.

For each n, a validation position t uses the n-gram ending at t (the memory key
for predicting token t+1). Coverage buckets count those keys by their training
frequency. Usage: .venv/bin/python scripts/ngram_stats.py [max_train_tokens]
Results: notes/ngram_stats_2026-11-02.md
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VOCAB_BITS = 13  # 8,192 tokens
BUCKETS = (0, 1, 10, 100, 1000)


def keys(tokens, n):
    t = tokens.astype(np.uint64)
    k = t[: len(t) - n + 1].copy()
    for i in range(1, n):
        k = (k << np.uint64(VOCAB_BITS)) | t[i: len(t) - n + 1 + i]
    return k


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    train = np.fromfile(ROOT / "data/fineweb/train.bin", dtype=np.uint16)[:limit]
    val = np.fromfile(ROOT / "data/fineweb/val.bin", dtype=np.uint16)
    if train.max() >= 2 ** VOCAB_BITS or val.max() >= 2 ** VOCAB_BITS:
        raise SystemExit("token id above the 13-bit vocabulary; adjust VOCAB_BITS")
    print(f"train tokens {len(train):,}, val tokens {len(val):,}")
    for n in (2, 3, 4):
        uniq, counts = np.unique(keys(train, n), return_counts=True)
        vk = keys(val, n)
        pos = np.searchsorted(uniq, vk)
        pos[pos == len(uniq)] = 0
        freq = np.where(uniq[pos] == vk, counts[pos], 0)
        edges = list(BUCKETS) + [np.inf]
        shares = [np.mean((freq >= lo) & (freq < hi)) if lo else np.mean(freq == 0)
                  for lo, hi in zip(edges[:-1], edges[1:])]
        labels = ["unseen", "1-9", "10-99", "100-999", ">=1000"]
        once = np.mean(counts == 1)
        print(f"n={n}: distinct {len(uniq):,}; seen once {once:.1%} of distinct; val keys by train count: "
              + ", ".join(f"{lab} {s:.1%}" for lab, s in zip(labels, shares)), flush=True)


if __name__ == "__main__":
    main()
