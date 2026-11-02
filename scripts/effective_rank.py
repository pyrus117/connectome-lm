"""Exploratory, post-Stage-1: effective rank of the brain's recurrent matrix per arm.

Effective rank = participation ratio of the singular values, (sum s^2)^2 / sum s^4.
"topology" weights each edge by its fan-in init scale (1/sqrt(in-degree)), so it
measures the wiring alone; "trained" uses the checkpoint's learned synapse values.
Usage: .venv/bin/python scripts/effective_rank.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from clm.graphs import make_wiring  # noqa: E402

REGION = ROOT / "data" / "mb_alpn_783.npz"
RUNS = [("real", 0), ("shuf", 0), ("shuf", 1), ("shuf", 2), ("er", 0), ("er", 2)]


def participation_ratio(m):
    s2 = torch.linalg.svdvals(m).pow(2)
    return float(s2.sum() ** 2 / s2.pow(2).sum())


def dense(w, vals):
    m = torch.zeros(w.n, w.n, dtype=torch.float64)
    m[torch.from_numpy(w.post), torch.from_numpy(w.pre)] = vals.double()
    return m


def main():
    for arm, seed in RUNS:
        w = make_wiring(REGION, arm, seed)
        ckpt = torch.load(ROOT / "runs" / f"{arm}-{seed}" / "ckpt.pt", map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)
        keys = [k for k in sd if k.endswith("brain.values")]
        if len(keys) != 1:
            raise SystemExit(f"{arm}-{seed}: expected one brain.values key, found {keys}")
        indeg = np.bincount(w.post, minlength=w.n)
        topo = torch.from_numpy(1.0 / np.sqrt(indeg[w.post]))
        print(f"{arm}-{seed}: n={w.n} nnz={w.nnz} "
              f"PR_topology={participation_ratio(dense(w, topo)):.1f} "
              f"PR_trained={participation_ratio(dense(w, sd[keys[0]])):.1f}", flush=True)


if __name__ == "__main__":
    main()
