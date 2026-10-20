from pathlib import Path

import numpy as np
import pytest

REAL_REGION = Path(__file__).resolve().parent.parent / "data" / "mb_alpn_783.npz"


def hubby_graph(n=120, m=600, seed=0):
    """A small directed graph with a heavy-tailed in-degree, like a connectome."""
    rng = np.random.default_rng(seed)
    weight = 1.0 / np.arange(1, n + 1) ** 0.8      # neuron 0 is the biggest hub
    p = weight / weight.sum()
    keys = set()
    while len(keys) < m:
        pre, post = int(rng.integers(n)), int(rng.choice(n, p=p))
        if pre != post:
            keys.add((pre, post))
    pre, post = np.array(sorted(keys)).T
    return n, pre, post


@pytest.fixture
def region(tmp_path):
    n, pre, post = hubby_graph()
    path = tmp_path / "region.npz"
    np.savez(path, root_id=np.arange(n), pre=pre, post=post, w=np.ones(pre.size, np.float32))
    return path
