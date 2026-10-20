"""Wiring for the shared brain: the real region and its controls.

An arm is a directed edge list. Edges run pre -> post; the brain computes post
activity from pre activity, so the sparse matrix is indexed [post, pre], and
edges are kept sorted by (post, pre), the order torch calls coalesced. Every
arm except `dense` uses the region's N neurons and edge count; `dense` is a
fully connected brain sized so its weights plus biases match the region's.
"""
import hashlib
import math
from dataclasses import dataclass

import numpy as np

ARMS = ("real", "shuf", "er", "real_randio", "dense")
K_IO = 256
_ARM_TAG = {"real": 0, "shuf": 1, "er": 2, "real_randio": 3, "dense": 4}


@dataclass(frozen=True, eq=False)
class Wiring:
    arm: str
    seed: int
    n: int
    pre: np.ndarray
    post: np.ndarray
    inp: np.ndarray
    out: np.ndarray

    @property
    def nnz(self):
        return int(self.pre.size)

    def fingerprint(self):
        h = hashlib.sha256(f"{self.arm}|{self.n}|".encode())
        for a in (self.pre, self.post, self.inp, self.out):
            h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
        return h.hexdigest()


def bad_edges(n, pre, post):
    """True for every self-loop and every repeat of an earlier (pre, post) pair."""
    key = pre.astype(np.int64) * n + post
    order = np.argsort(key, kind="stable")
    s = key[order]
    dup = np.zeros(key.size, bool)
    dup[order[1:]] = s[1:] == s[:-1]
    return dup | (pre == post)


def load_region(path):
    z = np.load(path)
    n = int(len(z["root_id"]))
    pre, post = z["pre"].astype(np.int64), z["post"].astype(np.int64)
    if pre.min() < 0 or max(pre.max(), post.max()) >= n:
        raise ValueError(f"{path}: edge index outside 0..{n - 1}")
    if bad_edges(n, pre, post).any():
        raise ValueError(f"{path}: region has self-loops or duplicate edges")
    return n, pre, post


def degree_preserving_shuffle(n, pre, post, rng, max_passes=20, tries=1000):
    """Permute post ends across edges, so every neuron keeps its in- and out-degree.

    The permutation leaves some self-loops and duplicate edges. Each is then
    fixed by a targeted swap with a random edge, (a->b, c->d) to (a->d, c->b),
    accepted only when both new edges are new and not self-loops (Maslov and
    Sneppen 2002). Random re-permutation does not converge on the mushroom
    body: its hubs take input from half of all neurons, so it plateaus near
    8,000 conflicts.
    """
    post = rng.permutation(post)
    for _ in range(max_passes):
        bad = np.flatnonzero(bad_edges(n, pre, post))
        if bad.size == 0:
            return post
        keys, counts = np.unique(pre * n + post, return_counts=True)
        count = dict(zip(keys.tolist(), counts.tolist()))
        for i in rng.permutation(bad).tolist():
            a, b = int(pre[i]), int(post[i])
            if a != b and count[a * n + b] == 1:
                continue                    # an earlier swap already fixed it
            for j in rng.integers(0, post.size, size=tries).tolist():
                c, d = int(pre[j]), int(post[j])
                ad, cb = a * n + d, c * n + b
                if a == d or c == b or ad == cb or count.get(ad, 0) or count.get(cb, 0):
                    continue
                for k in (a * n + b, c * n + d):
                    count[k] -= 1
                count[ad], count[cb] = 1, 1
                post[i], post[j] = d, b
                break
    raise RuntimeError(f"shuffle still has conflicts after {max_passes} passes")


def erdos_renyi(n, m, rng):
    """m distinct directed edges drawn uniformly, no self-loops."""
    if m > n * (n - 1):
        raise ValueError(f"{m} edges do not fit in {n} neurons")
    keys = np.empty(0, np.int64)
    while keys.size < m:
        k = rng.integers(0, n * n, size=2 * (m - keys.size), dtype=np.int64)
        keys = np.union1d(keys, k[k // n != k % n])
    keys = rng.choice(keys, size=m, replace=False)
    return keys // n, keys % n


def select_io(n, post, k, rng=None):
    """Input = the k lowest in-degree neurons, readout = the k highest.

    This copies the fly-criticality E5/E6 setup where real wiring beat its
    shuffles. With rng, both sets are drawn at random (the real_randio arm).
    """
    if 2 * k > n:
        raise ValueError(f"need {2 * k} distinct neurons, region has {n}")
    if rng is not None:
        pick = rng.choice(n, size=2 * k, replace=False)
        return np.sort(pick[:k]), np.sort(pick[k:])
    order = np.argsort(np.bincount(post, minlength=n), kind="stable")
    return np.sort(order[:k]), np.sort(order[::-1][:k])


def dense_size(n, nnz):
    """Neurons in a fully connected brain whose n*(n-1) weights + n biases is
    closest to the region's nnz weights + n biases."""
    return round(math.sqrt(nnz + n))


def make_wiring(region_path, arm, seed, k=K_IO):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    n, pre, post = load_region(region_path)
    rng = np.random.default_rng([seed, _ARM_TAG[arm]])
    if arm == "dense":
        n = dense_size(n, pre.size)
        pre, post = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n)))
        keep = pre != post
        pre, post = pre[keep], post[keep]
    if arm in ("real", "dense"):
        seed = 0            # these wirings are the same at every seed
    elif arm == "shuf":
        post = degree_preserving_shuffle(n, pre, post, rng)
    elif arm == "er":
        pre, post = erdos_renyi(n, pre.size, rng)
    inp, out = select_io(n, post, k, rng if arm == "real_randio" else None)
    order = np.lexsort((pre, post))
    return Wiring(arm, seed, n, pre[order], post[order], inp, out)


def forward_reach(w, hops):
    """Neurons reachable from an input neuron in at most `hops` edges."""
    seen = np.zeros(w.n, bool)
    seen[w.inp] = True
    for _ in range(hops):
        seen[w.post[seen[w.pre]]] = True
    return seen


def edge_usage(w, steps):
    """Fraction of edges that get a gradient on the first step, biases at zero.

    Only neurons that input activity has reached are non-zero, and the last
    of the steps-1 sparse passes reads the state after steps-2 hops. Because
    the block normalises over all neurons every step, every lit neuron affects
    the output, so the source neuron being lit is the whole condition.
    """
    return float(forward_reach(w, steps - 2)[w.pre].mean())


def readout_reach(w, steps):
    """Fraction of readout neurons that input activity reaches in steps-1 hops."""
    return float(forward_reach(w, steps - 1)[w.out].mean())
