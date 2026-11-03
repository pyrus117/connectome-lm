"""Kenyon hash: a fixed, similarity-preserving address for token histories (Stage 2B).

Receptor codes are frozen distributional token vectors (PPMI + SVD, or random as
the control). A history's odour is its tokens' codes concatenated, centred and
L2-normalised. A fixed random projection expands it into two sub-codebooks of
`units` Kenyon cells each; the winner(s) of each sub-codebook under per-unit
thresholds name the bucket (Dasgupta, Stevens & Navlakha 2017; Litwin-Kumar et
al. 2017). Because the odour is a concatenation, a cell's input is a sum over
history positions of a per-token table, so a history costs k table lookups and
one add, and the norm is the root of a sum of per-token squared norms.
"""
import json
from pathlib import Path

import numpy as np
import torch

from clm import counts as C

_MASK64 = (1 << 64) - 1


def cooccurrence(tokens, vocab, window=2, eot=None, chunk=1 << 25):
    """Symmetric (vocab, vocab) token co-occurrence counts within `window` places;
    pairs that include `eot` are not counted."""
    forward = np.zeros(vocab * vocab, dtype=np.int64)
    for start in range(0, len(tokens), chunk):
        a = np.asarray(tokens[start:start + chunk + window]).astype(np.int64)
        for off in range(1, window + 1):
            m = min(chunk, a.size - off)
            if m <= 0:
                continue
            x, y = a[:m], a[off:off + m]
            if eot is not None:
                ok = (x != eot) & (y != eot)
                x, y = x[ok], y[ok]
            forward += np.bincount(x * vocab + y, minlength=vocab * vocab)
    f = forward.reshape(vocab, vocab)
    return f + f.T


def ppmi_codes(counts, dim=64, alpha=0.75, seed=0):
    """(vocab, dim) float32 token codes: truncated SVD of the positive PMI matrix with
    context smoothing alpha (Levy, Goldberg & Dagan 2015). A token never seen gets a zero code."""
    c = counts.astype(np.float64)
    total = c.sum()
    row = c.sum(1)
    col = c.sum(0) ** alpha
    col = col / col.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(c / total / (np.outer(row / total, col)))
    ppmi = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0.0).astype(np.float32)
    torch.manual_seed(seed)
    u, s, _ = torch.svd_lowrank(torch.from_numpy(ppmi), q=dim + 16, niter=4)
    return (u[:, :dim] * s[:dim].sqrt()).numpy().astype(np.float32)


def random_codes(vocab, dim=64, seed=0):
    return np.random.default_rng((seed, 0xC0DE)).standard_normal((vocab, dim)).astype(np.float32) / np.sqrt(dim)


def _mix(h):
    """splitmix64 finaliser on a uint64 array."""
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


def hash_bucket(packed, n_buckets):
    """The random-bucket control: a plain hash of the history's token ids."""
    with np.errstate(over="ignore"):
        return (_mix(packed.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)) % np.uint64(n_buckets))


class KenyonHash:
    def __init__(self, codes, k, units=2048, winners=1, in_degree=None, thresholds=True, token_freq=None, seed=0):
        """codes: (vocab, dim). k: history length. in_degree: inputs per cell (None = dense Gaussian).
        thresholds: use the per-cell thresholds (set by `calibrate`); False gives plain winner-take-all."""
        vocab, dim = codes.shape
        self.settings = {"k": k, "units": units, "winners": winners, "in_degree": in_degree,
                         "thresholds": thresholds, "seed": seed}
        self.k, self.units, self.winners, self.use_thresholds = k, units, winners, thresholds
        freq = np.ones(vocab) if token_freq is None else np.asarray(token_freq, dtype=np.float64)
        mean = (codes * (freq / freq.sum())[:, None]).sum(0)
        centred = (codes - mean).astype(np.float32)
        self.norm2 = (centred ** 2).sum(1)
        rng = np.random.default_rng((seed, k, 0xFEED))
        w = rng.standard_normal((2 * units, k * dim)).astype(np.float32)
        if in_degree is not None:
            if not 1 <= in_degree <= k * dim:
                raise ValueError(f"in_degree {in_degree} outside 1..{k * dim}")
            keep = np.zeros_like(w, dtype=bool)
            for u in range(2 * units):
                keep[u, rng.choice(k * dim, in_degree, replace=False)] = True
            w = np.where(keep, w, 0.0).astype(np.float32)
        self.table = np.stack([centred @ w[:, j * dim:(j + 1) * dim].T for j in range(k)])   # (k, vocab, 2 units)
        self._flat = torch.from_numpy(self.table.reshape(k * vocab, 2 * units))      # history position j, token v -> row j * vocab + v
        self._offsets = torch.arange(k, dtype=torch.int64) * vocab
        self._norm = torch.from_numpy(self.norm2)
        self.theta = np.zeros(2 * units, dtype=np.float32)

    def _drive(self, tokens):
        """(N, 2 * units) torch tensor: cell drive of N histories of k tokens, before thresholds."""
        t = torch.from_numpy(np.ascontiguousarray(tokens))
        drive = torch.nn.functional.embedding_bag(t + self._offsets, self._flat, mode="sum")
        if self.use_thresholds:
            drive /= self._norm[t].sum(1).sqrt().clamp_min(1e-6)[:, None]
        return drive

    def activation(self, tokens):
        """(N, 2 * units) cell drive for N histories of k tokens, before thresholds."""
        return self._drive(tokens).numpy()

    def _winners(self, drive, inplace=False):
        """(N, 2, winners) winning cell of each sub-codebook, sorted (numpy int64). inplace: may overwrite `drive`."""
        theta = torch.from_numpy(self.theta)
        d = (drive.sub_(theta) if inplace else drive - theta).view(len(drive), 2, self.units)
        if self.winners == 1:
            return d.argmax(2)[..., None].numpy()
        return d.topk(self.winners, dim=2).indices.sort(dim=2).values.numpy()

    def bucket_keys(self, tokens, chunk=1 << 11):
        """(N,) uint64 bucket id of each history; injective for one winner per sub-codebook."""
        out = np.empty(len(tokens), dtype=np.uint64)
        for s in range(0, len(tokens), chunk):
            win = self._winners(self._drive(tokens[s:s + chunk]), inplace=True).astype(np.uint64)
            if self.winners == 1:
                out[s:s + chunk] = win[:, 0, 0] * np.uint64(self.units) + win[:, 1, 0]
            else:
                h = np.zeros(len(win), dtype=np.uint64)
                with np.errstate(over="ignore"):
                    for i in range(2):
                        for j in range(self.winners):
                            h = _mix(h * np.uint64(0x9E3779B97F4A7C15) + win[:, i, j] + np.uint64(1))
                out[s:s + chunk] = h
        return out

    def calibrate(self, tokens, steps=100, rate=0.03):
        """Homeostatic thresholds: raise the threshold of a cell that wins more often than its
        share and lower it otherwise, until every cell wins about equally often
        (the auxiliary-loss-free bias rule). `tokens`: (N, k) sample of histories."""
        if not self.use_thresholds:
            return
        drive = self._drive(tokens)
        step = rate * float(drive.std())
        target = self.winners / self.units
        self.theta[:] = 0.0
        for _ in range(steps):
            win = self._winners(drive)
            freq = np.zeros((2, self.units))
            for i in range(2):
                freq[i] = np.bincount(win[:, i].ravel(), minlength=self.units) / len(drive)
            self.theta += (step * np.sign(freq - target)).reshape(-1).astype(np.float32)

    def save(self, path):
        """The calibrated state, without the projection or the codes: both are rebuilt from the
        settings (the projection is a seeded draw) and the codes file."""
        np.savez(path, theta=self.theta, settings=np.array(json.dumps(self.settings)))

    @classmethod
    def load(cls, path, codes, token_freq):
        z = np.load(path)
        h = cls(codes, token_freq=token_freq, **json.loads(str(z["settings"])))
        h.theta[:] = z["theta"]
        return h

    def unit_usage(self, tokens):
        """Per sub-codebook: share of the busiest cell and entropy of cell use as a fraction of the maximum."""
        win = self._winners(self._drive(tokens))
        out = []
        for i in range(2):
            p = np.bincount(win[:, i].ravel(), minlength=self.units).astype(np.float64)
            p /= p.sum()
            nz = p[p > 0]
            out.append({"max_share": float(p.max()), "entropy_fraction": float(-(nz * np.log(nz)).sum() / np.log(self.units))})
        return out


NO_BUCKET = np.uint64(2**64 - 1)


def exclude_eot(keyfn, eot):
    """A history holding the end-of-text id gets no bucket (as it gets no exact row): its key is one no table holds."""
    def wrapped(packed, tokens):
        keys = np.array(keyfn(packed, tokens), dtype=np.uint64)
        keys[(tokens == eot).any(axis=1)] = NO_BUCKET
        return keys
    return wrapped


def unpack(packed, k):
    """(N, k) token ids of N packed histories (oldest token in the high bits)."""
    shifts = np.arange(k - 1, -1, -1, dtype=np.uint64) * np.uint64(C.VOCAB_BITS)
    return ((packed[:, None] >> shifts) & np.uint64((1 << C.VOCAB_BITS) - 1)).astype(np.int64)


def bucket_table(tokens, k, eot, keyfn, c_min=1, parts=16, chunk=C.CHUNK, merge_every=4):
    """Count table keyed by bucket: the continuation counts of every history in a bucket, pooled.
    keyfn(packed, tokens) -> (N,) uint64 bucket of each distinct history."""
    acc = None

    def reduce(triples):
        group = C.group_weighted(*triples)
        hist, hcount, fanout, entry_hist, tok, cnt = group
        return hist[entry_hist], tok, cnt

    for n, (hist, hcount, fanout, entry_hist, tok, cnt) in enumerate(C._partitions(tokens, k, eot, parts, chunk)):
        buckets = keyfn(hist, unpack(hist, k))
        part = (buckets[entry_hist], tok, cnt)
        acc = part if acc is None else tuple(np.concatenate(pair) for pair in zip(acc, part))
        if (n + 1) % merge_every == 0:
            acc = reduce(acc)
    if acc is None:
        raise ValueError("no history in the corpus")
    return C.build_from_groups([C.group_weighted(*acc)], k, c_min)


def save_buckets(directory, tables, kind, eot, hashes=None, codes=None, freq=None, n_buckets=None):
    """Write bucket tables (keyed by history length) and what a reader needs to hash new histories:
    for 'kenyon' the calibrated hashes and the codes, for 'random' the bucket count."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for k, table in tables.items():
        C.save(table, d / f"order{k + 1}", {"c_min": 1, "eot_id": eot, "kind": kind})
    if kind == "kenyon":
        for k, h in hashes.items():
            h.save(d / f"hash_k{k}.npz")
        np.savez(d / "codes.npz", codes=codes, freq=freq)
    elif kind != "random":
        raise ValueError(f"unknown bucket kind {kind!r}")
    (d / "bucket.json").write_text(json.dumps({"kind": kind, "orders": sorted(tables), "eot_id": eot,
                                               "n_buckets": n_buckets}, indent=2) + "\n")


def load_buckets(directory):
    """{history length: (bucket table, keyfn)} for CountReader(buckets=...)."""
    d = Path(directory)
    info = json.loads((d / "bucket.json").read_text())
    out = {}
    if info["kind"] == "kenyon":
        z = np.load(d / "codes.npz")
    for k in info["orders"]:
        table = C.load(d / f"order{k + 1}")[0]
        if info["kind"] == "random":
            n = info["n_buckets"]
            fn = lambda packed, toks, n=n: hash_bucket(packed, n)
        else:
            h = KenyonHash.load(d / f"hash_k{k}.npz", z["codes"], z["freq"])
            fn = lambda packed, toks, h=h: h.bucket_keys(toks)
        out[k] = (table, exclude_eot(fn, info["eot_id"]))
    return out
