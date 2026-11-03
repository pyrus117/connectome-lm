"""COUNT read (Magnusson, Smith & Koh 2026, arXiv 2610.20456, Eq. 1-2) over the
tables built by clm.counts.

For every position the reader returns, per order, the first `k_top` stored
continuations of that order's history with their count ratios C(h,w)/C(h).
Scores carry Eq. 1's 1/J, so the model forms f_t = sum_j sum_k S(w_k|h_j) B[w_k]
/ J with one weighted sum over the head rows B (B0GPT.forward, `count`) and adds
f_t to the token embedding.

Choices fixed here, all stated in PREREG_stage2a.md:
- One slot per table order; a history that is not stored (count below c_min,
  never seen, or holding an end-of-text id) leaves its slot empty. The
  average keeps the constant 1/J of Eq. 1.
- A history uses only tokens inside the model's window, so the first k-1
  positions of a window have no order-k slot.
- Eq. 2's 0.4 back-off applies to an unobserved continuation; a stored top-K
  continuation is always observed, so scores are plain count ratios.
- Leave-one-out (training only, `y` given): the position's own occurrence is
  subtracted from C(h,w) and C(h); a history left under c_min is dropped.
  The count corpus starts with the training stream, so the occurrence is
  there exactly once. The first k_top continuations are taken before the
  subtraction and are not re-ranked after it.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from clm import counts as C


BUCKET_EDGES = (1, 10, 100, 1000)
BUCKET_LABELS = ("not stored", "1-9", "10-99", "100-999", ">=1000")
# Gated read (s2model.B0GPT._gate): count bins of a stored history, 1-9 / 10-99 / 100-999 / >=1000.
GATE_EDGES = BUCKET_EDGES[1:]
N_GATE_BINS = len(GATE_EDGES) + 1


def bucket_of(history_counts):
    """Bucket index 0..4 of each stored-history count (see BUCKET_LABELS); positions
    whose window is too short (count -1) get -1. Orders 4 and 5 store no history
    seen once, so their bucket 0 also holds the seen-once histories."""
    return np.where(history_counts < 0, -1, np.digitize(history_counts, BUCKET_EDGES))


def history_keys(x, k):
    """Packed k-token histories ending at each time index in `last` (k-1 .. time-1), (batch, len(last))."""
    last = np.arange(k - 1, x.shape[1])
    key = np.zeros((x.shape[0], last.size), dtype=np.uint64)
    for j in range(k):
        key = (key << np.uint64(C.VOCAB_BITS)) | x[:, last - (k - 1) + j].astype(np.uint64)
    return key, last


def find_rows(table, x, keyfn=None, want=None):
    """Positions whose history of table.k tokens, taken inside the window, is stored.
    Returns batch index, time index and row of each. A history holding the
    end-of-text id is never stored, so it is never found. `keyfn(packed, tokens)`
    maps each history to another key (a bucket id) before the lookup; `want`, a
    boolean (batch, time - k + 1) mask, restricts the lookup to those positions."""
    key, last = history_keys(x, table.k)
    if want is None:
        want = np.ones(key.shape, dtype=bool)
    bi, pi = np.nonzero(want)
    key = key[bi, pi]
    if keyfn is not None:
        toks = np.stack([x[bi, last[pi] - (table.k - 1) + j] for j in range(table.k)], axis=-1)
        key = keyfn(key, toks)
    row = np.searchsorted(table.keys, key)
    hit = table.keys[np.minimum(row, table.keys.size - 1)] == key
    return bi[hit], last[pi[hit]], row[hit]


class CountReader:
    def __init__(self, directory, k_top=8, buckets=None):
        """buckets: optional {history length: (bucket table, keyfn)} (Stage 2B). Where an order's exact
        history is not stored, its slot holds the pooled counts of the history's bucket instead."""
        found = sorted(Path(directory).glob("order*"))
        if not found:
            raise FileNotFoundError(f"no order* tables in {directory}")
        loaded = [C.load(d) for d in found]
        loaded.sort(key=lambda tm: tm[0].k)
        self.tables = [t for t, _ in loaded]
        self.c_min = [m["c_min"] for _, m in loaded]
        eots = {m["eot_id"] for _, m in loaded}
        if len(eots) != 1:
            raise ValueError(f"tables disagree on the end-of-text id: {eots}")
        self.eot = eots.pop()
        if not 1 <= k_top <= C.TOP:
            raise ValueError(f"k_top {k_top} outside 1..{C.TOP}")
        self.k_top = k_top
        self.slots = len(self.tables) * k_top
        self.buckets = buckets or {}
        for k in self.buckets:
            if k not in {t.k for t in self.tables}:
                raise ValueError(f"buckets for history length {k}, which has no exact table")

    def lookup(self, x, y=None, states=False):
        """x, y: integer arrays (batch, time). Returns tokens int64 and scores
        float32, both (batch, time, slots); unused slots hold token 0, score 0.
        Scores are count ratios divided by the number of orders J.
        With `y` (the next token at each position) the position's own
        occurrence is left out of the counts.
        With `states`, also returns int8 (batch, time, J): 0 where the order has
        no stored history, else 1 + source * N_GATE_BINS + count bin (source 0
        exact, 1 bucket; the bin of the history count after leave-one-out)."""
        x = np.asarray(x)
        b, t = x.shape
        tok = np.zeros((b, t, self.slots), dtype=np.int64)
        score = np.zeros((b, t, self.slots), dtype=np.float32)
        state = np.zeros((b, t, len(self.tables)), dtype=np.int8)
        kk = self.k_top
        for o, (table, c_min) in enumerate(zip(self.tables, self.c_min)):
            k = table.k
            if t < k:
                continue
            sl = slice(o * kk, (o + 1) * kk)
            bi, ti, row = find_rows(table, x)
            keep, hc = self._fill(tok, score, sl, table, c_min, bi, ti, row, y)
            state[bi[keep], ti[keep], o] = 1 + np.digitize(hc[keep], GATE_EDGES)
            if k in self.buckets:
                used = np.zeros((b, t), dtype=bool)
                used[bi[keep], ti[keep]] = True
                bucket_table, keyfn = self.buckets[k]
                bi2, ti2, row2 = find_rows(bucket_table, x, keyfn, want=~used[:, k - 1:])
                keep2, hc2 = self._fill(tok, score, sl, bucket_table, 1, bi2, ti2, row2, y)
                state[bi2[keep2], ti2[keep2], o] = 1 + N_GATE_BINS + np.digitize(hc2[keep2], GATE_EDGES)
        return (tok, score, state) if states else (tok, score)

    def _fill(self, tok, score, sl, table, c_min, bi, ti, row, y):
        """Write the first k_top stored continuations of each found row into the slot columns `sl`,
        with leave-one-out when `y` is given. Returns which rows were kept (history count still
        at least c_min after the position's own occurrence is removed) and every row's history count."""
        kk = self.k_top
        lo = table.offsets[row].astype(np.int64)
        n = np.minimum(table.offsets[row + 1].astype(np.int64) - lo, kk)
        live = np.arange(kk) < n[:, None]
        at = np.where(live, lo[:, None] + np.arange(kk), 0)
        etok, ecnt = table.tok[at].astype(np.int64), table.cnt[at].astype(np.int64)
        hc = table.hcount[row].astype(np.int64)
        keep = np.ones(hc.size, dtype=bool)
        if y is not None:
            ecnt = ecnt - (live & (etok == np.asarray(y)[bi, ti][:, None]))
            hc = hc - 1
            keep = hc >= c_min
        ok = live & keep[:, None]
        tok[bi, ti, sl] = np.where(ok, etok, 0)
        score[bi, ti, sl] = np.where(ok, ecnt / np.maximum(hc, 1)[:, None], 0.0) / len(self.tables)
        return keep, hc

    def history_counts(self, x, k):
        """C(h) of the stored k-token history ending at each position; 0 where it is
        not stored (never seen, under c_min, or holding an end-of-text id) and -1
        where the window is too short to hold k tokens."""
        x = np.asarray(x)
        table = next(t for t in self.tables if t.k == k)
        out = np.full(x.shape, -1, dtype=np.int64)
        out[:, k - 1:] = 0
        bi, ti, row = find_rows(table, x)
        out[bi, ti] = table.hcount[row]
        return out


def reader_for(count_dir, k_top, bucket_dir=None):
    """The COUNT reader of a run: exact tables, plus bucket fallbacks when `bucket_dir` is given."""
    from clm.kenyon import load_buckets
    return CountReader(count_dir, k_top, load_buckets(bucket_dir) if bucket_dir else None)


@dataclass
class StepData:
    step: int
    x: np.ndarray
    y: np.ndarray
    tok: np.ndarray | None
    score: np.ndarray | None
    lookup_seconds: float
    state: np.ndarray | None = None

    def count(self, sl, device):
        """The model's `count` argument for batch rows `sl`: (tokens, scores), plus states when read."""
        arrays = (self.tok, self.score) if self.state is None else (self.tok, self.score, self.state)
        return tuple(torch.from_numpy(a[sl]).to(device) for a in arrays)


def make_step(train, ctx, batch, seed, step, reader, loo, states=False):
    """Inputs, targets and COUNT scores of one optimizer step. Depends only on
    (seed, step), so a resumed run reads exactly what an unbroken run would."""
    from clm.data import batch_starts
    starts = batch_starts(len(train), ctx, batch, seed, step)
    x = np.stack([train[s:s + ctx] for s in starts]).astype(np.int64)
    y = np.stack([train[s + 1:s + ctx + 1] for s in starts]).astype(np.int64)
    if reader is None:
        return StepData(step, x, y, None, None, 0.0)
    t0 = time.perf_counter()
    got = reader.lookup(x, y if loo else None, states)
    return StepData(step, x, y, got[0], got[1], time.perf_counter() - t0, got[2] if states else None)


class Prefetcher:
    """Builds step s + 1 on a worker thread while step s trains."""

    def __init__(self, make, end):
        self.make, self.end = make, end
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.hits = 0

    def get(self, step):
        if self.pending is not None and self.pending[0] == step:
            data = self.pending[1].result()
            self.hits += 1
        else:
            data = self.make(step)
        nxt = step + 1
        self.pending = (nxt, self.pool.submit(self.make, nxt)) if nxt < self.end else None
        return data

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
