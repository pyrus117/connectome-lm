"""A normalised n-gram language model over the count tables, the model-free
reference for Stage 2A.

At each position, from the longest history down: a stored history gives each
stored top-TOP continuation the discounted ratio (C(h,w) - d) / C(h), and the
rest of its mass (the discounts plus everything C(h) counted beyond those
entries) is shared among the other tokens in proportion to the next shorter
history's probabilities (absolute discounting with back-off). A history that
is not stored defers wholly to the next shorter one; the bottom is the add-one
unigram. d = 0.75 is the textbook value (Chen & Goodman 1999), fixed before
any validation read. Probabilities sum to one over the vocabulary (tested).
Histories use only tokens inside the window, as the COUNT read does.
"""
import numpy as np

from clm import counts as C
from clm.countread import find_rows

DISCOUNT = 0.75


def add_one_unigram(tokens, vocab, chunk=1 << 26):
    counts = np.zeros(vocab, dtype=np.int64)
    for start in range(0, len(tokens), chunk):
        part = np.bincount(np.asarray(tokens[start:start + chunk]), minlength=vocab)
        if part.size > vocab:
            raise ValueError(f"token id above the vocabulary of {vocab}")
        counts += part
    return (counts + 1) / (counts.sum() + vocab)


class NgramLM:
    def __init__(self, tables, unigram, discount=DISCOUNT, keyfns=None):
        """tables: count tables by ascending history length. keyfns: optional, one per table (None
        for an exact table), each mapping packed histories and their tokens to the table's keys."""
        self.tables, self.uni, self.d = tables, unigram, discount
        self.keyfns = keyfns or [None] * len(tables)

    def prob(self, x, cand):
        """x: (batch, time) window tokens; cand: (batch, time, M) candidate next
        tokens. Returns P(candidate | the window's history up to that position)."""
        x = np.asarray(x)
        b, t = x.shape
        rows = []
        for table, keyfn in zip(self.tables, self.keyfns):
            r = np.full(b * t, -1, dtype=np.int64)
            bi, ti, row = find_rows(table, x, keyfn)
            r[bi * t + ti] = row
            rows.append(r)
        return self._level(len(self.tables), cand.reshape(b * t, -1), rows).reshape(cand.shape)

    def _level(self, level, cand, rows):
        if level == 0:
            return self.uni[cand]
        table, r = self.tables[level - 1], rows[level - 1]
        has = r >= 0
        r = np.where(has, r, 0)
        lo = table.offsets[r].astype(np.int64)
        n = np.where(has, np.minimum(table.offsets[r + 1].astype(np.int64) - lo, C.TOP), 0)
        live = np.arange(C.TOP) < n[:, None]
        at = np.where(live, lo[:, None] + np.arange(C.TOP), 0)
        stored = table.tok[at].astype(np.int64)
        ratio = np.where(live, (table.cnt[at] - self.d) / table.hcount[r][:, None], 0.0)
        mass = np.maximum(1.0 - ratio.sum(1), 0.0)
        m = cand.shape[1]
        low = self._level(level - 1, np.concatenate([cand, stored], axis=1), rows)
        low_cand, low_stored = low[:, :m], low[:, m:]
        denom = np.maximum(1.0 - (low_stored * live).sum(1), 1e-12)
        match = (cand[:, :, None] == stored[:, None, :]) & live[:, None, :]
        own = (match * ratio[:, None, :]).sum(2)
        p = np.where(match.any(2), own, mass[:, None] * low_cand / denom[:, None])
        return np.where(has[:, None], p, low_cand)

    def nll(self, x, y, chunk=8):
        """Negative log-likelihood of every target, (batch, time)."""
        x, y = np.asarray(x), np.asarray(y)
        out = np.empty(x.shape)
        for i in range(0, len(x), chunk):
            out[i:i + chunk] = -np.log(self.prob(x[i:i + chunk], y[i:i + chunk, :, None])[..., 0])
        return out
