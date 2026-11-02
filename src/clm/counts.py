"""Counted n-gram table for the Stage 2A memory (COUNT).

An order-n table maps every history of k = n - 1 tokens seen in the count
corpus to its total count and its TOP most frequent continuations with their
counts. The corpus is one uint16 token stream with an end-of-text id closing
each document. A history never contains the end-of-text id, so no n-gram
crosses a document boundary, but a continuation may be the end-of-text id.

Rows are sorted by packed key (oldest token in the high bits), so a lookup is
one searchsorted. The build runs in partitions by the oldest history token to
bound host RAM; the partitions are contiguous key ranges, so concatenating
them keeps the global order.

The minimum history count (c_min) comes from a size budget alone: `profile`
counts rows and entries for every threshold, `pick_cmin` takes the smallest
threshold that fits. No validation loss enters the choice.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VOCAB_BITS = 13
TOP = 16
CAP = 1 << 16                 # history counts above this share one profile bucket
ROW_BYTES = 16                # packed key 8 + history count 4 + offset 4
ENTRY_BYTES = 6               # continuation token 2 + count 4
CHUNK = 1 << 25
_TOKEN_MASK = np.uint64((1 << VOCAB_BITS) - 1)


@dataclass
class Table:
    k: int                    # history length in tokens
    keys: np.ndarray          # uint64 packed histories, strictly ascending
    hcount: np.ndarray        # uint32 occurrences of each history
    offsets: np.ndarray       # uint32, len(keys) + 1; row r owns tok/cnt[offsets[r]:offsets[r+1]]
    tok: np.ndarray           # uint16 continuations, highest count first, ties by lower token id
    cnt: np.ndarray           # uint32 occurrences of history + continuation


def table_bytes(rows, entries):
    return ROW_BYTES * rows + ENTRY_BYTES * entries


def pack_history(tokens):
    key = 0
    for t in tokens:
        key = (key << VOCAB_BITS) | int(t)
    return key


def _windows(n_tokens, k, chunk):
    for start in range(0, n_tokens - k, chunk):
        yield start, min(start + chunk, n_tokens - k)


def partition_bounds(tokens, parts, eot, chunk=CHUNK):
    """Cut points over the oldest history token that give each partition about
    the same number of positions. Returns parts + 1 ascending values."""
    freq = np.zeros(1 << VOCAB_BITS, dtype=np.int64)
    for start in range(0, len(tokens), chunk):
        part = np.bincount(np.asarray(tokens[start:start + chunk]), minlength=1 << VOCAB_BITS)
        if part.size > 1 << VOCAB_BITS:
            raise ValueError(f"a token id does not fit in {VOCAB_BITS} bits")
        freq += part
    freq[eot] = 0
    cum = np.cumsum(freq)
    cuts = [int(np.searchsorted(cum, cum[-1] * p / parts, side="left")) + 1 for p in range(1, parts)]
    return [0, *cuts, 1 << VOCAB_BITS]


def _collect(tokens, k, eot, lo, hi, chunk):
    """Packed histories and continuations for every position whose history
    starts with a token in [lo, hi) and holds no end-of-text id."""
    keys, conts = [], []
    for start, stop in _windows(len(tokens), k, chunk):
        a = np.asarray(tokens[start:stop + k])
        n = stop - start
        ok = (a[:n] >= lo) & (a[:n] < hi)
        for j in range(k):
            ok &= a[j:j + n] != eot
        sel = np.flatnonzero(ok)
        key = a[sel].astype(np.uint64)
        for j in range(1, k):
            key = (key << np.uint64(VOCAB_BITS)) | a[sel + j]
        keys.append(key)
        conts.append(a[sel + k])
    if not keys:
        return np.empty(0, np.uint64), np.empty(0, np.uint16)
    return np.concatenate(keys), np.concatenate(conts)


def _group(keys, conts):
    """Counts per (history, continuation), plus per-history totals.

    Returns hist (sorted unique packed histories), hcount, fanout (distinct
    continuations per history), and the continuation entries in history order:
    entry_hist (row index), tok, cnt."""
    hist, inverse = np.unique(keys, return_inverse=True)
    pair = (inverse.astype(np.uint64) << np.uint64(VOCAB_BITS)) | conts.astype(np.uint64)
    pair, cnt = np.unique(pair, return_counts=True)
    entry_hist = (pair >> np.uint64(VOCAB_BITS)).astype(np.int64)
    tok = (pair & _TOKEN_MASK).astype(np.uint16)
    starts = np.flatnonzero(np.r_[True, entry_hist[1:] != entry_hist[:-1]])
    hcount = np.add.reduceat(cnt, starts)
    fanout = np.diff(np.r_[starts, len(entry_hist)])
    return hist, hcount, fanout, entry_hist, tok, cnt


def _partitions(tokens, k, eot, parts, chunk):
    if not 1 <= k <= 4:
        raise ValueError(f"history length {k} outside 1..4 (a packed key must fit in 64 bits)")
    bounds = partition_bounds(tokens, parts, eot, chunk)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if lo == hi:
            continue
        keys, conts = _collect(tokens, k, eot, lo, hi, chunk)
        if keys.size:
            yield _group(keys, conts)


def profile(tokens, k, eot, parts=16, chunk=CHUNK):
    """rows[c] and entries[c]: histories whose count falls in bucket c (counts
    of CAP or more share the last bucket) and the continuation entries they
    would keep. Thresholds are read off the suffix sums."""
    rows = np.zeros(CAP + 1, dtype=np.int64)
    entries = np.zeros(CAP + 1, dtype=np.int64)
    for _, hcount, fanout, *_ in _partitions(tokens, k, eot, parts, chunk):
        bucket = np.minimum(hcount, CAP)
        rows += np.bincount(bucket, minlength=CAP + 1)
        entries += np.bincount(bucket, weights=np.minimum(fanout, TOP), minlength=CAP + 1).astype(np.int64)
    return rows, entries


def sizes_by_threshold(rows, entries):
    """(rows kept, entries kept) for each minimum history count c = 0..CAP."""
    return rows[::-1].cumsum()[::-1], entries[::-1].cumsum()[::-1]


def pick_cmin(rows, entries, budget_bytes):
    """Smallest minimum history count whose table fits the byte budget."""
    kept_rows, kept_entries = sizes_by_threshold(rows, entries)
    for c in range(1, CAP + 1):
        if kept_rows[c] == 0:
            break
        if table_bytes(kept_rows[c], kept_entries[c]) <= budget_bytes:
            return c
    raise ValueError(f"no threshold keeps a non-empty table within the {budget_bytes}-byte budget")


def build(tokens, k, eot, c_min, parts=16, chunk=CHUNK):
    if c_min < 1:
        raise ValueError("c_min must be at least 1")
    out = {name: [] for name in ("keys", "hcount", "lens", "tok", "cnt")}
    for hist, hcount, fanout, entry_hist, tok, cnt in _partitions(tokens, k, eot, parts, chunk):
        keep = hcount >= c_min
        if not keep.any():
            continue
        live = keep[entry_hist]
        e_hist, e_tok, e_cnt = entry_hist[live], tok[live], cnt[live]
        order = np.lexsort((e_tok, -e_cnt, e_hist))
        e_hist, e_tok, e_cnt = e_hist[order], e_tok[order], e_cnt[order]
        first = np.flatnonzero(np.r_[True, e_hist[1:] != e_hist[:-1]])
        rank = np.arange(e_hist.size) - np.repeat(first, np.diff(np.r_[first, e_hist.size]))
        top = rank < TOP
        out["keys"].append(hist[keep])
        out["hcount"].append(hcount[keep].astype(np.uint32))
        out["lens"].append(np.minimum(fanout[keep], TOP))
        out["tok"].append(e_tok[top])
        out["cnt"].append(e_cnt[top].astype(np.uint32))
    if not out["keys"]:
        raise ValueError(f"no history reaches c_min={c_min}")
    lens = np.concatenate(out.pop("lens"))
    entries = int(lens.sum())
    if entries >= 1 << 32:
        raise ValueError(f"{entries} entries do not fit uint32 offsets")
    offsets = np.zeros(lens.size + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(lens)
    joined = {name: np.concatenate(out.pop(name)) for name in ("keys", "hcount", "tok", "cnt")}   # one copy live at a time
    return Table(k, joined["keys"], joined["hcount"], offsets, joined["tok"], joined["cnt"])


def save(table, directory, meta):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for name in ("keys", "hcount", "offsets", "tok", "cnt"):
        np.save(d / f"{name}.npy", getattr(table, name))
    (d / "meta.json").write_text(json.dumps({"k": table.k, **meta}, indent=2) + "\n")


def load(directory, mmap=True):
    d = Path(directory)
    meta = json.loads((d / "meta.json").read_text())
    arrays = {n: np.load(d / f"{n}.npy", mmap_mode="r" if mmap else None)
              for n in ("keys", "hcount", "offsets", "tok", "cnt")}
    return Table(meta["k"], **arrays), meta
