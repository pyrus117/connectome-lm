"""Stage 2B-0: score an exact n-gram model, a bucket n-gram model and their mixture.

The exact model is clm.ngram_lm.NgramLM over the count tables. The bucket model
is the same model with the tables for histories of two or more tokens replaced
by bucket tables (every history in a bucket pools its continuations), so an
unseen history still lands in a bucket that similar histories filled. The
mixture is p = (1 - lam_c) p_exact + lam_c p_bucket, one weight per class c of
the exact 4-token history's stored count; the weights are fit by EM on text held
out of the counts, never on validation.
"""
import numpy as np
import torch

from clm.countread import BUCKET_LABELS, bucket_of, find_rows
from clm.data import eval_windows, get_batch

N_CLASSES = len(BUCKET_LABELS) + 1          # the five count buckets, plus "window too short" (never mixed)


def history_class(table4, x):
    """Class of each position by the stored count of its exact table4.k-token history:
    0..4 as in BUCKET_LABELS, 5 where the window is too short to hold the history."""
    counts = np.full(x.shape, -1, dtype=np.int64)
    counts[:, table4.k - 1:] = 0
    bi, ti, row = find_rows(table4, x)
    counts[bi, ti] = table4.hcount[row]
    cls = bucket_of(counts)
    return np.where(cls < 0, N_CLASSES - 1, cls)


def _batches(val, ctx, n_eval, micro):
    starts = eval_windows(len(val), ctx, n_eval)
    for i in range(0, len(starts), micro):
        yield tuple(t.numpy() for t in get_batch(val, starts[i:i + micro], ctx, torch.device("cpu")))


def score_model(lm, val, ctx, n_eval, micro=8):
    """Probability of every target of the fixed windows of `val` under `lm`, flat in window order."""
    return np.concatenate([lm.prob(x, y[..., None])[..., 0].ravel() for x, y in _batches(val, ctx, n_eval, micro)])


def score_classes(table4, val, ctx, n_eval, micro=8):
    return np.concatenate([history_class(table4, x).ravel() for x, _ in _batches(val, ctx, n_eval, micro)])


def score_windows(exact, bucket, table4, val, ctx, n_eval, micro=8):
    """Probability of each target under both models and its class, as flat arrays in window order."""
    return (score_model(exact, val, ctx, n_eval, micro), score_model(bucket, val, ctx, n_eval, micro),
            score_classes(table4, val, ctx, n_eval, micro))


def fit_lambdas(pe, pb, cls, iters=200):
    """Per-class mixing weight maximising sum log((1 - lam) pe + lam pb), by EM; the last
    class (window too short) is not mixed."""
    lam = np.zeros(N_CLASSES)
    for c in range(N_CLASSES - 1):
        a, b = pe[cls == c], pb[cls == c]
        if a.size == 0:
            continue
        l = 0.5
        for _ in range(iters):
            r = l * b / ((1 - l) * a + l * b)
            l = float(r.mean())
        lam[c] = l
    return lam


def mixture_probs(pe, pb, cls, lam):
    w = lam[cls]
    return (1 - w) * pe + w * pb


def class_loss(p, cls):
    """Mean negative log-likelihood per class and overall (nats/token); None for an empty class."""
    nll = -np.log(p)
    out = {BUCKET_LABELS[c]: (float(nll[cls == c].mean()) if (cls == c).any() else None)
           for c in range(N_CLASSES - 1)}
    out["short window"] = float(nll[cls == N_CLASSES - 1].mean()) if (cls == N_CLASSES - 1).any() else None
    out["all"] = float(nll.mean())
    return out


def rare_loss(p, cls):
    """Mean NLL over the "not stored" and "1-9" classes: the quantity settings are chosen on."""
    return float((-np.log(p))[(cls == 0) | (cls == 1)].mean())
