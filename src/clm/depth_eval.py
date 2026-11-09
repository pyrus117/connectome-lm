"""Run a trained LoopGPT over the validation windows: per-token loss at every depth,
and the loss of an executed per-token exit policy. Pure numpy/torch glue for
scripts/stage2d_analyze.py; the arithmetic on the results is in clm.depth_law."""
import numpy as np
import torch
import torch.nn.functional as F

from clm import halting
from clm.data import eval_windows, get_batch
from clm.train import _autocast


def _windows(val, ctx, n_eval, micro, starts=None):
    starts = eval_windows(len(val), ctx, n_eval) if starts is None else starts
    for i in range(0, len(starts), micro):
        yield get_batch(val, starts[i:i + micro], ctx, torch.device("cpu"))


@torch.no_grad()
def collect(model, val, ctx, n_eval, micro, device):
    """(losses, change), each (max_depth, tokens): per-token loss at every depth and the
    relative state change of every pass, tokens in window order."""
    model.eval()
    losses, change = [], []
    for x, y in _windows(val, ctx, n_eval, micro):
        with _autocast(device):
            l, c = model.all_depths(x.to(device), y.to(device))
        losses.append(l.flatten(1).float().cpu().numpy())
        change.append(c.flatten(1).float().cpu().numpy())
    return np.concatenate(losses, axis=1), np.concatenate(change, axis=1)


@torch.no_grad()
def executed_loss(model, val, ctx, n_eval, micro, device, depths):
    """Mean loss when token i (window order) exits after depths[i] passes and later tokens
    read its frozen state. depths: integer array of length tokens."""
    model.eval()
    total, n, pos = 0.0, 0, 0
    for x, y in _windows(val, ctx, n_eval, micro):
        d = torch.from_numpy(depths[pos:pos + x.numel()]).view(x.shape).to(device)
        with _autocast(device):
            logits, _ = model(x.to(device), depth=d)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.to(device).reshape(-1), reduction="sum")
        total, n, pos = total + float(nll), n + x.numel(), pos + x.numel()
    return total / n


@torch.no_grad()
def collect_signals(model, val, ctx, n_eval, micro, device, predictors=None):
    """From one full-depth run: per-token loss, state change, readout entropy and margin at every pass, each
    (max_depth, tokens) in window order, plus predicted[name] (max_depth - 1, tokens) for every predictor
    predict(r, state, logits, change) -> (batch, time) in `predictors` (a probe's predicted gain of pass r + 1)."""
    model.eval()
    keys = ("loss", "change", "entropy", "margin")
    out, pred = {k: [] for k in keys}, {k: [] for k in (predictors or {})}
    for x, y in _windows(val, ctx, n_eval, micro):
        x, y = x.to(device), y.to(device)
        per = {k: [] for k in keys}
        per_pred = {k: [] for k in pred}
        with _autocast(device):
            h, ves = model._stem(x)
            for r in range(1, model.max_depth + 1):
                new = model._pass(h, r)
                change = (new - h).float().norm(dim=-1) / new.float().norm(dim=-1).clamp_min(1e-12)
                h = new
                logits = model._read(h, ves)
                ent, mar = halting.signals(logits)
                per["loss"].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                                   reduction="none").view(y.shape))
                per["change"].append(change)
                per["entropy"].append(ent)
                per["margin"].append(mar)
                if r < model.max_depth:
                    for k, fn in (predictors or {}).items():
                        per_pred[k].append(fn(r, h, logits, change))
        for k in keys:
            out[k].append(torch.stack(per[k]).flatten(1).float().cpu().numpy())
        for k in pred:
            pred[k].append(torch.stack(per_pred[k]).flatten(1).float().cpu().numpy())
    res = {k: np.concatenate(v, axis=1) for k, v in out.items()}
    res["predicted"] = {k: np.concatenate(v, axis=1) for k, v in pred.items()}
    return res


@torch.no_grad()
def collect_fit(model, tokens, starts, ctx, micro, device, stride=8):
    """Probe training data from a full-depth run (nothing stopped) over the windows at `starts`, at the positions
    p with p % stride == stride - 1. For passes 1..R-1: small (3 signals, see clm.halting.features), state (the
    stream after the pass) and gain (loss at depth r minus loss at depth r + 1), each (R - 1, samples, ...) in
    window order. state is float16."""
    model.eval()
    small, state, gains = [], [], []
    pick = slice(stride - 1, ctx, stride)
    for x, y in _windows(tokens, ctx, None, micro, starts):
        x, y = x.to(device), y.to(device)
        with _autocast(device):
            h, ves = model._stem(x)
            feats, states, losses = [], [], []
            for r in range(1, model.max_depth + 1):
                new = model._pass(h, r)
                change = (new - h).float().norm(dim=-1) / new.float().norm(dim=-1).clamp_min(1e-12)
                h = new
                logits = model._read(h, ves)
                losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                              reduction="none").view(y.shape)[:, pick])
                feats.append(halting.features(h, logits, change, False)[:, pick])
                states.append(h[:, pick].half())
        losses = torch.stack(losses).float()
        gains.append((losses[:-1] - losses[1:]).flatten(1).cpu().numpy())
        small.append(torch.stack(feats[:-1]).flatten(1, 2).cpu().numpy())
        state.append(torch.stack(states[:-1]).flatten(1, 2).cpu().numpy())
    return np.concatenate(small, axis=1), np.concatenate(state, axis=1), np.concatenate(gains, axis=1)


def free_exit_depths(change, theta):
    """Depth of the first pass whose relative state change is below theta (the last pass if none)."""
    below = change < theta
    first = np.where(below.any(axis=0), below.argmax(axis=0), change.shape[0] - 1)
    return first + 1
