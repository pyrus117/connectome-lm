"""Run a trained LoopGPT over the validation windows: per-token loss at every depth,
and the loss of an executed per-token exit policy. Pure numpy/torch glue for
scripts/stage2d_analyze.py; the arithmetic on the results is in clm.depth_law."""
import numpy as np
import torch
import torch.nn.functional as F

from clm.data import eval_windows, get_batch
from clm.train import _autocast


def _windows(val, ctx, n_eval, micro):
    starts = eval_windows(len(val), ctx, n_eval)
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


def free_exit_depths(change, theta):
    """Depth of the first pass whose relative state change is below theta (the last pass if none)."""
    below = change < theta
    first = np.where(below.any(axis=0), below.argmax(axis=0), change.shape[0] - 1)
    return first + 1
