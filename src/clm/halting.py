"""Stage 2I: exit rules for a looped model, and how they are scored (PREREG_stage2i.md).

A rule reads, for every token and pass r, signals that token has computed by pass r (entropy and top-1 minus top-2
margin of its readout at that depth, the relative change of its state, or a linear probe's predicted gain of the next
pass) and stops at the first pass where the signal says stop. As in Stage 2D the signals come from one full-depth run
and the loss comes from really executing the chosen depths (`clm.depth_eval.executed_loss`: a stopped token keeps its
state and later tokens read it frozen). A probe predicts the gain of pass r + 1 (loss at depth r minus loss at depth
r + 1) and a token stops when that is below the price of a pass."""
import numpy as np
import torch

from clm.analyze import verdict

REGION = (1.25, 2.70)            # mean depths where the Stage 2H oracle has room (its mean depth ran 1.25 to 2.69)
MIN_POINTS = 3                   # frontier points a rule needs inside REGION to be scored


def signals(logits):
    """Entropy (nats) and top-1 minus top-2 logit margin of each token's next-token distribution, float32."""
    z = logits.float()
    logp = torch.log_softmax(z, dim=-1)
    entropy = -(logp.exp() * logp).sum(-1)
    top2 = z.topk(2, dim=-1).values
    return entropy, top2[..., 0] - top2[..., 1]


def features(x, logits, change, use_state):
    """Probe inputs for one pass: entropy, margin, state change, and (if use_state) the state. (..., F), float32."""
    h, m = signals(logits)
    small = torch.stack([h, m, change.float()], dim=-1)
    return torch.cat([small, x.float()], dim=-1) if use_state else small


def probe_predictor(probes, use_state):
    """predict(r, x, logits, change) -> (batch, time) predicted gain of pass r + 1, from probes[r - 1] (a ridge_fit)."""
    def predict(r, x, logits, change):
        p = probes[r - 1]
        f = features(x, logits, change, use_state)
        mean, std, w = (torch.as_tensor(p[k], dtype=torch.float32, device=f.device) for k in ("mean", "std", "w"))
        return ((f - mean) / std) @ w + p["b"]
    return predict


def stop_matrix(rule, sig, level):
    """bool (R, N): where `rule` stops a token after each pass. sig: per-pass arrays from depth_eval.collect_signals.
    entropy and change stop at or below the level, margin at or above it, a probe when its predicted gain is below it
    (the price of a pass); the last row is never read."""
    if rule == "entropy":
        return sig["entropy"] <= level
    if rule == "margin":
        return sig["margin"] >= level
    if rule == "change":
        return sig["change"] <= level
    pred = sig["predicted"][rule]
    return np.vstack([pred < level, np.zeros((1, pred.shape[1]), dtype=bool)])


def level_for_share(rule, values, share):
    """The threshold at which `share` of `values` (a pooled array of one signal) would stop under `rule`."""
    return float(np.quantile(values, 1 - share if rule == "margin" else share))


def exit_depths(stop):
    """stop: bool (R, N), True where a token would stop after that pass. Each token's first stopping pass; the last
    pass if none (row R - 1 is never read)."""
    first = np.where(stop[:-1].any(axis=0), stop[:-1].argmax(axis=0), stop.shape[0] - 1)
    return first + 1


def ridge_fit(X, y, alphas, holdout=0.25):
    """Standardise X, fit ridge with an unpenalised intercept. alpha is chosen on the last `holdout` rows
    (rows are in window order, so they are whole later windows), then the model is refitted on all rows."""
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mean, std = X.mean(0), X.std(0)
    std = np.where(std > 1e-12, std, 1.0)
    Z = (X - mean) / std

    def solve(Zt, yt, a):
        zm, ym = Zt.mean(0), yt.mean()
        A = (Zt - zm).T @ (Zt - zm) + a * np.eye(Zt.shape[1])
        w = np.linalg.solve(A, (Zt - zm).T @ (yt - ym))
        return w, ym - zm @ w

    cut = int(len(y) * (1 - holdout))
    err = {}
    for a in alphas:
        w, b = solve(Z[:cut], y[:cut], a)
        err[a] = float(np.mean((Z[cut:] @ w + b - y[cut:]) ** 2))
    alpha = min(err, key=err.get)
    w, b = solve(Z, y, alpha)
    return {"mean": mean, "std": std, "w": w, "b": float(b), "alpha": alpha, "holdout_mse": err}


def region_scores(points, fixed_points, region=REGION):
    """points: (mean_depth, executed loss) of one rule's frontier; fixed_points: the same for fixed depth 1..R.
    Over the rule's points whose mean depth lies in `region`: how many, their mean loss, and the mean loss of
    fixed depth at those same mean depths (linear interpolation). None when fewer than MIN_POINTS points qualify."""
    fx = sorted(fixed_points)
    inside = [(d, l) for d, l in points if region[0] <= d <= region[1]]
    if len(inside) < MIN_POINTS:
        return {"n": len(inside), "rule_loss": None, "fixed_loss": None, "gap": None}
    rule = float(np.mean([l for _, l in inside]))
    fixed = float(np.mean(np.interp([d for d, _ in inside], [p[0] for p in fx], [p[1] for p in fx])))
    return {"n": len(inside), "rule_loss": rule, "fixed_loss": fixed, "gap": fixed - rule}


def beats_fixed(scores_by_seed):
    """The Stage 1 paired rule on rule loss against fixed-depth loss at equal mean depth, over seeds (needs a score
    on every seed). X_BETTER means the rule is at least SESOI lower on average and lower on every seed."""
    if any(s["gap"] is None for s in scores_by_seed.values()):
        return "NOT_SCORED"
    return verdict({k: s["rule_loss"] for k, s in scores_by_seed.items()},
                   {k: s["fixed_loss"] for k, s in scores_by_seed.items()})
