"""Stage 2D analysis: exact optimal depth per token and DeepSeek's effort law.

A looped model that reads out after every pass gives, for each validation token,
its loss at every depth 1..R. At effort b the price of one extra pass is
k(b) = k0 * exp(-b / tau) nats (DeepSeek-V4.1-Flash App. C, price decaying
exponentially with effort). The best depth for a token is the one minimising
loss + k(b) * depth, exact because every depth's loss is known. If the marginal
gain of one more pass decays geometrically with depth, the average best depth is
affine in b (the law P9 tests): with gain(r) = g0 * exp(-r / sigma), a token's
best depth is sigma * ln(g0 / k(b)), whose slope in b is sigma / tau.
"""
import math

import numpy as np


def price(effort, k0, tau):
    return k0 * np.exp(-np.asarray(effort, dtype=float) / tau)


def oracle_depths(losses, k):
    """losses: (R, N) per-token loss at depths 1..R. Best depth per token at price k;
    a tie goes to the shallower depth."""
    cost = losses + k * np.arange(1, losses.shape[0] + 1)[:, None]
    return cost.argmin(axis=0) + 1


def frontier(losses, efforts, k0, tau):
    """Per effort: the oracle's mean depth and mean predicted loss (each token read at
    its own best depth, as if no token's depth changed what later tokens see)."""
    rows = []
    for b in efforts:
        d = oracle_depths(losses, price(b, k0, tau))
        rows.append({"effort": float(b), "price": float(price(b, k0, tau)), "mean_depth": float(d.mean()),
                     "mean_loss": float(losses[d - 1, np.arange(losses.shape[1])].mean())})
    return rows


def affine_fit(x, y):
    """Least squares y = a + s x. Returns slope, intercept, R^2 (1.0 for a constant y)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    s, a = np.polyfit(x, y, 1)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 if ss_tot == 0 else 1.0 - ((y - (a + s * x)) ** 2).sum() / ss_tot
    return float(s), float(a), float(r2)


def effort_law(rows, max_depth, lo=1.5, margin=0.5):
    """P9: fit mean depth against effort over the efforts whose mean depth lies in
    [lo, max_depth - margin]; the ends bend because depth is capped at 1 and max_depth.
    Returns the fit and the efforts used (slope, intercept and r2 are None below four points)."""
    mid = [r for r in rows if lo <= r["mean_depth"] <= max_depth - margin]
    out = {"n_points": len(mid), "slope": None, "intercept": None, "r2": None, "efforts": [r["effort"] for r in mid]}
    if len(mid) >= 4:
        out["slope"], out["intercept"], out["r2"] = affine_fit([r["effort"] for r in mid],
                                                                [r["mean_depth"] for r in mid])
    return out


def marginal_gains(mean_losses):
    """gain[r-1] = loss at depth r minus loss at depth r+1, for r = 1..R-1."""
    m = np.asarray(mean_losses, dtype=float)
    return m[:-1] - m[1:]


def geometric_decay(gains):
    """Fit log gain = log g0 - r / sigma over the positive gains. Returns sigma, g0, R^2;
    None entries when fewer than three gains are positive."""
    g = np.asarray(gains, dtype=float)
    r = np.arange(1, g.size + 1, dtype=float)
    ok = g > 0
    if ok.sum() < 3:
        return {"sigma": None, "g0": None, "r2": None}
    s, a, r2 = affine_fit(r[ok], np.log(g[ok]))
    return {"sigma": -1.0 / s if s < 0 else None, "g0": math.exp(a), "r2": r2}


def interpolate_loss(points, depth):
    """Loss of a frontier at a given mean depth by linear interpolation. points: (mean_depth, loss)
    pairs; None when depth is outside their range."""
    pts = sorted(points)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if not xs[0] <= depth <= xs[-1]:
        return None
    return float(np.interp(depth, xs, ys))
