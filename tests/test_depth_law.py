import math

import numpy as np
import pytest

from clm import depth_law as D

R = 6


def geometric_losses(n=4000, sigma=1.5, seed=0):
    """Each token's gain from pass r to r+1 is g0 * exp(-r / sigma), g0 log-uniform across tokens."""
    rng = np.random.default_rng(seed)
    g0 = np.exp(rng.uniform(math.log(0.02), math.log(20.0), n))
    gains = g0[None, :] * np.exp(-np.arange(1, R) / sigma)[:, None]          # (R-1, n)
    losses = np.empty((R, n))
    losses[-1] = 3.0
    for r in range(R - 2, -1, -1):
        losses[r] = losses[r + 1] + gains[r]
    return losses


def test_price_falls_exponentially_with_effort():
    k = D.price([0, 1, 2], k0=0.2, tau=2.0)
    assert k[0] == pytest.approx(0.2)
    assert k[1] / k[0] == pytest.approx(math.exp(-0.5)) and k[2] / k[1] == pytest.approx(math.exp(-0.5))


def test_oracle_picks_the_depth_minimising_loss_plus_price_times_depth():
    losses = np.array([[3.0, 3.0, 3.0], [2.5, 2.9, 3.0], [2.4, 2.0, 3.0], [2.4, 1.9, 3.0]])   # 4 depths, 3 tokens
    assert D.oracle_depths(losses, 0.0).tolist() == [3, 4, 1]            # free depth: the lowest loss, ties shallower
    assert D.oracle_depths(losses, 0.3).tolist() == [2, 3, 1]            # each pass now costs 0.3
    assert D.oracle_depths(losses, 5.0).tolist() == [1, 1, 1]


def test_frontier_reports_mean_depth_and_the_loss_at_those_depths():
    losses = geometric_losses(n=500)
    rows = D.frontier(losses, [0, 5, 10], k0=0.2, tau=2.1715)
    depths = [r["mean_depth"] for r in rows]
    assert depths == sorted(depths) and depths[0] < depths[-1]
    d = D.oracle_depths(losses, rows[1]["price"])
    assert rows[1]["mean_loss"] == pytest.approx(losses[d - 1, np.arange(500)].mean())
    assert rows[0]["mean_loss"] > rows[-1]["mean_loss"]


def test_geometric_marginal_gains_give_an_affine_effort_law():
    losses = geometric_losses()
    rows = D.frontier(losses, range(11), k0=0.2, tau=10 / math.log(100))
    law = D.effort_law(rows, max_depth=R)
    assert law["n_points"] >= 4 and law["r2"] >= 0.9 and law["slope"] > 0


def test_flat_marginal_gains_do_not_give_one():
    """Equal gain at every pass: the best depth jumps from 1 to R at one price, so no effort has a middle depth."""
    losses = np.repeat(np.arange(R, 0, -1, dtype=float)[:, None] * 0.05, 300, axis=1)
    rows = D.frontier(losses, range(11), k0=0.2, tau=10 / math.log(100))
    assert D.effort_law(rows, max_depth=R)["slope"] is None


def test_affine_fit_recovers_a_line_and_scores_noise_low():
    x = np.arange(10)
    s, a, r2 = D.affine_fit(x, 2.0 * x + 1.0)
    assert (s, a, r2) == pytest.approx((2.0, 1.0, 1.0))
    rng = np.random.default_rng(0)
    assert D.affine_fit(x, rng.normal(size=10))[2] < 0.5


def test_geometric_decay_fit_finds_sigma():
    sigma = 1.7
    gains = 0.4 * np.exp(-np.arange(1, 6) / sigma)
    fit = D.geometric_decay(gains)
    assert fit["sigma"] == pytest.approx(sigma) and fit["r2"] == pytest.approx(1.0)
    assert D.geometric_decay([0.1, -0.1, 0.0, 0.0, 0.0])["sigma"] is None


def test_marginal_gains_and_interpolation():
    assert D.marginal_gains([3.0, 2.5, 2.4]).tolist() == pytest.approx([0.5, 0.1])
    pts = [(1.0, 3.0), (3.0, 2.0), (2.0, 2.6)]
    assert D.interpolate_loss(pts, 2.5) == pytest.approx(2.3)
    assert D.interpolate_loss(pts, 0.5) is None and D.interpolate_loss(pts, 3.5) is None
