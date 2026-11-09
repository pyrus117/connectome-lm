import numpy as np
import pytest
import torch

from clm import depth_eval as E
from clm import halting as H
from test_s2loop import opened

CPU = torch.device("cpu")
N_WIN, CTX = 7, 16


@pytest.fixture
def val():
    return np.random.default_rng(0).integers(0, 64, CTX * N_WIN + 1).astype(np.uint16)


def collect(m, val, predictors=None):
    return E.collect_signals(m, val, CTX, CTX * N_WIN, 3, CPU, predictors)


def test_signals_read_entropy_and_the_top_two_margin():
    h, g = H.signals(torch.zeros(2, 3, 8))
    assert torch.allclose(h, torch.full((2, 3), np.log(8.0)), atol=1e-6) and torch.all(g == 0)
    h, g = H.signals(torch.tensor([[[10.0, 4.0, 0.0, -3.0]]]))
    assert g.item() == pytest.approx(6.0) and h.item() == pytest.approx(0.0178, abs=1e-3)       # hand-computed: ~ p2 (1 + ln 1/p2)


def test_collect_signals_matches_the_per_depth_forward_for_every_token_in_window_order(val):
    m = opened()
    s = collect(m, val)
    losses, change = E.collect(m, val, CTX, CTX * N_WIN, 3, CPU)
    assert np.allclose(s["loss"], losses, atol=1e-5) and np.allclose(s["change"], change, atol=1e-5)
    x = torch.from_numpy(np.stack([val[i * CTX:(i + 1) * CTX] for i in range(N_WIN)]).astype(np.int64))
    for r in range(1, 5):
        ent, mar = H.signals(m(x, depth=r)[0].detach())
        assert np.allclose(s["entropy"][r - 1], ent.flatten().numpy(), atol=1e-4)
        assert np.allclose(s["margin"][r - 1], mar.flatten().numpy(), atol=1e-4)
    assert s["loss"].shape == (4, N_WIN * CTX) and s["predicted"] == {}


def test_a_predictor_is_called_for_passes_one_to_r_minus_one_with_the_state_and_signals_of_that_pass(val):
    m = opened()
    seen = []

    def spy(r, x, logits, change):
        seen.append((r, tuple(x.shape), tuple(logits.shape), tuple(change.shape)))
        return torch.full(change.shape, float(r))
    s = collect(m, val, {"spy": spy})
    assert {r for r, *_ in seen} == {1, 2, 3} and seen[0][1:] == ((3, CTX, 16), (3, CTX, 64), (3, CTX))
    assert s["predicted"]["spy"].shape == (3, N_WIN * CTX)
    assert np.array_equal(s["predicted"]["spy"][:, 0], [1.0, 2.0, 3.0])


def test_the_probe_predictor_applies_the_standardised_weights_of_the_pass_it_is_given(val):
    m = opened()
    probes = [{"mean": np.zeros(3), "std": np.ones(3), "w": np.eye(3)[k], "b": 0.5 * k} for k in range(3)]
    s = collect(m, val, {"p": H.probe_predictor(probes, use_state=False)})
    assert np.allclose(s["predicted"]["p"][0], s["entropy"][0], atol=1e-4)                  # pass 1 reads entropy
    assert np.allclose(s["predicted"]["p"][1], s["margin"][1] + 0.5, atol=1e-4)             # pass 2 reads margin + 0.5
    assert np.allclose(s["predicted"]["p"][2], s["change"][2] + 1.0, atol=1e-4)             # pass 3 reads change + 1
    state = [{"mean": np.ones(19), "std": np.full(19, 2.0), "w": np.zeros(19), "b": 0.0}] * 3
    state[0]["w"] = np.zeros(19); state[0]["w"][3] = 2.0                                     # state channel 0, after standardising
    s = collect(m, val, {"p": H.probe_predictor(state, use_state=True)})
    x = torch.from_numpy(np.stack([val[i * CTX:(i + 1) * CTX] for i in range(N_WIN)]).astype(np.int64))
    h, _ = m._stem(x)
    h = m._pass(h, 1)
    assert np.allclose(s["predicted"]["p"][0], (h[..., 0] - 1.0).flatten().detach().numpy(), atol=1e-4)


def test_collect_fit_returns_the_signals_state_and_gain_of_every_sampled_token_for_passes_one_to_r_minus_one(val):
    m = opened()
    starts = np.arange(N_WIN) * CTX
    small, state, gains = E.collect_fit(m, val, starts, CTX, 3, CPU, stride=4)
    n = N_WIN * 4                                            # positions 3, 7, 11, 15 of each window
    assert small.shape == (3, n, 3) and state.shape == (3, n, 16) and gains.shape == (3, n)
    assert state.dtype == np.float16
    s = collect(m, val)
    tok = np.concatenate([w * CTX + np.arange(3, CTX, 4) for w in range(N_WIN)])
    for r in range(3):
        assert np.allclose(gains[r], s["loss"][r, tok] - s["loss"][r + 1, tok], atol=1e-5)
        assert np.allclose(small[r, :, 0], s["entropy"][r, tok], atol=1e-4)
        assert np.allclose(small[r, :, 1], s["margin"][r, tok], atol=1e-4)
        assert np.allclose(small[r, :, 2], s["change"][r, tok], atol=1e-5)


def test_exit_depths_take_the_first_stopping_pass_and_never_read_the_last_row():
    stop = np.array([[0, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=bool)   # 4 passes, 4 tokens
    assert H.exit_depths(stop).tolist() == [3, 2, 1, 4]
    assert H.exit_depths(np.array([[0, 0], [0, 0], [1, 1]], dtype=bool)).tolist() == [3, 3]    # only the forced last pass


def test_each_rule_stops_on_the_side_of_its_threshold_that_means_done_and_a_probe_never_stops_the_last_row():
    sig = {"entropy": np.array([[0.5, 2.0]] * 3), "margin": np.array([[4.0, 1.0]] * 3), "change": np.array([[0.1, 0.9]] * 3),
           "predicted": {"probe_small": np.array([[0.001, 0.2]] * 2)}}
    assert H.stop_matrix("entropy", sig, 1.0)[0].tolist() == [True, False]          # confident (low entropy) stops
    assert H.stop_matrix("margin", sig, 2.0)[0].tolist() == [True, False]           # a wide margin stops
    assert H.stop_matrix("change", sig, 0.5)[0].tolist() == [True, False]           # a settled state stops
    p = H.stop_matrix("probe_small", sig, 0.01)                                      # predicted gain below the price stops
    assert p.shape == (3, 2) and p[:2].tolist() == [[True, False]] * 2 and not p[2].any()
    assert H.stop_matrix("entropy", sig, 1.0).shape == (3, 2)


def test_the_threshold_for_a_share_stops_about_that_share_of_the_pooled_values():
    v = np.random.default_rng(0).normal(size=10000)
    for rule in ("entropy", "margin", "change"):
        level = H.level_for_share(rule, v, 0.3)
        stopped = {"entropy": v <= level, "change": v <= level, "margin": v >= level}[rule].mean()
        assert stopped == pytest.approx(0.3, abs=0.001)


def test_ridge_recovers_a_planted_linear_target_and_shrinks_hard_on_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 5)) * [1, 5, 1, 0.2, 1]
    X[:, 4] = 3.0                                            # a constant column must not break the fit
    w = np.array([2.0, -0.5, 0.0, 1.0, 0.0])
    fit = H.ridge_fit(X, X @ w + 0.7, alphas=[1e-6, 1.0, 1e4])
    pred = ((X - fit["mean"]) / fit["std"]) @ fit["w"] + fit["b"]
    assert fit["alpha"] == 1e-6 and np.allclose(pred, X @ w + 0.7, atol=1e-4)
    noise = H.ridge_fit(rng.normal(size=(4000, 5)), rng.normal(size=4000), alphas=[1e-6, 1.0, 1e4])
    assert noise["alpha"] == 1e4 and np.abs(noise["w"]).max() < 0.01


def test_ridge_picks_alpha_on_the_last_rows_not_the_first():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 3))
    y = X[:, 0].copy()
    y[1500:] = rng.normal(size=500)                          # the held-out tail is pure noise
    assert H.ridge_fit(X, y, alphas=[1e-6, 1e5], holdout=0.25)["alpha"] == 1e5


FIXED = [(1, 3.40), (2, 3.34), (3, 3.333), (4, 3.3326), (5, 3.3326), (6, 3.3326)]


def test_region_scores_compare_a_rule_with_fixed_depth_at_equal_mean_depth_inside_the_region():
    pts = [(1.0, 3.40), (1.5, 3.30), (2.0, 3.30), (2.5, 3.29), (2.7, 3.29), (2.9, 3.28)]
    s = H.region_scores(pts, FIXED)
    assert s["n"] == 4                                        # 1.0 and 2.9 lie outside; the 2.7 edge is inside
    assert s["rule_loss"] == pytest.approx(np.mean([3.30, 3.30, 3.29, 3.29]))
    fixed = np.interp([1.5, 2.0, 2.5, 2.7], [d for d, _ in FIXED], [l for _, l in FIXED])
    assert s["fixed_loss"] == pytest.approx(fixed.mean()) and s["gap"] == pytest.approx(fixed.mean() - s["rule_loss"])
    assert H.region_scores(pts[:3], FIXED)["gap"] is None and H.region_scores(pts[:3], FIXED)["n"] == 2


def test_a_rule_beats_fixed_depth_only_if_it_is_a_sesoi_lower_on_every_seed():
    def scores(delta):
        return {s: {"rule_loss": 3.34 - dl, "fixed_loss": 3.34, "gap": dl} for s, dl in zip((0, 1, 2), delta)}
    assert H.beats_fixed(scores([0.02, 0.03, 0.02])) == "X_BETTER"
    assert H.beats_fixed(scores([0.02, 0.03, -0.001])) == "INCONCLUSIVE"
    assert H.beats_fixed(scores([0.004, 0.003, 0.002])) == "NO_MEANINGFUL_DIFFERENCE"
    assert H.beats_fixed(scores([-0.02, -0.03, -0.02])) == "X_WORSE"
    unscored = scores([0.02, 0.03, 0.02])
    unscored[1] = {"rule_loss": None, "fixed_loss": None, "gap": None}
    assert H.beats_fixed(unscored) == "NOT_SCORED"
