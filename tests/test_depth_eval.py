import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clm import depth_eval as E
from test_s2loop import TINY, opened

CPU = torch.device("cpu")


@pytest.fixture
def val():
    return np.random.default_rng(0).integers(0, 64, 16 * 7 + 1).astype(np.uint16)       # 7 windows of 16


def test_collect_returns_every_depth_for_every_token_in_window_order(val):
    m = opened()
    losses, change = E.collect(m, val, 16, n_eval=16 * 7, micro=3, device=CPU)
    assert losses.shape == change.shape == (4, 7 * 16)
    x = torch.from_numpy(val[:16].astype(np.int64))[None]
    y = torch.from_numpy(val[1:17].astype(np.int64))[None]
    for r in range(1, 5):
        want = F.cross_entropy(m(x, depth=r)[0][0], y[0], reduction="none")
        assert np.allclose(losses[r - 1, :16], want.detach().numpy(), atol=1e-5)


def test_executed_uniform_depth_equals_that_depth_s_mean_loss_and_mixed_depths_differ_from_the_prediction(val):
    m = opened()
    losses, _ = E.collect(m, val, 16, 16 * 7, 3, CPU)
    n = losses.shape[1]
    for r in (1, 3):
        got = E.executed_loss(m, val, 16, 16 * 7, 3, CPU, np.full(n, r))
        assert got == pytest.approx(losses[r - 1].mean(), abs=1e-5)
    rng = np.random.default_rng(1)
    depths = rng.integers(1, 5, n)
    predicted = losses[depths - 1, np.arange(n)].mean()
    assert abs(E.executed_loss(m, val, 16, 16 * 7, 3, CPU, depths) - predicted) > 1e-4     # cross-token effect


def test_free_exit_stops_at_the_first_pass_below_the_threshold_or_runs_to_the_end():
    change = np.array([[0.9, 0.9, 0.9], [0.5, 0.2, 0.9], [0.05, 0.1, 0.9], [0.01, 0.01, 0.9]])
    assert E.free_exit_depths(change, 0.15).tolist() == [3, 3, 4]
    assert E.free_exit_depths(change, 0.6).tolist() == [2, 2, 4]
    assert E.free_exit_depths(change, 0.0).tolist() == [4, 4, 4]
