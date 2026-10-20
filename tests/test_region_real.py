import numpy as np

from clm.graphs import ARMS, bad_edges, edge_usage, make_wiring, readout_reach
from conftest import REAL_REGION


def wiring(arm, seed=0):
    return make_wiring(REAL_REGION, arm, seed)


def test_region_matches_its_manifest():
    w = wiring("real")
    assert (w.n, w.nnz) == (6293, 570682)


def test_real_shuffle_keeps_degrees_and_moves_over_90_percent_of_edges():
    real, shuf = wiring("real"), wiring("shuf")
    np.testing.assert_array_equal(np.bincount(real.post, minlength=real.n),
                                  np.bincount(shuf.post, minlength=shuf.n))
    np.testing.assert_array_equal(np.bincount(real.pre, minlength=real.n),
                                  np.bincount(shuf.pre, minlength=shuf.n))
    key = lambda w: w.pre * w.n + w.post
    assert np.isin(key(real), key(shuf)).mean() < 0.10
    assert not bad_edges(shuf.n, shuf.pre, shuf.post).any()


def test_real_er_has_no_hubs():
    real, er = wiring("real"), wiring("er")
    assert np.bincount(er.post).max() < 0.1 * np.bincount(real.post).max()


def test_reachability_grows_with_steps_in_every_arm():
    for arm in ARMS:
        w = wiring(arm)
        u = [edge_usage(w, s) for s in (2, 3, 4)]
        r = [readout_reach(w, s) for s in (2, 3, 4)]
        assert all(0.0 <= v <= 1.0 for v in u + r), arm
        assert u == sorted(u) and r == sorted(r), arm
