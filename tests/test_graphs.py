import numpy as np
import pytest

from clm.graphs import (ARMS, Wiring, bad_edges, edge_usage, forward_reach, load_region,
                        make_wiring, readout_reach, select_io)

K = 8


def degrees(w):
    return np.bincount(w.pre, minlength=w.n), np.bincount(w.post, minlength=w.n)


def edge_set(w):
    return set(zip(w.pre.tolist(), w.post.tolist()))


def test_shuffle_keeps_every_neurons_in_and_out_degree(region):
    real, shuf = make_wiring(region, "real", 0, K), make_wiring(region, "shuf", 0, K)
    for a, b in zip(degrees(real), degrees(shuf)):
        np.testing.assert_array_equal(a, b)


def test_shuffle_moves_most_edges(region):
    real, shuf = make_wiring(region, "real", 0, K), make_wiring(region, "shuf", 0, K)
    assert len(edge_set(real) & edge_set(shuf)) / real.nnz < 0.5


def test_no_arm_has_self_loops_or_duplicate_edges(region):
    for arm in ARMS:
        w = make_wiring(region, arm, 0, K)
        assert not bad_edges(w.n, w.pre, w.post).any(), arm


def test_every_sparse_arm_has_the_real_edge_count(region):
    n_real = make_wiring(region, "real", 0, K).nnz
    sparse = [a for a in ARMS if a != "dense"]
    assert [make_wiring(region, a, 0, K).nnz for a in sparse] == [n_real] * len(sparse)


def test_dense_arm_is_fully_connected_and_matches_the_regions_parameters(region):
    real, dense = make_wiring(region, "real", 0, K), make_wiring(region, "dense", 0, K)
    assert dense.nnz == dense.n * (dense.n - 1)           # every pair, no self-loops
    target = real.nnz + real.n                             # weights + biases
    assert abs((dense.nnz + dense.n) - target) <= dense.n  # nearest whole neuron count
    assert dense.fingerprint() == make_wiring(region, "dense", 3, K).fingerprint()


def test_er_erases_the_hubs(region):
    real, er = make_wiring(region, "real", 0, K), make_wiring(region, "er", 0, K)
    assert degrees(er)[1].max() < 0.5 * degrees(real)[1].max()


def test_same_seed_repeats_the_wiring_and_a_new_seed_changes_it(region):
    for arm in ("shuf", "er", "real_randio"):
        a, b, c = (make_wiring(region, arm, s, K) for s in (0, 0, 1))
        assert a.fingerprint() == b.fingerprint(), arm
        assert a.fingerprint() != c.fingerprint(), arm


def test_real_wiring_does_not_depend_on_seed(region):
    assert (make_wiring(region, "real", 0, K).fingerprint()
            == make_wiring(region, "real", 5, K).fingerprint())


def test_fingerprints_differ_between_arms(region):
    assert len({make_wiring(region, a, 0, K).fingerprint() for a in ARMS}) == len(ARMS)


def test_degree_io_uses_lowest_in_degree_for_input_and_highest_for_readout(region):
    w = make_wiring(region, "real", 0, K)
    indeg = np.bincount(w.post, minlength=w.n)
    ranked = np.sort(indeg)
    assert indeg[w.inp].max() <= ranked[K - 1]
    assert indeg[w.out].min() >= ranked[-K]
    assert len(w.inp) == len(w.out) == K and not set(w.inp) & set(w.out)


def test_random_io_is_disjoint_and_differs_from_degree_io(region):
    w, d = make_wiring(region, "real_randio", 0, K), make_wiring(region, "real", 0, K)
    assert len(w.inp) == len(w.out) == K and not set(w.inp) & set(w.out)
    assert set(w.out) != set(d.out)
    assert edge_set(w) == edge_set(d)


def test_edges_are_sorted_by_post_then_pre(region):
    # torch is told the COO indices are coalesced; unsorted indices give wrong sums
    for arm in ARMS:
        w = make_wiring(region, arm, 0, K)
        assert np.all(np.diff(w.post * w.n + w.pre) > 0), arm


def test_unknown_arm_is_rejected(region):
    with pytest.raises(ValueError, match="unknown arm"):
        make_wiring(region, "standard_ffn", 0, K)


def test_region_with_a_duplicate_edge_is_rejected(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, root_id=np.arange(3), pre=np.array([0, 0]), post=np.array([1, 1]))
    with pytest.raises(ValueError, match="duplicate"):
        load_region(p)


def test_io_needs_enough_neurons():
    with pytest.raises(ValueError, match="distinct neurons"):
        select_io(10, np.array([1, 2]), 6)


def chain():
    # 0 -> 1 -> 2 -> 3, input {0}, readout {3}
    return Wiring("real", 0, 5, np.array([0, 1, 2]), np.array([1, 2, 3]),
                  np.array([0]), np.array([3]))


def test_forward_reach_counts_hops_out_from_the_input():
    w = chain()
    assert forward_reach(w, 0).nonzero()[0].tolist() == [0]
    assert forward_reach(w, 2).nonzero()[0].tolist() == [0, 1, 2]


def test_edge_usage_and_readout_reach_on_the_chain():
    w = chain()
    # an edge is active on the first step if its pre neuron is lit within steps-2 hops
    assert edge_usage(w, 3) == pytest.approx(2 / 3)    # pre in {0, 1}: edges 0->1, 1->2
    assert edge_usage(w, 4) == 1.0                     # pre in {0, 1, 2}: all three
    assert readout_reach(w, 3) == 0.0      # input reaches node 2 in 2 hops, not 3
    assert readout_reach(w, 4) == 1.0
