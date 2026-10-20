import numpy as np
import pytest
import torch
from scipy.special import erf

from clm.block import SharedBrain, brain_param_count
from clm.graphs import ARMS, edge_usage, forward_reach, make_wiring

K = 8


def make_brain(region, arm="real", norm=True, bias_scale=0.1):
    w = make_wiring(region, arm, 0, K)
    b = SharedBrain(w, 0.5, norm)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        b.values.copy_(torch.randn(w.nnz, generator=g) * b.init_std)
        b.bias.copy_(torch.randn(w.n, generator=g) * bias_scale)
    return b, w


def writes(layers, tokens, seed=2):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(tokens, K, generator=g) for _ in range(layers)]


def run_chain(b, ws):
    h, reads = None, []
    for w in ws:
        h = b.step(h, w)
        reads.append(b.read(h))
    return h, reads


def reference(b, w, ws):
    """The brain's equations with a dense matrix, one token at a time."""
    a = np.zeros((w.n, w.n))
    a[w.post, w.pre] = b.values.detach().double().numpy()    # row = receiving neuron
    bias = b.bias.detach().double().numpy()
    gelu = lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2)))
    norm = (lambda v: v / np.sqrt(np.mean(v ** 2) + 1e-6)) if b.norm else (lambda v: v)
    out = np.zeros((len(ws), ws[0].shape[0], K))
    for t in range(ws[0].shape[0]):
        h = None
        for layer, write in enumerate(ws):
            drive = bias.copy()
            drive[w.inp] += write[t].double().numpy()
            h = b.leak * gelu(drive) if h is None else (1 - b.leak) * h + b.leak * gelu(a @ h + drive)
            h = norm(h)
            out[layer, t] = h[w.out]
    return torch.from_numpy(out)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("layers", [1, 3, 6])
@pytest.mark.parametrize("norm", [True, False])
def test_brain_matches_the_dense_reference_layer_by_layer(region, arm, layers, norm):
    b, w = make_brain(region, arm, norm)
    ws = writes(layers, 5)
    _, reads = run_chain(b, ws)
    torch.testing.assert_close(torch.stack(reads).double(), reference(b, w, ws), rtol=1e-4, atol=1e-5)


def test_tokens_never_share_brain_state(region):
    b, _ = make_brain(region)
    ws = writes(4, 7)
    _, together = run_chain(b, ws)
    for t in range(7):
        _, alone = run_chain(b, [w[t:t + 1] for w in ws])
        for layer in range(4):
            torch.testing.assert_close(together[layer][t:t + 1], alone[layer], rtol=1e-5, atol=1e-6)


def test_first_layer_step_does_not_use_the_wiring(region):
    b, _ = make_brain(region)
    h = b.step(None, writes(1, 4)[0])
    h.pow(2).sum().backward()
    assert b.values.grad is None or not b.values.grad.any()


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("layers", [3, 6])
def test_first_step_gradient_reaches_exactly_the_edges_edge_usage_counts(region, arm, layers):
    b, w = make_brain(region, arm, bias_scale=0.0)   # zero bias: only reached neurons lit
    _, reads = run_chain(b, writes(layers, 64, seed=5))
    g = torch.Generator().manual_seed(9)
    loss = sum((r @ torch.randn(K, 4, generator=g)).pow(2).sum() for r in reads)
    loss.backward()
    got = (b.values.grad != 0).numpy()
    np.testing.assert_array_equal(got, forward_reach(w, layers - 2)[w.pre])
    assert got.mean() == pytest.approx(edge_usage(w, layers))


@pytest.mark.parametrize("layers", [2, 3, 4])
def test_with_zero_bias_only_forward_reachable_readouts_are_lit(region, layers):
    b, w = make_brain(region, "real", bias_scale=0.0)
    with torch.no_grad():
        h, _ = run_chain(b, writes(layers, 32, seed=6))
    lit = (h[b.out] != 0).numpy()
    reached = forward_reach(w, layers - 1)[w.out]
    # unreached readouts must be exactly zero on every token; a reached one can
    # still read 0 on some tokens, because float32 gelu(z) is exactly 0 below
    # about z = -5.3 (1 + erf rounds to 0), so it must be lit on at least one
    assert not lit[~reached].any()
    np.testing.assert_array_equal(lit.any(axis=1), reached)


def test_parameter_count_formula_matches_the_module(region):
    for arm in ARMS:
        b, w = make_brain(region, arm)
        assert brain_param_count(w) == sum(p.numel() for p in b.parameters()), arm
