import numpy as np
import pytest
import torch
from scipy.special import erf

from clm.block import ConnectomeFFN, DenseFFN, connectome_param_count, matched_hidden
from clm.graphs import ARMS, edge_usage, forward_reach, make_wiring

D, K = 16, 8


def make_block(region, arm="real", steps=4, leak=0.5, bias_scale=0.1):
    w = make_wiring(region, arm, 0, K)
    b = ConnectomeFFN(D, w, steps, leak)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        b.values.copy_(torch.randn(w.nnz, generator=g) * b.init_std)
        b.bias.copy_(torch.randn(w.n, generator=g) * bias_scale)
        b.w_in.weight.copy_(torch.randn(b.w_in.weight.shape, generator=g) * 0.3)
        b.w_out.weight.copy_(torch.randn(b.w_out.weight.shape, generator=g) * 0.3)
    return b, w


def reference(b, w, x):
    """The block's equations written out with a dense matrix, one token at a time."""
    a = np.zeros((w.n, w.n))
    a[w.post, w.pre] = b.values.detach().double().numpy()    # row = receiving neuron
    bias = b.bias.detach().double().numpy()
    w_in = b.w_in.weight.detach().double().numpy()
    w_out = b.w_out.weight.detach().double().numpy()
    gelu = lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2)))
    norm = lambda v: v / np.sqrt(np.mean(v ** 2) + 1e-6)     # over all neurons
    out = []
    for tok in x.double().numpy():
        drive = bias.copy()
        drive[w.inp] += w_in @ tok
        h = norm(b.leak * gelu(drive))
        for _ in range(b.steps - 1):
            h = norm((1 - b.leak) * h + b.leak * gelu(a @ h + drive))
        out.append(w_out @ h[w.out])
    return torch.from_numpy(np.array(out))


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("steps", [2, 3, 4])
def test_block_matches_the_dense_reference(region, arm, steps):
    b, w = make_block(region, arm, steps)
    x = torch.randn(5, D, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(b(x).double(), reference(b, w, x), rtol=1e-4, atol=1e-5)


def test_block_never_mixes_tokens(region):
    b, _ = make_block(region)
    x = torch.randn(2, 7, D, generator=torch.Generator().manual_seed(3))
    whole = b(x)
    one_by_one = torch.stack([torch.stack([b(x[i, t:t + 1])[0] for t in range(7)]) for i in range(2)])
    torch.testing.assert_close(whole, one_by_one, rtol=1e-5, atol=1e-6)


def test_both_blocks_run_in_float32_under_bf16_autocast(region):
    b, _ = make_block(region)
    dense = DenseFFN(D, 32)
    x = torch.randn(4, D).bfloat16()
    for block in (b, dense):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = block(x)
        assert y.dtype == torch.float32
        torch.testing.assert_close(y, block(x.float()), rtol=0, atol=0)


def test_changing_only_the_wiring_changes_the_output(region):
    real, _ = make_block(region, "real")
    shuf, _ = make_block(region, "shuf")
    assert torch.equal(real.values, shuf.values)       # same draws, same order
    x = torch.randn(6, D, generator=torch.Generator().manual_seed(4))
    assert (real(x) - shuf(x)).abs().max() > 1e-3


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("steps", [3, 4])
def test_first_step_gradient_reaches_exactly_the_edges_edge_usage_counts(region, arm, steps):
    b, w = make_block(region, arm, steps, bias_scale=0.0)   # zero bias: only reached neurons lit
    x = torch.randn(64, D, generator=torch.Generator().manual_seed(5))
    b(x).pow(2).sum().backward()
    got = (b.values.grad != 0).numpy()
    expected = forward_reach(w, steps - 2)[w.pre]
    np.testing.assert_array_equal(got, expected)
    assert got.mean() == pytest.approx(edge_usage(w, steps))


@pytest.mark.parametrize("steps", [2, 3, 4])
def test_with_zero_bias_only_forward_reachable_readouts_are_active(region, steps):
    b, w = make_block(region, "real", steps, bias_scale=0.0)
    x = torch.randn(32, D, generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        lit = (b.state(x)[b.out] != 0).numpy()
    reached = forward_reach(w, steps - 1)[w.out]
    # unreached readouts must be exactly zero on every token; a reached one can
    # still read 0 on some tokens, because float32 gelu(z) is exactly 0 below
    # about z = -5.3 (1 + erf rounds to 0), so it must be lit on at least one
    assert not lit[~reached].any()
    np.testing.assert_array_equal(lit.any(axis=1), reached)


def test_parameter_count_formula_matches_the_module(region):
    b, w = make_block(region)
    assert connectome_param_count(D, w) == sum(p.numel() for p in b.parameters())


def test_matched_dense_ffn_is_within_half_a_hidden_unit(region):
    _, w = make_block(region)
    target = connectome_param_count(D, w)
    dense = DenseFFN(D, matched_hidden(D, target))
    assert abs(sum(p.numel() for p in dense.parameters()) - target) <= D + 0.5


def test_fewer_than_two_steps_is_rejected(region):
    w = make_wiring(region, "real", 0, K)
    with pytest.raises(ValueError, match="steps"):
        ConnectomeFFN(D, w, 1, 0.5)
