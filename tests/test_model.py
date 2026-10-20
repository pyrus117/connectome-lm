import torch

from clm.graphs import make_wiring
from clm.model import ModelConfig, RUN_ARMS, build_model


def cfg(**kw):
    base = dict(vocab=64, ctx=16, d_model=16, n_layer=2, n_head=2, conn_layers=(0, 1),
                steps=4, leak=0.5, k_io=8)
    return ModelConfig(**{**base, **kw})


def test_logits_at_a_position_ignore_later_tokens(region):
    x = torch.randint(0, 64, (1, 16), generator=torch.Generator().manual_seed(1))
    y = x.clone()
    y[0, 10:] = (y[0, 10:] + 1) % 64
    for arm in RUN_ARMS:
        m = build_model(cfg(), arm, 0, region).eval()
        with torch.no_grad():
            a, _ = m(x)
            b, _ = m(y)
        torch.testing.assert_close(a[0, :10], b[0, :10], rtol=0, atol=1e-6)
        assert (a[0, 10] - b[0, 10]).abs().max() > 1e-4, arm


def test_parameters_shared_by_two_arms_start_identical(region):
    c = cfg(conn_layers=(1,))
    real = dict(build_model(c, "real", 3, region).named_parameters())
    for arm in ("shuf", "er", "real_randio", "dense"):
        other = dict(build_model(c, arm, 3, region).named_parameters())
        shared = [n for n in real if n in other and real[n].shape == other[n].shape
                  and not n.startswith("layers.1.ffn")]
        assert len(shared) > 10, arm
        for n in shared:
            assert torch.equal(real[n], other[n]), (arm, n)


def test_connectome_values_start_from_the_same_draws_in_real_and_shuffled_arms(region):
    a = build_model(cfg(), "real", 0, region).layers[0].ffn.values
    b = build_model(cfg(), "shuf", 0, region).layers[0].ffn.values
    assert torch.equal(a, b)


def test_a_different_seed_changes_the_init(region):
    a = build_model(cfg(), "real", 0, region).tok.weight
    b = build_model(cfg(), "real", 1, region).tok.weight
    assert not torch.equal(a, b)


def test_connectome_arms_share_one_parameter_count_and_dense_matches_it(region):
    counts = {arm: sum(p.numel() for p in build_model(cfg(), arm, 0, region).parameters())
              for arm in RUN_ARMS}
    conn = {counts[a] for a in RUN_ARMS if a != "dense"}
    assert len(conn) == 1
    assert abs(counts["dense"] - conn.pop()) <= 2 * (16 + 0.5)   # two layers, half a unit each


def test_conn_layers_hold_the_arms_wiring_and_other_layers_hold_a_standard_ffn(region):
    m = build_model(cfg(n_layer=3, conn_layers=(1,)), "shuf", 2, region)
    assert [type(l.ffn).__name__ for l in m.layers] == ["DenseFFN", "ConnectomeFFN", "DenseFFN"]
    assert m.layers[1].ffn.fingerprint == make_wiring(region, "shuf", 2, 8).fingerprint()
    assert m.layers[0].ffn.fc.out_features == 4 * 16


def test_each_arm_can_memorise_one_batch(region):
    x = torch.randint(0, 64, (4, 16), generator=torch.Generator().manual_seed(0))
    for arm in RUN_ARMS:
        m = build_model(cfg(), arm, 0, region)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        losses = []
        for _ in range(300):
            _, loss = m(x[:, :-1], x[:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < 0.25 * losses[0], (arm, losses[0], losses[-1])
