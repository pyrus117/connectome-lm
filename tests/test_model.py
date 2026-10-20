import torch

from clm.graphs import make_wiring
from clm.model import ModelConfig, RUN_ARMS, build_model


def cfg(**kw):
    base = dict(vocab=64, ctx=16, d_model=16, n_layer=4, n_head=2, leak=0.5, k_io=8)
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


def test_every_layer_uses_one_shared_brain(region):
    m = build_model(cfg(), "real", 0, region)
    assert [n for n, _ in m.named_parameters() if n.startswith("brain.")] == ["brain.values", "brain.bias"]
    assert m.brain.fingerprint == make_wiring(region, "real", 0, 8).fingerprint()


def test_each_layer_receives_the_state_the_previous_layer_left(region):
    m = build_model(cfg(), "shuf", 0, region)
    seen = {}
    for i, layer in enumerate(m.layers):
        layer.register_forward_hook(lambda mod, args, out, i=i: seen.update({i: (args[1], out[1])}))
    m(torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(2)))
    assert seen[0][0] is None                                # empty at the first layer
    for i in range(1, len(m.layers)):
        assert seen[i][0] is not None, f"layer {i} received no state"
        assert torch.equal(seen[i][0], seen[i - 1][1]), i


def test_brain_state_stays_float32_under_bf16_autocast(region):
    m = build_model(cfg(), "real", 0, region)
    seen = []
    m.layers[-1].register_forward_hook(lambda mod, args, out: seen.append(out[1].dtype))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        m(torch.randint(0, 64, (1, 16)))
    assert seen == [torch.float32]


def test_parameters_shared_by_two_arms_start_identical(region):
    real = dict(build_model(cfg(), "real", 3, region).named_parameters())
    for arm in RUN_ARMS[1:]:
        other = dict(build_model(cfg(), arm, 3, region).named_parameters())
        shared = [n for n in real if not n.startswith("brain.")]
        assert len(shared) > 10 and all(n in other for n in shared), arm
        for n in shared:
            assert torch.equal(real[n], other[n]), (arm, n)


def test_brain_values_start_from_the_same_draws_in_real_and_shuffled_arms(region):
    a = build_model(cfg(), "real", 0, region).brain.values
    b = build_model(cfg(), "shuf", 0, region).brain.values
    assert torch.equal(a, b)


def test_a_different_seed_changes_the_init(region):
    a = build_model(cfg(), "real", 0, region).tok.weight
    b = build_model(cfg(), "real", 1, region).tok.weight
    assert not torch.equal(a, b)


def test_sparse_arms_share_one_parameter_count_and_dense_is_within_one_neuron(region):
    counts = {arm: sum(p.numel() for p in build_model(cfg(), arm, 0, region).parameters())
              for arm in RUN_ARMS}
    sparse = {counts[a] for a in RUN_ARMS if a != "dense"}
    assert len(sparse) == 1
    n_dense = make_wiring(region, "dense", 0, 8).n
    assert abs(counts["dense"] - sparse.pop()) <= n_dense


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
