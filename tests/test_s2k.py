import json
import math

import pytest
import torch

import clm.s2train as S
from clm import rules2k as R
from clm.s2model import B0Config, build_b0, param_groups
from clm.s2train import run
from test_s2count import CPU, KK, TINY, gated_inputs, randomised, setup  # noqa: F401  (setup is a fixture)

SCALED = B0Config(vocab=64, ctx=16, d_model=16, n_layer=3, n_head=2, count_scale=True)


def scaled_copy(m):
    s = build_b0(SCALED, 0)
    got = s.load_state_dict(m.state_dict(), strict=False)
    assert set(got.missing_keys) == {"count_scale"} and not got.unexpected_keys
    return s


def test_a_fresh_scaled_model_reads_exactly_as_plain_count_and_has_one_extra_parameter():
    fresh, plain = build_b0(SCALED, 0), dict(build_b0(TINY, 0).named_parameters())
    for name, p in fresh.named_parameters():
        if name in plain:
            assert torch.equal(p, plain[name]), name
    assert fresh.count_scale.shape == () and fresh.count_scale.item() == 1.0
    assert fresh.non_embedding_params() == build_b0(TINY, 0).non_embedding_params() + 1
    m = randomised()
    s = scaled_copy(m)
    x = torch.randint(0, 64, (2, 16))
    tok, score, _ = gated_inputs()
    assert torch.equal(s(x, count=(tok, score))[0], m(x, count=(tok, score))[0])


def test_the_scalar_multiplies_every_score_and_nothing_else():
    m = randomised()
    s = scaled_copy(m)
    with torch.no_grad():
        s.count_scale.fill_(2.5)
    x = torch.randint(0, 64, (2, 16))
    tok, score, _ = gated_inputs()
    assert torch.allclose(s(x, count=(tok, score))[0], m(x, count=(tok, score * 2.5))[0], atol=1e-5)
    assert torch.equal(s(x)[0], m(x)[0])                        # no count read, no effect


def test_the_scalar_gets_gradient_from_a_read_and_a_scaled_model_refuses_a_gated_read():
    s = scaled_copy(randomised())
    x = torch.randint(0, 64, (2, 16))
    tok, score, state = gated_inputs()
    s(x, x, count=(tok, score))[1].backward()
    assert s.count_scale.grad is not None and s.count_scale.grad.abs() > 0
    with pytest.raises(ValueError, match="gated"):
        s(x, count=(tok, score, state))


def test_the_scalar_trains_in_the_adamw_group_at_the_scaled_rate_with_no_weight_decay():
    s = build_b0(SCALED, 0)
    opts = param_groups(s, "muon", 0.008, 0.1, adam_mult=8.0)
    groups = [g for o in opts for g in o.param_groups if any(p is s.count_scale for p in g["params"])]
    assert len(groups) == 1 and groups[0]["lr"] == pytest.approx(0.064) and groups[0]["weight_decay"] == 0.0
    assert not any(p is s.count_scale for g in opts[0].param_groups for p in g["params"])      # not Muon's


def test_a_config_that_sets_both_read_changes_is_refused():
    with pytest.raises(ValueError, match="separate arms"):
        build_b0(B0Config(vocab=64, ctx=16, d_model=16, n_layer=3, n_head=2, count_orders=4, count_scale=True), 0)


def test_a_scaled_run_trains_its_scalar_names_its_arm_and_reports_it(setup):
    tcfg, tmp = setup
    run("T", 0, TINY, S.dataclasses.replace(tcfg, count_scale=True), tmp / "s", CPU)
    run("T", 0, TINY, tcfg, tmp / "c", CPU)
    got = {n: json.loads((tmp / n / "results.json").read_text()) for n in "sc"}
    assert [got[n]["arm"] for n in "sc"] == ["count_scaled", "count"]
    assert got["s"]["model"]["count_scale"] == 1 and got["c"]["model"]["count_scale"] == 0
    assert isinstance(got["s"]["count_scale"], float) and got["s"]["count_scale"] != 1.0
    assert got["c"]["count_scale"] is None and got["s"]["count_gate"] is None
    assert math.isfinite(got["s"]["final_val_loss"])         # after six steps the scalar's effect on the loss is below float32 resolution


def test_a_scalar_without_a_count_memory_alongside_a_gate_or_set_by_the_size_config_is_refused(setup):
    tcfg, tmp = setup
    with pytest.raises(ValueError, match="count_scale"):
        run("T", 0, TINY, S.dataclasses.replace(tcfg, memory="none", count_scale=True), tmp / "x", CPU)
    with pytest.raises(ValueError, match="separate arms"):
        run("T", 0, TINY, S.dataclasses.replace(tcfg, count_gate=True, count_scale=True), tmp / "y", CPU)
    with pytest.raises(ValueError, match="size config"):
        run("T", 0, SCALED, tcfg, tmp / "z", CPU)


def losses(count, gated, scaled):
    return {"count": dict(enumerate(count)), "gated": dict(enumerate(gated)), "scaled": dict(enumerate(scaled))}


def test_p19_needs_the_gated_read_to_beat_count_by_the_sesoi_on_every_seed():
    assert R.p19(losses([3.30, 3.31, 3.30], [3.28, 3.29, 3.28], [3.30, 3.31, 3.30]))
    assert not R.p19(losses([3.30, 3.31, 3.30], [3.295, 3.30, 3.295], [3.30, 3.31, 3.30]))      # mean 0.0067
    assert not R.p19(losses([3.30, 3.31, 3.30], [3.28, 3.29, 3.31], [3.30, 3.31, 3.30]))        # one seed worse
    assert not R.p19(losses([3.28, 3.29, 3.28], [3.30, 3.31, 3.30], [3.30, 3.31, 3.30]))        # gate worse


def test_the_reading_separates_structure_from_scale():
    count = [3.30, 3.31, 3.30]
    gated = [3.28, 3.29, 3.28]
    assert R.reading(losses(count, gated, [3.30, 3.31, 3.30])) == "STRUCTURE"                     # scalar does nothing
    assert R.reading(losses(count, gated, [3.285, 3.295, 3.285])) == "SCALE"                      # scalar gets nearly all of it
    assert R.reading(losses(count, [3.29, 3.30, 3.29], [3.30, 3.31, 3.30])) == "NOT_HOLDING"        # 0.01 mean needs every seed below
    assert R.reading(losses(count, [3.288, 3.298, 3.288], [3.294, 3.304, 3.294])) == "UNRESOLVED"   # gate -0.012, scalar -0.006, gate vs scalar -0.006
    assert R.reading(losses(count, gated, [3.27, 3.28, 3.27])) == "SCALE"                         # scalar better than the gate still reads as scale


def test_the_five_seed_rule_fires_once_on_three_seeds_for_an_inconclusive_contrast_under_two_sesoi():
    count = [3.30, 3.31, 3.30]
    mixed = [3.27, 3.29, 3.305]                                                                    # d = -0.03, -0.02, +0.005: mean -0.015, one seed worse
    assert R.wants_seeds(losses(count, mixed, count))                                              # gated vs count INCONCLUSIVE
    assert R.wants_seeds(losses(count, [3.28, 3.29, 3.28], [3.27, 3.29, 3.30]))                    # gated vs scaled INCONCLUSIVE, mean -0.0033
    assert not R.wants_seeds(losses(count, [3.28, 3.29, 3.28], count))                             # both resolved
    assert not R.wants_seeds(losses(count, [3.20, 3.31, 3.30], count))                             # INCONCLUSIVE but the mean gap is -0.033, over 2 x SESOI
    five = {a: {**v, 3: v[0], 4: v[0]} for a, v in losses(count, mixed, count).items()}
    assert not R.wants_seeds(five)                                                                  # only decided on three seeds
