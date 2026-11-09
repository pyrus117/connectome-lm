import json

import numpy as np
import pytest
import torch

import clm.s2train as S
from clm import rules2k2 as R
from clm.countread import N_GATE_BINS
from clm.s2model import B0Config, build_b0, param_groups
from clm.s2train import run
from test_s2count import CPU, KK, TINY, gated_inputs, randomised, setup  # noqa: F401  (setup is a fixture)

ORDERS = B0Config(vocab=64, ctx=16, d_model=16, n_layer=3, n_head=2, count_scale=4)
GATED = B0Config(vocab=64, ctx=16, d_model=16, n_layer=3, n_head=2, count_orders=4)


def orders_copy(m):
    s = build_b0(ORDERS, 0)
    got = s.load_state_dict(m.state_dict(), strict=False)
    assert set(got.missing_keys) == {"count_scale"} and not got.unexpected_keys
    return s


def test_a_fresh_per_order_scale_reads_exactly_as_plain_count_and_has_one_parameter_per_order():
    s = orders_copy(randomised())
    assert s.count_scale.shape == (4,) and torch.equal(s.count_scale, torch.ones(4))
    x = torch.randint(0, 64, (2, 16))
    tok, score, _ = gated_inputs()
    assert torch.equal(s(x, count=(tok, score))[0], randomised()(x, count=(tok, score))[0])


def test_each_order_s_slots_are_scaled_by_that_order_s_own_scalar():
    m = randomised()
    s = orders_copy(m)
    with torch.no_grad():
        s.count_scale.copy_(torch.tensor([0.5, 2.0, 3.0, 0.0]))
    x = torch.randint(0, 64, (2, 16))
    tok, score, _ = gated_inputs()
    by_order = torch.tensor([0.5, 2.0, 3.0, 0.0]).repeat_interleave(KK)           # slots are grouped order by order
    assert torch.allclose(s(x, count=(tok, score))[0], m(x, count=(tok, score * by_order))[0], atol=1e-5)
    with pytest.raises(ValueError, match="cannot split"):
        s(x, count=(tok[..., :KK * 4 - 1], score[..., :KK * 4 - 1]))


def test_every_order_scalar_a_read_used_gets_gradient():
    s = orders_copy(randomised())
    x = torch.randint(0, 64, (2, 16))
    tok, score, _ = gated_inputs()
    score[..., 2 * KK:3 * KK] = 0                                                 # order 4 contributes nothing
    s(x, x, count=(tok, score))[1].backward()
    assert s.count_scale.grad[[0, 1, 3]].abs().min() > 0 and s.count_scale.grad[2] == 0


def groups_of(model, **kw):
    opts = param_groups(model, "muon", 0.008, 0.1, adam_mult=8.0, **kw)
    return {name: next(g for o in opts for g in o.param_groups if any(p is q for p in g["params"]))
            for name, q in model.named_parameters()}


def test_a_gate_rate_moves_the_gate_parameters_to_their_own_group_and_leaves_everything_else():
    g = build_b0(GATED, 0)
    base, low = groups_of(g), groups_of(g, gate_mult=1.0)
    for name in ("count_gate", "count_absent"):
        assert base[name]["lr"] == pytest.approx(0.064)                            # no gate_mult: with the rest, as Stages 2E to 2K
        assert low[name]["lr"] == pytest.approx(0.008) and low[name]["lr_mult"] == 1.0 and low[name]["weight_decay"] == 0.0
    assert low["head.weight"]["lr"] == pytest.approx(0.064) and low["embed.weight"]["lr"] == pytest.approx(0.064)
    assert low["count_gate"] is low["count_absent"] and low["count_gate"] is not low["head.weight"]
    for name, grp in base.items():
        if name not in ("count_gate", "count_absent"):
            assert low[name]["lr"] == grp["lr"] and low[name]["weight_decay"] == grp["weight_decay"], name
    s = build_b0(ORDERS, 0)
    assert groups_of(s, gate_mult=0.5)["count_scale"]["lr"] == pytest.approx(0.004)


def test_a_model_with_no_gate_parameters_ignores_a_gate_rate():
    p = build_b0(TINY, 0)
    assert groups_of(p, gate_mult=1.0)["head.weight"]["lr"] == pytest.approx(0.064)


def test_the_gate_rate_reaches_training_and_a_zero_rate_freezes_the_gate(setup):
    tcfg, tmp = setup
    base = S.dataclasses.replace(tcfg, count_gate=True, optimizer="muon", lr=3e-3, adam_mult=4.0)
    run("T", 0, TINY, S.dataclasses.replace(base, gate_mult=0.0), tmp / "frozen", CPU)
    run("T", 0, TINY, base, tmp / "moving", CPU)
    got = {n: json.loads((tmp / n / "results.json").read_text()) for n in ("frozen", "moving")}
    assert np.allclose(got["frozen"]["count_gate"], 1.0) and not np.allclose(got["moving"]["count_gate"], 1.0)
    assert got["frozen"]["train"]["gate_mult"] == 0.0 and got["moving"]["train"]["gate_mult"] is None
    assert np.allclose(got["frozen"]["count_absent_norm"], 0.0) and max(got["moving"]["count_absent_norm"]) > 0


def test_a_per_order_scaled_run_names_its_arm_trains_four_scalars_and_reports_them(setup):
    tcfg, tmp = setup
    run("T", 0, TINY, S.dataclasses.replace(tcfg, count_scale_orders=True), tmp / "o", CPU)
    got = json.loads((tmp / "o" / "results.json").read_text())
    assert got["arm"] == "count_oscaled" and got["model"]["count_scale"] == 4
    assert len(got["count_scale"]) == 4 and not np.allclose(got["count_scale"], 1.0) and got["count_gate"] is None


def test_arms_that_cannot_be_combined_and_a_gate_rate_without_a_gate_are_refused(setup):
    tcfg, tmp = setup
    with pytest.raises(ValueError, match="separate arms"):
        run("T", 0, TINY, S.dataclasses.replace(tcfg, count_scale=True, count_scale_orders=True), tmp / "a", CPU)
    with pytest.raises(ValueError, match="separate arms"):
        run("T", 0, TINY, S.dataclasses.replace(tcfg, count_gate=True, count_scale_orders=True), tmp / "b", CPU)
    with pytest.raises(ValueError, match="gate_mult"):
        run("T", 0, TINY, S.dataclasses.replace(tcfg, gate_mult=1.0), tmp / "c", CPU)
    with pytest.raises(ValueError, match="count_scale"):
        run("T", 0, TINY, S.dataclasses.replace(tcfg, memory="none", count_scale_orders=True), tmp / "d", CPU)


def losses(count, **arms):
    return {"count": dict(enumerate(count)), **{a: dict(enumerate(v)) for a, v in arms.items()}}


COUNT = [3.24, 3.29, 3.30, 3.26, 3.28]
BETTER = [x - 0.02 for x in COUNT]


def test_each_candidate_read_is_judged_against_count_by_the_stage_one_rule_on_five_seeds():
    loss = losses(COUNT, gated_low=BETTER, scaled_low=[x + 0.02 for x in COUNT],
                  oscaled_low=[x - 0.02 if i else x + 0.001 for i, x in enumerate(COUNT)])
    v = R.versus_count(loss)
    assert v == {"gated_low": "X_BETTER", "scaled_low": "X_WORSE", "oscaled_low": "INCONCLUSIVE"}


def test_the_read_carried_forward_is_the_best_that_beats_count_and_plain_count_if_none_does():
    both = losses(COUNT, gated_low=BETTER, scaled_low=[x - 0.03 for x in COUNT], oscaled_low=COUNT)
    assert R.carry_forward(both) == "scaled_low"                                   # lowest mean among the X_BETTER arms
    one = losses(COUNT, gated_low=BETTER, scaled_low=COUNT, oscaled_low=[x + 0.05 for x in COUNT])
    assert R.carry_forward(one) == "gated_low"
    none = losses(COUNT, gated_low=[x - 0.004 for x in COUNT], scaled_low=COUNT, oscaled_low=COUNT)
    assert R.carry_forward(none) == "count" and not R.p20(none)
    assert R.p20(one)


def test_p20_and_the_carry_forward_do_not_count_an_arm_that_wins_on_average_but_loses_a_seed():
    mostly = losses(COUNT, gated_low=[x - 0.03 for x in COUNT[:4]] + [COUNT[4] + 0.001], scaled_low=COUNT, oscaled_low=COUNT)
    assert R.versus_count(mostly)["gated_low"] == "INCONCLUSIVE" and R.carry_forward(mostly) == "count"
