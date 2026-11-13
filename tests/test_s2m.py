import pytest
import torch

import clm.rules2m as R
import clm.s2train as S
from clm.s2loop import build_loop
from clm.s2model import build_b0, param_groups
from clm.s2train import run
from test_s2count import CPU, setup  # noqa: F401  (setup is a fixture)
from test_s2l import TINY6


def groups_of(model, **kw):
    opts = param_groups(model, "muon", 0.008, 0.1, adam_mult=8.0, **kw)
    return {name: next(g for o in opts for g in o.param_groups if any(p is q for p in g["params"]))
            for name, q in model.named_parameters()}


def test_a_gain_rate_moves_the_loop_gain_to_its_own_group_and_leaves_everything_else():
    m = build_loop(TINY6, 0, 6, gain=True)
    base, low = groups_of(m), groups_of(m, gain_mult=1.0)
    assert base["loop_gain"]["lr"] == pytest.approx(0.064)                       # no gain_mult: with the rest, as Stage 2L
    g = low["loop_gain"]
    assert g["lr"] == pytest.approx(0.008) and g["lr_mult"] == 1.0 and g["weight_decay"] == 0.0
    assert low["head.weight"]["lr"] == pytest.approx(0.064) and low["embed.weight"]["lr"] == pytest.approx(0.064)
    assert g is not low["head.weight"] and g is not low["blocks.2.qkv.weight"]
    for name, grp in base.items():
        if name != "loop_gain":
            assert low[name]["lr"] == grp["lr"] and low[name]["weight_decay"] == grp["weight_decay"], name
    assert groups_of(m, gain_mult=0.5)["loop_gain"]["lr"] == pytest.approx(0.004)


def test_a_model_with_no_loop_gain_ignores_a_gain_rate():
    p = build_b0(TINY6, 0)
    assert groups_of(p, gain_mult=1.0)["head.weight"]["lr"] == pytest.approx(0.064)


def test_the_gain_rate_reaches_training_and_a_zero_rate_freezes_the_gain(setup):
    tcfg, tmp = setup
    base = S.dataclasses.replace(tcfg, arch="loop", loop_gain=True, loop_loss="all", optimizer="muon", lr=3e-3, adam_mult=4.0)
    run("T", 0, TINY6, S.dataclasses.replace(base, gain_mult=0.0), tmp / "frozen", CPU)
    run("T", 0, TINY6, base, tmp / "moving", CPU)
    gain = {n: torch.load(tmp / n / "ckpt.pt", map_location=CPU)["model"]["loop_gain"] for n in ("frozen", "moving")}
    assert torch.equal(gain["frozen"], torch.ones_like(gain["frozen"])) and not torch.allclose(gain["moving"], torch.ones_like(gain["moving"]))


def test_a_gain_rate_without_a_loop_gain_is_refused(setup):
    tcfg, tmp = setup
    with pytest.raises(ValueError, match="gain_mult"):
        run("T", 0, TINY6, S.dataclasses.replace(tcfg, gain_mult=1.0), tmp / "a", CPU)
    with pytest.raises(ValueError, match="gain_mult"):
        run("T", 0, TINY6, S.dataclasses.replace(tcfg, arch="loop", gain_mult=1.0), tmp / "b", CPU)


def summ(by_depth):
    return R.summary(by_depth)


def test_the_loop_s_benefit_is_the_depth_one_loss_minus_the_best_depth_loss():
    assert R.benefit(summ([3.40, 3.36, 3.35, 3.36, 3.37, 3.38])) == pytest.approx(0.05)
    assert R.benefit(summ([3.40, 3.401, 3.402])) == 0.0                               # depth 1 is best: nothing bought
    assert R.benefit(summ([3.4000, 3.39995, 3.39998])) == pytest.approx(0.00005)      # Stage 2L's inert loop


def test_the_loop_is_alive_only_if_every_seed_earns_the_sesoi():
    live, dead = summ([3.40, 3.38, 3.37]), summ([3.40, 3.395, 3.392])
    assert R.alive({0: live, 1: live, 2: live})
    assert not R.alive({0: live, 1: live, 2: dead})
    assert R.alive({0: summ([3.40, 3.3899])}) and not R.alive({0: summ([3.40, 3.3901])})   # either side of 0.01


def test_the_stage_stops_after_seed_zero_only_if_neither_loop_is_alive():
    live, dead = summ([3.40, 3.38, 3.37]), summ([3.40, 3.40, 3.40])
    assert R.futile(dead, dead)
    assert not R.futile(live, dead) and not R.futile(dead, live) and not R.futile(live, live)
