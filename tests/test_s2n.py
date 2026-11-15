import subprocess
import sys

import pytest
import torch

import clm.rules2n as R
import clm.s2train as S
from clm.s2loop import build_loop
from clm.s2model import param_groups
from clm.s2train import run
from test_s2count import CPU, setup  # noqa: F401  (setup is a fixture)
from test_s2l import TINY6


def groups_of(model, **kw):
    opts = param_groups(model, "muon", 0.008, 0.1, **kw)
    return {name: next(g for o in opts for g in o.param_groups if any(p is q for p in g["params"]))
            for name, q in model.named_parameters()}


def test_a_head_rate_moves_head_weight_to_its_own_group_and_leaves_everything_else():
    m = build_loop(TINY6, 0, 6, gain=True)
    base = groups_of(m, adam_mult=8.0, gain_mult=1.0)
    headlow = groups_of(m, adam_mult=8.0, gain_mult=1.0, head_mult=1.0)
    embedlow = groups_of(m, adam_mult=1.0, head_mult=8.0)

    assert base["head.weight"]["lr"] == pytest.approx(0.064)
    h = headlow["head.weight"]
    assert h["lr"] == pytest.approx(0.008) and h["lr_mult"] == 1.0 and h["weight_decay"] == 0.0
    assert headlow["loop_gain"]["lr"] == pytest.approx(0.008)
    assert headlow["embed.weight"]["lr"] == pytest.approx(0.064)
    assert headlow["value_embed.0.weight"]["lr"] == pytest.approx(0.064)
    assert headlow["value_embed.1.weight"]["lr"] == pytest.approx(0.064)
    assert headlow["blocks.0.ve_gate"]["lr"] == pytest.approx(0.064)
    assert headlow["blocks.5.ve_gate"]["lr"] == pytest.approx(0.064)
    assert h is not headlow["embed.weight"] and h is not headlow["loop_gain"]
    for name, grp in base.items():
        if name != "head.weight":
            assert headlow[name]["lr"] == grp["lr"] and headlow[name]["weight_decay"] == grp["weight_decay"], name

    assert embedlow["head.weight"]["lr"] == pytest.approx(0.064) and embedlow["head.weight"]["lr_mult"] == 8.0
    assert embedlow["head.weight"]["weight_decay"] == 0.0
    for name in ("loop_gain", "embed.weight", "value_embed.0.weight", "value_embed.1.weight", "blocks.0.ve_gate", "blocks.5.ve_gate"):
        assert embedlow[name]["lr"] == pytest.approx(0.008), name


def test_the_head_rate_reaches_training_and_a_zero_rate_freezes_the_head(setup):
    tcfg, tmp = setup
    base = S.dataclasses.replace(tcfg, arch="loop", loop_gain=True, loop_loss="all", optimizer="muon", lr=3e-3, adam_mult=4.0)
    run("T", 0, TINY6, S.dataclasses.replace(base, head_mult=0.0), tmp / "frozen", CPU)
    run("T", 0, TINY6, base, tmp / "moving", CPU)
    head = {n: torch.load(tmp / n / "ckpt.pt", map_location=CPU)["model"]["head.weight"] for n in ("frozen", "moving")}
    assert torch.equal(head["frozen"], torch.zeros_like(head["frozen"])) and not torch.allclose(head["moving"], torch.zeros_like(head["moving"]))


def test_the_head_mult_cli_switch_reaches_the_training_config(setup, monkeypatch):
    tcfg, tmp = setup
    cfg_path = tmp / "cfg.json"
    cfg_path.write_text(S.json.dumps({"sizes": {"T": S.dataclasses.asdict(TINY6)}, "train": S.dataclasses.asdict(tcfg)}))
    seen = {}
    monkeypatch.setattr(S, "run", lambda size, seed, mcfg, tc, out, device, **kw: seen.update(head_mult=tc.head_mult))
    monkeypatch.setattr(sys, "argv", ["s2train", "--config", str(cfg_path), "--size", "T", "--seed", "0",
                                      "--out", str(tmp / "o"), "--device", "cpu", "--head-mult", "1.0"])
    S.main()
    assert seen["head_mult"] == 1.0


def summ(by_depth):
    return R.summary(by_depth)


def test_keep_arm_continues_only_when_seed_zero_earns_at_least_the_sesoi():
    assert R.keep_arm(summ([0.01, 0.0]))
    assert R.keep_arm(summ([3.40, 3.3899]))
    assert not R.keep_arm(summ([3.40, 3.3901]))
    assert not R.keep_arm(summ([3.40, 3.40, 3.40]))


def test_status_distinguishes_futility_stop_from_three_seed_decisions():
    live, dead = summ([3.40, 3.38, 3.37]), summ([3.40, 3.395, 3.392])
    assert R.status({0: dead}) == "NOT_SUPPORTED_FUTILITY"
    assert R.status({0: live}) == "NOT_SUPPORTED_FUTILITY"
    assert R.status({0: live, 1: live, 2: live}) == "SUPPORTED"
    assert R.status({0: live, 1: live, 2: dead}) == "NOT_SUPPORTED"
