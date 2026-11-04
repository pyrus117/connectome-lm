"""Stage 2F: separate rates for the Muon block matrices and the AdamW embeddings, head and gates."""
import sys

import pytest
import torch

import clm.s2train as S
from clm.s2model import B0Config, build_b0, param_groups
from clm.s2train import S2TrainConfig, run
from test_s2 import CPU, tiny  # noqa: F401  (the fixture)

TINY = B0Config(vocab=64, ctx=16, d_model=16, n_layer=3, n_head=2)


def test_the_adam_group_runs_at_the_multiple_and_muon_at_the_base_rate():
    m = build_b0(TINY, 0)
    muon, adam = param_groups(m, "muon", 1e-3, 0.1, adam_mult=4.0)
    assert muon.param_groups[0]["lr"] == 1e-3 and "lr_mult" not in muon.param_groups[0]
    assert adam.param_groups[0]["lr"] == 4e-3 and adam.param_groups[0]["lr_mult"] == 4.0
    (adamw,) = param_groups(m, "adamw", 1e-3, 0.1, adam_mult=4.0)
    assert [g["lr"] for g in adamw.param_groups] == [1e-3, 4e-3]           # block matrices first, then the rest


def test_without_a_multiple_both_groups_share_one_rate():
    m = build_b0(TINY, 0)
    muon, adam = param_groups(m, "muon", 1e-3, 0.1)
    assert muon.param_groups[0]["lr"] == adam.param_groups[0]["lr"] == 1e-3
    assert adam.param_groups[0]["lr_mult"] == 1.0


def test_every_step_scales_the_adam_group_by_the_multiple(tiny):  # noqa: F811
    mcfg, tcfg, tmp = tiny
    tcfg = S.dataclasses.replace(tcfg, optimizer="muon", adam_mult=4.0)
    run("T", 0, mcfg, tcfg, tmp / "m", CPU, max_steps=3)                     # stops after step 2, at the full rate
    muon, adam = torch.load(tmp / "m" / "ckpt.pt")["opts"]
    assert muon["param_groups"][0]["lr"] == pytest.approx(3e-3)
    assert adam["param_groups"][0]["lr"] == pytest.approx(12e-3)


def test_the_command_line_multiple_reaches_the_training_config(monkeypatch):
    seen = {}
    monkeypatch.setattr(S, "load_config", lambda c, s: (TINY, S2TrainConfig()))
    monkeypatch.setattr(S, "run", lambda size, seed, mcfg, tcfg, out, device, **k: seen.update(tcfg=tcfg))
    monkeypatch.setattr(sys, "argv", ["s2train", "--config", "x", "--size", "T", "--seed", "0", "--out", "o",
                                      "--device", "cpu", "--lr", "0.008", "--adam-mult", "4"])
    S.main()
    assert seen["tcfg"].adam_mult == 4.0 and seen["tcfg"].lr == 0.008
