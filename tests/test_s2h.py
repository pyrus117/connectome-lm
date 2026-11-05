"""Stage 2H (follow-up 5A): a per-pass residual gain in the looped model, and the loss on every depth."""
import json
import sys

import pytest
import torch
import torch.nn.functional as F

import clm.s2train as S
from clm.s2loop import build_loop
from clm.s2model import param_groups
from clm.s2train import S2TrainConfig, run
from test_s2loop import CPU, TINY, tiny  # noqa: F401  (the fixture)

R = 4


def gained(seed=0):
    """A gained model whose zero-initialised outputs are opened and whose gains differ from pass to pass."""
    m = build_loop(TINY, seed, R, gain=True)
    torch.manual_seed(seed)
    for blk in m.blocks:
        torch.nn.init.normal_(blk.proj.weight, std=0.3)
        torch.nn.init.normal_(blk.down.weight, std=0.3)
    torch.nn.init.normal_(m.head.weight, std=0.5)
    with torch.no_grad():
        m.loop_gain.copy_(0.5 + torch.rand(R, TINY.d_model))
    return m.eval()


def test_gain_adds_one_parameter_per_pass_and_channel_and_a_fresh_model_reads_as_the_ungained_one():
    plain, g = build_loop(TINY, 5, R), build_loop(TINY, 5, R, gain=True)
    assert "loop_gain" not in dict(plain.named_parameters())
    assert g.loop_gain.shape == (R, TINY.d_model) and torch.equal(g.loop_gain, torch.ones(R, TINY.d_model))
    plain, g = plain.eval(), g.eval()
    for blk in (*plain.blocks, *g.blocks):                      # open the zero-initialised outputs identically
        torch.manual_seed(1)
        torch.nn.init.normal_(blk.proj.weight, std=0.3)
        torch.nn.init.normal_(blk.down.weight, std=0.3)
    g.load_state_dict({**plain.state_dict(), "loop_gain": g.loop_gain.detach()})
    x = torch.randint(0, 64, (2, 16))
    for r in range(1, R + 1):
        assert torch.allclose(plain(x, depth=r)[0], g(x, depth=r)[0], atol=1e-5), r


def test_each_pass_scales_its_own_change_by_its_own_gain():
    m = gained()
    x = torch.randint(0, 64, (2, 16))
    h, ves = m._stem(x)
    h1 = h + m.loop_gain[0] * (m._loop(h) - h)
    h2 = h1 + m.loop_gain[1] * (m._loop(h1) - h1)
    assert torch.allclose(m(x, depth=1)[0], m._read(h1, ves), atol=1e-6)
    assert torch.allclose(m(x, depth=2)[0], m._read(h2, ves), atol=1e-6)
    assert not torch.allclose(m._read(h2, ves), m._read(h1, ves))
    with torch.no_grad():
        m.loop_gain[1].zero_()                                   # a pass with gain 0 changes nothing
    assert torch.allclose(m(x, depth=2)[0], m(x, depth=1)[0], atol=1e-6)


def test_the_gain_reaches_the_exit_policy_and_the_all_depth_readout_too():
    m = gained()
    x, y = torch.randint(0, 64, (2, 16)), torch.randint(0, 64, (2, 16))
    assert torch.allclose(m(x, depth=torch.full((2, 16), 3))[0], m(x, depth=3)[0], atol=1e-6)
    with torch.no_grad():
        m.loop_gain[2].mul_(3.0)
    assert not torch.allclose(m(x, depth=torch.full((2, 16), 3))[0], m(x, depth=2)[0], atol=1e-4)
    losses, _ = m.all_depths(x, y)
    for r in range(1, R + 1):
        want = F.cross_entropy(m(x, depth=r)[0].reshape(-1, 64), y.reshape(-1), reduction="none").view(2, 16)
        assert torch.allclose(losses[r - 1], want, atol=1e-5), r


def test_forward_all_is_the_loss_at_every_depth_and_its_mean_has_the_mean_gradient():
    m = gained().train()
    x, y = torch.randint(0, 64, (2, 16)), torch.randint(0, 64, (2, 16))
    per = m.forward_all(x, y)
    assert per.shape == (R,)
    for r in range(1, R + 1):
        assert per[r - 1].item() == pytest.approx(m(x, y, depth=r)[1].item(), abs=1e-5), r
    per.mean().backward()
    got = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad()
    sum(m(x, y, depth=r)[1] for r in range(1, R + 1)).div(R).backward()
    for n, p in m.named_parameters():
        assert torch.allclose(got[n], p.grad, atol=1e-5), n
    assert got["loop_gain"].abs().sum() > 0 and all(got["loop_gain"][r].abs().sum() > 0 for r in range(R))


def test_the_gain_trains_under_adamw_without_weight_decay_and_every_parameter_is_optimised_once():
    m = gained()
    muon, adam = param_groups(m, "muon", 1e-3, 0.1)
    in_muon = {id(p) for g in muon.param_groups for p in g["params"]}
    in_adam = {id(p) for g in adam.param_groups for p in g["params"]}
    assert id(m.loop_gain) in in_adam and id(m.loop_gain) not in in_muon
    assert adam.param_groups[0]["weight_decay"] == 0.0
    assert in_muon | in_adam == {id(p) for p in m.parameters()} and not in_muon & in_adam


def test_an_all_depth_gained_run_reports_every_depths_train_loss_and_trains_its_gain(tiny):  # noqa: F811
    tcfg, tmp = tiny
    tcfg = S.dataclasses.replace(tcfg, loop_gain=True, loop_loss="all")
    run("T", 0, TINY, tcfg, tmp / "r", CPU)
    r = json.loads((tmp / "r" / "results.json").read_text())
    assert r["arm"] == "loop_gain_all" and r["train"]["loop_gain"] is True and r["train"]["loop_loss"] == "all"
    assert r["non_embedding_params"] == build_loop(TINY, 0, 4).non_embedding_params() + 4 * TINY.d_model
    log = [json.loads(l) for l in (tmp / "r" / "log.jsonl").read_text().splitlines()]
    assert all("depths" not in e and len(e["train_loss_by_depth"]) == 4 for e in log)
    assert all(e["loss"] == pytest.approx(sum(e["train_loss_by_depth"]) / 4, abs=1e-4) for e in log)
    gain = torch.load(tmp / "r" / "ckpt.pt")["model"]["loop_gain"]
    assert gain.shape == (4, TINY.d_model) and not torch.equal(gain, torch.ones(4, TINY.d_model))


def test_the_arm_names_follow_the_two_switches(tiny):  # noqa: F811
    tcfg, tmp = tiny
    for name, kw in (("loop_gain", dict(loop_gain=True)), ("loop_all", dict(loop_loss="all"))):
        run("T", 0, TINY, S.dataclasses.replace(tcfg, **kw), tmp / name, CPU)
        assert json.loads((tmp / name / "results.json").read_text())["arm"] == name
    log = [json.loads(l) for l in (tmp / "loop_gain" / "log.jsonl").read_text().splitlines()]
    assert all("depths" in e and "train_loss_by_depth" not in e for e in log)      # a sampled run still draws depths


@pytest.mark.parametrize("kw", [dict(loop_gain=True, loop_loss="all"), dict(loop_gain=True), dict(loop_loss="all")])
def test_an_interrupted_and_resumed_run_ends_bit_identical_with_the_gain_and_the_all_depth_loss(tiny, kw):  # noqa: F811
    tcfg, tmp = tiny
    tcfg = S.dataclasses.replace(tcfg, **kw)
    run("T", 0, TINY, tcfg, tmp / "straight", CPU)
    run("T", 0, TINY, tcfg, tmp / "resumed", CPU, max_steps=3)
    run("T", 0, TINY, tcfg, tmp / "resumed", CPU)
    a, b = (torch.load(tmp / n / "ckpt.pt")["model"] for n in ("straight", "resumed"))
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_the_switches_are_refused_outside_the_looped_arch_or_with_an_unknown_loss(tiny):  # noqa: F811
    tcfg, tmp = tiny
    with pytest.raises(ValueError, match="looped arch"):
        run("T", 0, TINY, S.dataclasses.replace(tcfg, arch="b0", loop_gain=True), tmp / "a", CPU)
    with pytest.raises(ValueError, match="looped arch"):
        run("T", 0, TINY, S.dataclasses.replace(tcfg, arch="b0", loop_loss="all"), tmp / "b", CPU)
    with pytest.raises(ValueError, match="loop_loss"):
        run("T", 0, TINY, S.dataclasses.replace(tcfg, loop_loss="most"), tmp / "c", CPU)


def test_the_command_line_switches_reach_the_training_config(monkeypatch):
    seen = {}
    monkeypatch.setattr(S, "load_config", lambda c, s: (TINY, S2TrainConfig()))
    monkeypatch.setattr(S, "run", lambda size, seed, mcfg, tcfg, out, device, **k: seen.update(tcfg=tcfg))
    monkeypatch.setattr(sys, "argv", ["s2train", "--config", "x", "--size", "T", "--seed", "0", "--out", "o",
                                      "--device", "cpu", "--loop-gain", "--loop-loss", "all", "--micro", "4"])
    S.main()
    assert (seen["tcfg"].loop_gain, seen["tcfg"].loop_loss, seen["tcfg"].micro) == (True, "all", 4)
    monkeypatch.setattr(sys, "argv", ["s2train", "--config", "x", "--size", "T", "--seed", "0", "--out", "o",
                                      "--device", "cpu"])
    S.main()
    assert (seen["tcfg"].loop_gain, seen["tcfg"].loop_loss, seen["tcfg"].micro) == (False, "sampled", 8)
