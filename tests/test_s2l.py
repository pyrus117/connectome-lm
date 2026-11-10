import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import clm.s2train as S
from clm.countread import reader_for
from clm.data import open_tokens
from clm.s2loop import LoopGPT, build_loop
from clm.s2model import B0Config, build_b0
from clm.s2train import run
from test_s2count import CPU, KK, gated_inputs, setup  # noqa: F401  (setup is a fixture)

TINY6 = B0Config(vocab=64, ctx=16, d_model=16, n_layer=6, n_head=2)
GATED6 = B0Config(vocab=64, ctx=16, d_model=16, n_layer=6, n_head=2, count_orders=4)
SCALED6 = B0Config(vocab=64, ctx=16, d_model=16, n_layer=6, n_head=2, count_scale=4)


def opened(cfg, seed=0, gain=False):
    """A looped model and the B0 model it starts from, with their zero-initialised outputs opened so depth and the read matter."""
    b0 = build_b0(cfg, seed)
    torch.manual_seed(seed)
    for blk in b0.blocks:
        torch.nn.init.normal_(blk.proj.weight, std=0.3)
        torch.nn.init.normal_(blk.down.weight, std=0.3)
    torch.nn.init.normal_(b0.head.weight, std=0.5)
    loop = build_loop(cfg, seed, 4, gain=gain)
    loop.load_state_dict(b0.state_dict(), strict=False)
    return b0.eval(), loop.eval()


def read(cfg):
    tok, score, state = gated_inputs()
    return (tok, score, state) if cfg.count_orders else (tok, score)


def test_the_loop_at_depth_one_reads_the_count_exactly_as_b0_does_for_every_kind_of_read():
    x = torch.randint(0, 64, (2, 16))
    for cfg in (TINY6, GATED6, SCALED6):
        b0, loop = opened(cfg)
        assert torch.equal(loop(x, depth=1, count=read(cfg))[0], b0(x, count=read(cfg))[0]), cfg
        assert not torch.allclose(loop(x, depth=1, count=read(cfg))[0], loop(x, depth=1)[0])      # the read matters
        assert not torch.allclose(loop(x, depth=3, count=read(cfg))[0], loop(x, depth=3)[0])      # and survives later passes


def test_a_loop_with_no_read_or_all_zero_scores_is_unchanged():
    x = torch.randint(0, 64, (2, 16))
    _, loop = opened(TINY6)
    tok, score, _ = gated_inputs()
    plain = loop(x, depth=3)[0]
    assert torch.equal(loop(x, depth=3, count=None)[0], plain)
    assert torch.allclose(loop(x, depth=3, count=(tok, score * 0))[0], plain, atol=1e-6)


def test_the_all_depth_loss_and_the_per_token_readouts_carry_the_read_to_every_depth():
    x, y = torch.randint(0, 64, (2, 16)), torch.randint(0, 64, (2, 16))
    _, loop = opened(TINY6, gain=True)
    with torch.no_grad():
        loop.loop_gain.copy_(torch.rand(loop.loop_gain.shape, generator=torch.Generator().manual_seed(2)) + 0.5)
    count = read(TINY6)
    got = loop.forward_all(x, y, count=count).detach()
    per_token, _ = loop.all_depths(x, y, count=count)
    for r in range(1, 5):
        want = loop(x, y, depth=r, count=count)[1]
        assert got[r - 1].item() == pytest.approx(want.item(), abs=1e-5)
        by_hand = F.cross_entropy(loop(x, depth=r, count=count)[0].transpose(1, 2), y, reduction="none")
        assert torch.allclose(per_token[r - 1], by_hand, atol=1e-5)
    uniform = torch.full((2, 16), 3)
    assert torch.allclose(loop(x, depth=uniform, count=count)[0], loop(x, depth=3, count=count)[0], atol=1e-6)
    assert not torch.allclose(got, loop.forward_all(x, y)[0:4].detach())                          # the read changed every depth


def test_the_read_sends_gradient_to_the_head_through_the_loop_input():
    x, y = torch.randint(0, 64, (2, 16)), torch.randint(0, 64, (2, 16))
    _, loop = opened(TINY6)
    loop.train()
    loop.forward_all(x, y, count=read(TINY6)).mean().backward()
    with_read = loop.head.weight.grad.clone()
    loop.zero_grad()
    loop.forward_all(x, y).mean().backward()
    assert not torch.allclose(with_read, loop.head.weight.grad)


def run_loop(tcfg, tmp, name, **kw):
    cfg = S.dataclasses.replace(tcfg, arch="loop", loop_gain=True, loop_loss="all", micro=4, **kw)
    run("T", 0, TINY6, cfg, tmp / name, CPU)
    return json.loads((tmp / name / "results.json").read_text()), cfg


def test_a_looped_run_with_a_count_memory_trains_names_its_arm_and_reads_the_counts_when_it_evaluates(setup, monkeypatch):
    tcfg, tmp = setup
    calls, real = [], S.evaluate

    def spy(model, val, ctx, n_eval, micro, device, reader=None, depth=None, states=False):
        calls.append((reader is not None, depth, states))
        return real(model, val, ctx, n_eval, micro, device, reader, depth, states)
    monkeypatch.setattr(S, "evaluate", spy)
    fed, real_all = [], LoopGPT.forward_all

    def spy_all(self, idx, targets, count=None):
        fed.append(count is not None)
        return real_all(self, idx, targets, count)
    monkeypatch.setattr(LoopGPT, "forward_all", spy_all)
    plain, _ = run_loop(S.dataclasses.replace(tcfg, memory="none"), tmp, "plain")
    plain_calls, calls[:] = list(calls), []
    assert fed and not any(fed)
    fed[:] = []
    counted, cfg = run_loop(tcfg, tmp, "counted")
    counted_calls, calls[:] = list(calls), []
    assert fed and all(fed)                                                         # every training step hands the read to the loop
    gated, _ = run_loop(tcfg, tmp, "gated", count_gate=True, gate_mult=1.0)
    assert [r["arm"] for r in (plain, counted, gated)] == ["loop_gain_all", "loop_gain_all_count", "loop_gain_all_count_gated"]
    assert counted["lookup_seconds_mean"] is not None and plain["lookup_seconds_mean"] is None
    assert len(counted["val_loss_by_depth"]) == 6
    assert plain_calls and not any(has for has, _, _ in plain_calls)
    assert sorted({d for _, d, _ in counted_calls}) == [1, 2, 3, 4, 5, 6] and all(has and not st for has, _, st in counted_calls)
    assert calls and all(has and st for has, _, st in calls)                       # a gated read also asks for the states
    model = build_loop(TINY6, 0, 6, gain=True)
    model.load_state_dict(torch.load(tmp / "counted" / "ckpt.pt", map_location=CPU)["model"])
    reader, val = reader_for(tmp / "count", cfg.count_k, None), open_tokens(tmp / "val.bin")
    for r in (1, 6):
        assert counted["val_loss_by_depth"][r - 1] == pytest.approx(real(model, val, 16, cfg.eval_tokens, 4, CPU, reader, depth=r), abs=1e-5)
