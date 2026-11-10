import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import clm.s2train as S
from clm.s2loop import LoopGPT, build_loop, sample_depth
from clm.s2model import B0Config, build_b0, rms
from clm.s2train import S2TrainConfig, run

CPU = torch.device("cpu")
TINY = B0Config(vocab=64, ctx=16, d_model=16, n_layer=6, n_head=2)


def opened(seed=0, max_depth=4):
    """A model whose zero-initialised outputs are opened, so depth changes the logits."""
    m = build_loop(TINY, seed, max_depth)
    torch.manual_seed(seed)
    for blk in m.blocks:
        torch.nn.init.normal_(blk.proj.weight, std=0.3)
        torch.nn.init.normal_(blk.down.weight, std=0.3)
    torch.nn.init.normal_(m.head.weight, std=0.5)
    return m.eval()


def test_loop_model_has_b0s_parameters_and_starts_from_b0s_weights():
    a, b = build_b0(TINY, 3), build_loop(TINY, 3)
    assert [n for n, _ in a.named_parameters()] == [n for n, _ in b.named_parameters()]
    for (n, p), q in zip(a.named_parameters(), b.parameters()):
        assert torch.equal(p, q), n


def test_depth_one_is_the_b0_forward_and_depth_r_runs_the_loop_r_times():
    m = opened()
    b0 = build_b0(TINY, 0)
    b0.load_state_dict(m.state_dict())
    x = torch.randint(0, 64, (2, 16))
    assert torch.allclose(m(x, depth=1)[0], b0(x)[0], atol=1e-6)
    h, ves = m._stem(x)
    for _ in range(3):
        h = m.blocks[3](m.blocks[2](h, m.cos, m.sin), m.cos, m.sin)
    by_hand = m._read(h, ves)
    assert torch.allclose(m(x, depth=3)[0], by_hand, atol=1e-6)
    assert not torch.allclose(m(x, depth=3)[0], m(x, depth=2)[0])


def test_all_depths_match_separate_forwards_and_report_the_state_change():
    m = opened()
    x, y = torch.randint(0, 64, (2, 16)), torch.randint(0, 64, (2, 16))
    losses, change = m.all_depths(x, y)
    assert losses.shape == change.shape == (4, 2, 16)
    for r in range(1, 5):
        logits, _ = m(x, depth=r)
        want = F.cross_entropy(logits.reshape(-1, 64), y.reshape(-1), reduction="none").view(2, 16)
        assert torch.allclose(losses[r - 1], want, atol=1e-5), r
    h, _ = m._stem(x)
    first = m._loop(h)
    assert torch.allclose(change[0], (first - h).norm(dim=-1) / first.norm(dim=-1), atol=1e-6)


def test_a_uniform_depth_tensor_equals_the_integer_depth():
    m = opened()
    x = torch.randint(0, 64, (2, 16))
    assert torch.allclose(m(x, depth=torch.full((2, 16), 3))[0], m(x, depth=3)[0], atol=1e-6)


def test_an_exited_token_keeps_its_state_and_later_tokens_read_it_frozen():
    m = opened()
    x = torch.randint(0, 64, (1, 16))
    d = torch.full((1, 16), 2)
    d[0, :3] = 1                                      # the first three tokens leave after one pass
    mixed, one, two = m(x, depth=d)[0], m(x, depth=1)[0], m(x, depth=2)[0]
    assert torch.allclose(mixed[0, :3], one[0, :3], atol=1e-6)           # exited tokens read out their own pass-1 state
    assert not torch.allclose(mixed[0, 3:], two[0, 3:], atol=1e-5)       # later tokens saw frozen keys and values


def test_logits_ignore_later_tokens_at_every_depth():
    m = opened()
    x = torch.randint(0, 64, (2, 16))
    y = x.clone()
    y[:, 10:] = (y[:, 10:] + 1) % 64
    for depth in (1, 4, torch.randint(1, 5, (2, 16))):
        a, b = m(x, depth=depth)[0], m(y, depth=depth)[0]
        assert torch.allclose(a[:, :10], b[:, :10], atol=1e-6)


def test_bad_depths_and_layouts_are_refused():
    m = opened()
    x = torch.randint(0, 64, (1, 16))
    with pytest.raises(ValueError, match="depth"):
        m(x, depth=0)
    with pytest.raises(ValueError, match="depth"):
        m(x, depth=5)
    with pytest.raises(ValueError, match="depth"):
        m(x, depth=torch.full((1, 16), 9))
    with pytest.raises(ValueError, match="six blocks"):
        LoopGPT(B0Config(vocab=64, ctx=16, d_model=16, n_layer=4, n_head=2))


def test_sampled_depths_cover_the_range_and_depend_only_on_seed_step_and_micro():
    draws = [sample_depth(0, s, k, 6) for s in range(60) for k in range(4)]
    assert set(draws) == {1, 2, 3, 4, 5, 6}
    assert sample_depth(0, 7, 2, 6) == sample_depth(0, 7, 2, 6)
    assert [sample_depth(0, s, 0, 6) for s in range(20)] != [sample_depth(1, s, 0, 6) for s in range(20)]


@pytest.fixture
def tiny(tmp_path):
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, 64, 37)
    np.tile(pattern, 40_000 // 37 + 1)[:40_000].astype(np.uint16).tofile(tmp_path / "train.bin")
    rng.integers(0, 64, 2_000).astype(np.uint16).tofile(tmp_path / "val.bin")
    tcfg = S2TrainConfig(train_bin=str(tmp_path / "train.bin"), val_bin=str(tmp_path / "val.bin"),
                         tokens=6 * 8 * 16, batch=8, micro=4, warmup=2, lr=3e-3, eval_every=3,
                         eval_tokens=512, ckpt_every=100, optimizer="muon", arch="loop", loop_max=4)
    return tcfg, tmp_path


def test_interrupted_and_resumed_loop_run_ends_bit_identical(tiny):
    tcfg, tmp = tiny
    run("T", 0, TINY, tcfg, tmp / "straight", CPU)
    run("T", 0, TINY, tcfg, tmp / "resumed", CPU, max_steps=3)
    assert not (tmp / "resumed" / "results.json").exists()
    run("T", 0, TINY, tcfg, tmp / "resumed", CPU)
    a, b = (torch.load(tmp / n / "ckpt.pt")["model"] for n in ("straight", "resumed"))
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_loop_run_reports_the_loss_at_every_depth(tiny):
    tcfg, tmp = tiny
    run("T", 0, TINY, tcfg, tmp / "r", CPU)
    r = json.loads((tmp / "r" / "results.json").read_text())
    assert r["arm"] == "loop" and len(r["val_loss_by_depth"]) == 4
    assert r["final_val_loss"] == r["val_loss_by_depth"][-1]
    log = [json.loads(l) for l in (tmp / "r" / "log.jsonl").read_text().splitlines()]
    assert all(len(e["val_loss_by_depth"]) == 4 for e in log if "val_loss" in e)
    accum = tcfg.batch // tcfg.micro
    for e in log:                                     # the run trained at the depths the registered rule draws
        assert e["depths"] == [sample_depth(0, e["step"], k, 4) for k in range(accum)]


def test_an_unknown_arch_is_refused(tiny):
    tcfg, tmp = tiny
    with pytest.raises(ValueError, match="arch"):
        run("T", 0, TINY, S.dataclasses.replace(tcfg, arch="ring"), tmp / "y", CPU)
