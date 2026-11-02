import json
import math

import numpy as np
import pytest
import torch

import clm.s2train as S
from clm.s2model import B0Config, HeadMuon, apply_rotary, build_b0, param_groups, rotary
from clm.s2train import S2TrainConfig, lr_at, run

CPU = torch.device("cpu")
TINY = B0Config(vocab=64, ctx=16, d_model=16, n_layer=3, n_head=2)


def test_logits_at_a_position_ignore_later_tokens():
    m = build_b0(TINY, 0)
    for blk in m.blocks:                     # zero-init projections hide leaks; open them
        torch.nn.init.normal_(blk.proj.weight, std=0.2)
        torch.nn.init.normal_(blk.down.weight, std=0.2)
    torch.nn.init.normal_(m.head.weight, std=0.2)
    x = torch.randint(0, 64, (2, 16))
    y = x.clone()
    y[:, 10:] = (y[:, 10:] + 1) % 64
    a, _ = m(x)
    b, _ = m(y)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-6)
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_fresh_model_predicts_uniformly():
    m = build_b0(TINY, 0)
    x = torch.randint(0, 64, (2, 16))
    _, loss = m(x, x)
    assert loss.item() == pytest.approx(math.log(64), abs=1e-6)
    for blk in m.blocks:
        assert blk.proj.weight.abs().sum() == 0 and blk.down.weight.abs().sum() == 0


def test_softcap_bounds_the_logits():
    m = build_b0(TINY, 0)
    torch.nn.init.normal_(m.head.weight, std=50.0)
    logits, _ = m(torch.randint(0, 64, (1, 16)))
    assert logits.abs().max() <= TINY.softcap + 1e-4
    assert logits.abs().max() > TINY.softcap * 0.9


def test_rotary_makes_attention_scores_depend_on_offset_only():
    cos, sin = rotary(32, 8, 10000.0)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)

    def score(i, j):
        qi = apply_rotary(q.expand(1, 1, 32, 8), cos, sin)[0, 0, i]
        kj = apply_rotary(k.expand(1, 1, 32, 8), cos, sin)[0, 0, j]
        return float(qi @ kj)
    assert score(5, 3) == pytest.approx(score(12, 10), rel=1e-5)
    assert score(5, 3) != pytest.approx(score(5, 1), rel=1e-3)


def test_same_seed_same_start_different_seed_different_start():
    a, b, c = build_b0(TINY, 0), build_b0(TINY, 0), build_b0(TINY, 1)
    for (n, p), q, r in zip(a.named_parameters(), b.parameters(), c.parameters()):
        assert torch.equal(p, q), n
    assert not torch.equal(a.blocks[0].qkv.weight, c.blocks[0].qkv.weight)


def test_muon_gets_only_block_matrices_and_every_parameter_is_optimised_once():
    m = build_b0(TINY, 0)
    muon, adam = param_groups(m, "muon", 1e-3, 0.1)
    in_muon = {id(p) for g in muon.param_groups for p in g["params"]}
    in_adam = {id(p) for g in adam.param_groups for p in g["params"]}
    names = {id(p): n for n, p in m.named_parameters()}
    assert not in_muon & in_adam
    assert in_muon | in_adam == set(names)
    assert all(names[i].startswith("blocks.") and dict(m.named_parameters())[names[i]].dim() == 2
               for i in in_muon)
    assert {names[i] for i in in_adam} >= {"embed.weight", "head.weight"}
    (adamw,) = param_groups(m, "adamw", 1e-3, 0.1)
    decayed = {id(p) for g in adamw.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    assert decayed == in_muon


def _muon_step(opt_cls, grad, **kw):
    w = torch.nn.Parameter(torch.zeros_like(grad))
    opt = opt_cls([w], lr=1e-2, weight_decay=0.0, adjust_lr_fn="match_rms_adamw", **kw)
    w.grad = grad.clone()
    opt.step()
    return w.detach()


def test_head_muon_without_split_matches_torch_muon_over_several_steps():
    torch.manual_seed(0)
    a, b = build_b0(TINY, 0), build_b0(TINY, 0)
    x = torch.randint(0, 64, (2, 16))
    oa = torch.optim.Muon([p for p in a.blocks.parameters() if p.dim() == 2], lr=1e-2,
                          weight_decay=0.1, adjust_lr_fn="match_rms_adamw")
    ob = HeadMuon([p for p in b.blocks.parameters() if p.dim() == 2], {}, lr=1e-2,
                  weight_decay=0.1, adjust_lr_fn="match_rms_adamw")
    for _ in range(3):
        for m, o in ((a, oa), (b, ob)):
            o.zero_grad()
            m(x, x)[1].backward()
            o.step()
    for (n, p), q in zip(a.named_parameters(), b.parameters()):
        assert torch.equal(p, q), n


def test_head_muon_orthogonalises_each_head_block_on_its_own():
    torch.manual_seed(0)
    g = torch.randn(48, 16)                       # TINY qkv: 3 matrices x 2 heads x 8 rows
    g2 = g.clone()
    g2[:8] += 5 * torch.randn(8, 16)              # change only q of head 0
    w =torch.nn.Parameter(torch.zeros(48, 16))
    split = {id(w): 6}

    def step(grad):
        w.data.zero_()
        opt = HeadMuon([w], split, lr=1e-2, weight_decay=0.0, adjust_lr_fn="match_rms_adamw")
        w.grad = grad.clone()
        opt.step()
        return w.detach().clone()
    u, u2 = step(g), step(g2)
    assert torch.equal(u[8:], u2[8:]) and not torch.equal(u[:8], u2[:8])
    f, f2 = _muon_step(torch.optim.Muon, g), _muon_step(torch.optim.Muon, g2)
    assert not torch.allclose(f[8:], f2[8:])      # fused Muon mixes heads: the check can fail
    for blk in u.view(6, 8, 16):                  # each block: 8 singular values near 1, lr scaled by its shape
        s = torch.linalg.svdvals(blk.float() / (1e-2 * 0.2 * math.sqrt(16)))
        assert s.min() > 0.5 and s.max() < 1.5


def test_muon_head_splits_every_qkv_into_three_blocks_per_head():
    m = build_b0(TINY, 0)
    muon, adam = param_groups(m, "muon_head", 1e-3, 0.1)
    assert isinstance(muon, HeadMuon)
    assert muon.split == {id(b.qkv.weight): 3 * TINY.n_head for b in m.blocks}
    plain, _ = param_groups(m, "muon", 1e-3, 0.1)
    assert [id(p) for p in muon.param_groups[0]["params"]] == [id(p) for p in plain.param_groups[0]["params"]]


def test_lr_warms_up_then_falls_linearly_to_zero():
    c = S2TrainConfig(lr=1e-3, warmup=10)
    assert lr_at(0, 100, c) == pytest.approx(1e-4)
    assert lr_at(9, 100, c) == pytest.approx(1e-3)
    assert lr_at(10, 100, c) == pytest.approx(1e-3)
    assert lr_at(99, 100, c) == 0.0
    after = [lr_at(s, 100, c) for s in range(10, 100)]
    gaps = np.diff(after)
    assert np.allclose(gaps, gaps[0]) and gaps[0] < 0


@pytest.fixture
def tiny(tmp_path):
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, 64, 37)
    tokens = np.tile(pattern, 40_000 // 37 + 1)[:40_000]          # learnable: a repeating sequence
    tokens.astype(np.uint16).tofile(tmp_path / "train.bin")
    rng.integers(0, 64, 2_000).astype(np.uint16).tofile(tmp_path / "val.bin")
    tcfg = S2TrainConfig(train_bin=str(tmp_path / "train.bin"), val_bin=str(tmp_path / "val.bin"),
                         tokens=6 * 8 * 16, batch=8, micro=4, warmup=2, lr=3e-3,
                         eval_every=3, eval_tokens=512, ckpt_every=100)
    return TINY, tcfg, tmp_path


@pytest.mark.parametrize("optimizer", ["adamw", "muon", "muon_head"])
def test_interrupted_and_resumed_run_ends_bit_identical(tiny, optimizer):
    mcfg, tcfg, tmp = tiny
    tcfg = S.dataclasses.replace(tcfg, optimizer=optimizer)
    run("T", 0, mcfg, tcfg, tmp / "straight", CPU)
    run("T", 0, mcfg, tcfg, tmp / "resumed", CPU, max_steps=3)
    assert not (tmp / "resumed" / "results.json").exists()
    run("T", 0, mcfg, tcfg, tmp / "resumed", CPU)
    a = torch.load(tmp / "straight" / "ckpt.pt")["model"]
    b = torch.load(tmp / "resumed" / "ckpt.pt")["model"]
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_training_learns_to_use_context(tiny):
    mcfg, tcfg, tmp = tiny
    tcfg = S.dataclasses.replace(tcfg, tokens=80 * 8 * 16, lr=1e-2, eval_every=1000)
    run("T", 0, mcfg, tcfg, tmp / "learn", CPU)
    log = [json.loads(l) for l in (tmp / "learn" / "log.jsonl").read_text().splitlines()]
    stream = np.fromfile(tmp / "train.bin", dtype=np.uint16)
    _, counts = np.unique(stream, return_counts=True)
    p = counts / counts.sum()
    unigram = -(p * np.log(p)).sum()           # the best loss without looking back
    assert np.mean([e["loss"] for e in log[-5:]]) < 0.5 * unigram


def test_completed_run_reports_what_stage_2_compares(tiny):
    mcfg, tcfg, tmp = tiny
    run("T", 1, mcfg, tcfg, tmp / "r", CPU)
    r = json.loads((tmp / "r" / "results.json").read_text())
    assert (r["status"], r["arm"], r["size"], r["seed"]) == ("complete", "b0", "T", 1)
    assert r["tokens_seen"] == 6 * 8 * 16 and r["steps"] == 6
    assert math.isfinite(r["final_val_loss"]) and r["train_seconds"] > 0
    log = [json.loads(l) for l in (tmp / "r" / "log.jsonl").read_text().splitlines()]
    assert [e["step"] for e in log if "val_loss" in e] == [2, 5]
    assert log[-1]["val_loss"] == r["final_val_loss"]
    assert r["train_seconds"] == pytest.approx(sum(e["seconds"] for e in log), abs=0.01)


def test_non_finite_loss_stops_the_run_and_records_divergence(tiny, monkeypatch):
    mcfg, tcfg, tmp = tiny
    real_build = S.build_b0

    def poisoned(*a, **k):
        m = real_build(*a, **k)
        forward = m.forward
        m.forward = lambda x, y=None, extra=None: (forward(x, y)[0], forward(x, y)[1] * float("nan"))
        return m
    monkeypatch.setattr(S, "build_b0", poisoned)
    with pytest.raises(SystemExit) as e:
        run("T", 0, mcfg, tcfg, tmp / "bad", CPU)
    assert e.value.code == 2
    assert json.loads((tmp / "bad" / "status.json").read_text())["status"] == "diverged"
