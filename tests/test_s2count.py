import json
import math

import numpy as np
import pytest
import torch

import clm.s2train as S
from clm import counts as C
from clm import kenyon as K
from clm.countread import N_GATE_BINS, CountReader
from clm.s2model import B0Config, build_b0
from clm.s2train import S2TrainConfig, run

CPU = torch.device("cpu")
TINY = B0Config(vocab=64, ctx=16, d_model=16, n_layer=3, n_head=2)
EOT = 63


@pytest.fixture
def setup(tmp_path):
    """An iid token stream with count tables built from that same stream: the only
    way to beat uniform on it is to read the position's own occurrence."""
    rng = np.random.default_rng(0)
    train = rng.integers(0, 63, 40_000).astype(np.uint16)
    train.tofile(tmp_path / "train.bin")
    rng.integers(0, 63, 2_000).astype(np.uint16).tofile(tmp_path / "val.bin")
    for k in range(1, 5):
        C.save(C.build(train, k, EOT, 1, parts=2), tmp_path / "count" / f"order{k + 1}",
               {"c_min": 1, "eot_id": EOT})
    tcfg = S2TrainConfig(train_bin=str(tmp_path / "train.bin"), val_bin=str(tmp_path / "val.bin"),
                         tokens=6 * 8 * 16, batch=8, micro=4, warmup=2, lr=3e-3, eval_every=3,
                         eval_tokens=512, ckpt_every=100, memory="count",
                         count_dir=str(tmp_path / "count"), count_k=4)
    return tcfg, tmp_path


def randomised(seed=0):
    m = build_b0(TINY, seed)
    torch.manual_seed(seed)
    for blk in m.blocks:
        torch.nn.init.normal_(blk.proj.weight, std=0.2)
        torch.nn.init.normal_(blk.down.weight, std=0.2)
    torch.nn.init.normal_(m.head.weight, std=0.5)
    return m


def test_count_input_is_the_score_weighted_sum_of_head_rows():
    m = randomised()
    x = torch.randint(0, 64, (2, 16))
    tok = torch.randint(0, 64, (2, 16, 5))
    score = torch.rand(2, 16, 5)
    by_hand = (score.unsqueeze(-1) * m.head.weight[tok]).sum(2)
    a, _ = m(x, count=(tok, score))
    b, _ = m(x, extra=by_hand)
    assert torch.allclose(a, b, atol=1e-5)
    c, _ = m(x)
    assert not torch.allclose(a, c)


def test_zero_scores_leave_the_model_unchanged_and_the_head_gets_gradient_through_the_input():
    m = randomised()
    x = torch.randint(0, 64, (2, 16))
    tok = torch.randint(0, 64, (2, 16, 5))
    plain, _ = m(x)
    zero, _ = m(x, count=(tok, torch.zeros(2, 16, 5)))
    assert torch.equal(plain, zero)
    score = torch.rand(2, 16, 5)
    m(x, x, count=(tok, score))[1].backward()
    via_input = m.head.weight.grad.clone()
    m.zero_grad()
    with torch.no_grad():
        frozen = (score.unsqueeze(-1) * m.head.weight[tok]).sum(2)
    m(x, x, extra=frozen)[1].backward()
    assert not torch.allclose(via_input, m.head.weight.grad)      # the rows B[w] are trained through f


def test_interrupted_and_resumed_count_run_ends_bit_identical(setup):
    tcfg, tmp = setup
    tcfg = S.dataclasses.replace(tcfg, optimizer="muon")
    run("T", 0, TINY, tcfg, tmp / "straight", CPU)
    run("T", 0, TINY, tcfg, tmp / "resumed", CPU, max_steps=3)
    assert not (tmp / "resumed" / "results.json").exists()
    run("T", 0, TINY, tcfg, tmp / "resumed", CPU)
    a = torch.load(tmp / "straight" / "ckpt.pt")["model"]
    b = torch.load(tmp / "resumed" / "ckpt.pt")["model"]
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_count_run_reports_its_arm_and_lookup_cost(setup):
    tcfg, tmp = setup
    run("T", 1, TINY, tcfg, tmp / "c", CPU)
    run("T", 1, TINY, S.dataclasses.replace(tcfg, loo=False), tmp / "n", CPU)
    run("T", 1, TINY, S.dataclasses.replace(tcfg, memory="none"), tmp / "b", CPU)
    got = {n: json.loads((tmp / n / "results.json").read_text()) for n in "cnb"}
    assert [got[n]["arm"] for n in "cnb"] == ["count", "count_noloo", "b0"]
    assert got["c"]["lookup_seconds_mean"] > 0 and got["b"]["lookup_seconds_mean"] is None
    assert got["c"]["host_rss_peak_mib"] > 0
    log = [json.loads(l) for l in (tmp / "c" / "log.jsonl").read_text().splitlines()]
    assert all(e["lookup_seconds"] > 0 for e in log)
    assert got["c"]["final_val_loss"] != got["b"]["final_val_loss"]       # the memory is wired in


def test_training_reads_leave_one_out_counts_only_when_asked(setup, monkeypatch):
    """On iid tokens the true next token is in a position's counts only through the
    position's own occurrence. Look at what the trainer feeds the model."""
    tcfg, tmp = setup
    seen = {}
    real = S.make_step

    def spy(*a):
        data = real(*a)
        seen.setdefault(tag, []).append(data)
        return data
    monkeypatch.setattr(S, "make_step", spy)
    for tag, loo in (("loo", True), ("noloo", False)):
        run("T", 0, TINY, S.dataclasses.replace(tcfg, loo=loo), tmp / tag, CPU)

    def own_token_share(batches):
        hit = [((d.tok == d.y[..., None]) & (d.score > 0)).any(-1) for d in batches]
        return float(np.mean(np.concatenate([h[:, 4:].ravel() for h in hit])))
    assert own_token_share(seen["noloo"]) > 0.95
    assert own_token_share(seen["loo"]) < 0.5


def test_validation_reads_counts_without_leave_one_out_and_training_reads_them_with_it(setup, monkeypatch):
    """Validation documents are not in the count corpus, so subtracting 'their own
    occurrence' would corrupt the read. Only training windows get leave-one-out."""
    tcfg, tmp = setup
    calls = []
    real = CountReader.lookup

    def spy(self, x, y=None, states=False):
        calls.append(y is not None)
        return real(self, x, y, states)
    monkeypatch.setattr(CountReader, "lookup", spy)
    run("T", 0, TINY, tcfg, tmp / "r", CPU)
    log = [json.loads(l) for l in (tmp / "r" / "log.jsonl").read_text().splitlines()]
    n_eval = sum("val_loss" in e for e in log)
    assert calls.count(False) == n_eval * 512 // (4 * TINY.ctx)        # eval: 512 tokens, micro 4 windows
    assert calls.count(True) >= len(log)                               # one lookup per training step


def test_unknown_memory_is_refused(setup):
    tcfg, tmp = setup
    with pytest.raises(ValueError, match="memory"):
        run("T", 0, TINY, S.dataclasses.replace(tcfg, memory="kenyon"), tmp / "x", CPU)


def test_a_run_with_bucket_tables_reads_them_and_names_its_arm(setup, monkeypatch):
    tcfg, tmp = setup
    stream = np.fromfile(tmp / "train.bin", dtype=np.uint16)
    fn = K.exclude_eot(lambda packed, toks: K.hash_bucket(packed, 50), EOT)
    tables = {k: K.bucket_table(stream, k, EOT, fn, c_min=1, parts=2) for k in (2, 3)}
    K.save_buckets(tmp / "buckets", tables, "random", EOT, n_buckets=50)
    filled = {}
    real = CountReader.lookup

    def spy(self, x, y=None, states=False):
        tok, score = real(self, x, y, states)
        filled.setdefault((tag, y is not None), []).append(int((score > 0).sum()))
        return tok, score
    monkeypatch.setattr(CountReader, "lookup", spy)
    for tag, cfg in (("bucket", S.dataclasses.replace(tcfg, bucket_dir=str(tmp / "buckets"))), ("plain", tcfg)):
        run("T", 0, TINY, cfg, tmp / tag, CPU)
    arms = [json.loads((tmp / n / "results.json").read_text())["arm"] for n in ("bucket", "plain")]
    assert arms == ["count_random", "count"]
    for is_train in (True, False):                                           # training (leave-one-out) and validation
        assert np.mean(filled[("bucket", is_train)]) > 1.15 * np.mean(filled[("plain", is_train)])


GATED = S.dataclasses.replace(TINY, count_orders=4)
KK = 2                                                     # continuations per order in the gated inputs


def gated_inputs(seed=0):
    """A read as CountReader.lookup(states=True) gives it: an absent order (state 0) has zero scores."""
    g = torch.Generator().manual_seed(seed)
    tok = torch.randint(0, 64, (2, 16, 4 * KK), generator=g)
    state = torch.randint(0, 1 + 2 * N_GATE_BINS, (2, 16, 4), generator=g).to(torch.int8)
    score = torch.rand(2, 16, 4 * KK, generator=g) * (state > 0).repeat_interleave(KK, -1)
    return tok, score, state


def gated_copy(m):
    g = build_b0(GATED, 0)
    got = g.load_state_dict(m.state_dict(), strict=False)
    assert set(got.missing_keys) == {"count_gate", "count_absent"} and not got.unexpected_keys
    return g


def test_a_fresh_gated_model_reads_exactly_as_the_ungated_one():
    fresh, plain = build_b0(GATED, 0), dict(build_b0(TINY, 0).named_parameters())
    for name, p in fresh.named_parameters():
        if name in plain:
            assert torch.equal(p, plain[name]), name                   # shared parameters start identical
    assert torch.equal(fresh.count_gate, torch.ones(4, 2 * N_GATE_BINS)) and not fresh.count_absent.any()
    m = randomised()
    g = gated_copy(m)
    x = torch.randint(0, 64, (2, 16))
    tok, score, state = gated_inputs()
    assert torch.equal(g(x, count=(tok, score, state))[0], m(x, count=(tok, score))[0])


def test_gated_read_scales_each_order_by_its_own_scalar_and_fills_absent_orders():
    m = randomised()
    g = gated_copy(m)
    with torch.no_grad():
        g.count_gate.copy_(torch.rand(g.count_gate.shape, generator=torch.Generator().manual_seed(1)) * 2)
        g.count_absent.normal_(generator=torch.Generator().manual_seed(2))
    x = torch.randint(0, 64, (2, 16))
    tok, score, state = gated_inputs()
    assert (state == 0).any() and len(state.unique()) > 4
    by_hand = torch.zeros(2, 16, TINY.d_model)
    with torch.no_grad():
        for b in range(2):
            for t in range(16):
                for o in range(4):
                    s = int(state[b, t, o])
                    if s == 0:
                        by_hand[b, t] += g.count_absent[o]
                    for k in range(KK) if s else ():
                        slot = o * KK + k
                        by_hand[b, t] += g.count_gate[o, s - 1] * score[b, t, slot] * m.head.weight[tok[b, t, slot]]
    a, _ = g(x, count=(tok, score, state))
    ref, _ = m(x, extra=by_hand)
    assert torch.allclose(a, ref, atol=1e-5)


def test_only_the_gates_a_read_used_and_the_absent_orders_get_gradient():
    g = gated_copy(randomised())
    x = torch.randint(0, 64, (2, 16))
    tok, score, state = gated_inputs()
    state[:, :, 3] = torch.where(state[:, :, 3] > 0, 1, 0)        # order 3: exact bin 0 or absent only
    score = score * (state > 0).repeat_interleave(KK, -1)
    g(x, x, count=(tok, score, state))[1].backward()
    used = {(o, int(s) - 1) for o in range(4) for s in state[..., o].unique() if s > 0}
    for o in range(4):
        for i in range(2 * N_GATE_BINS):
            assert bool(g.count_gate.grad[o, i] != 0) == ((o, i) in used), (o, i)
        assert bool(g.count_absent.grad[o].any()) == bool((state[..., o] == 0).any()), o


def test_a_read_that_does_not_match_the_model_is_refused():
    x = torch.randint(0, 64, (2, 16))
    tok, score, state = gated_inputs()
    with pytest.raises(ValueError, match="gated"):
        randomised()(x, count=(tok, score, state))
    with pytest.raises(ValueError, match="gated"):
        build_b0(GATED, 0)(x, count=(tok, score))
    with pytest.raises(ValueError, match="orders"):
        build_b0(GATED, 0)(x, count=(tok, score, state[..., :3]))


def test_a_gated_run_trains_its_gates_and_reports_them(setup):
    tcfg, tmp = setup
    run("T", 0, TINY, S.dataclasses.replace(tcfg, count_gate=True), tmp / "g", CPU)
    run("T", 0, TINY, tcfg, tmp / "c", CPU)
    got = {n: json.loads((tmp / n / "results.json").read_text()) for n in "gc"}
    assert [got[n]["arm"] for n in "gc"] == ["count_gated", "count"]
    assert got["g"]["model"]["count_orders"] == 4 and got["c"]["model"]["count_orders"] == 0
    gate = np.array(got["g"]["count_gate"])
    assert gate.shape == (4, 2 * N_GATE_BINS) and not np.allclose(gate, 1.0)
    assert len(got["g"]["count_absent_norm"]) == 4 and got["c"]["count_gate"] is None
    assert math.isfinite(got["g"]["final_val_loss"]) and got["g"]["final_val_loss"] != got["c"]["final_val_loss"]


def test_a_gate_without_a_count_memory_or_set_by_the_size_config_is_refused(setup):
    tcfg, tmp = setup
    with pytest.raises(ValueError, match="count_gate"):
        run("T", 0, TINY, S.dataclasses.replace(tcfg, memory="none", count_gate=True), tmp / "x", CPU)
    with pytest.raises(ValueError, match="count_orders"):
        run("T", 0, GATED, tcfg, tmp / "y", CPU)
