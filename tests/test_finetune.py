# tests/test_finetune.py
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from clm.finetune import FinetuneConfig, load_pretrained, run, run_sequential
from clm.model import ModelConfig, build_model
from clm.vocab_task import make_lists, make_pairs

CPU = torch.device("cpu")
MCFG = ModelConfig(vocab=64, ctx=16, d_model=16, n_layer=3, n_head=2, leak=0.5, k_io=8)
POOL = np.arange(1, 64)
TRAINABLE = {"brain.values", "brain.bias"}


def small_cfg(**kw):
    base = dict(pairs=8, per_seq=4, batch=8, lr=1e-2, warmup=5, steps=20, quiz_every=10)
    base.update(kw)
    return FinetuneConfig(**base)


def test_only_the_brain_changes(region, tmp_path):
    model = build_model(MCFG, "er", 0, region)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    run(model, *make_pairs(POOL, 8, seed=0), small_cfg(), tmp_path / "ft", CPU, eot=0)
    after = model.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]) != (k in TRAINABLE), k


def test_log_and_summary_record_the_quiz_curve(region, tmp_path):
    s = run(build_model(MCFG, "er", 0, region), *make_pairs(POOL, 8, seed=0), small_cfg(),
            tmp_path / "ft", CPU, eot=0)
    log = [json.loads(l) for l in (tmp_path / "ft" / "log.jsonl").read_text().splitlines()]
    assert len(log) == 21 and log[0] == {"step": 0, "quiz_acc": s["start_acc"]}
    assert [r["step"] for r in log if "quiz_acc" in r] == [0, 10, 20]
    assert s["final_acc"] == log[-1]["quiz_acc"]
    assert s["chance"] == pytest.approx(1 / 8) and s["pairs"] == 8
    assert all(0.0 <= r["quiz_acc"] <= 1.0 for r in log if "quiz_acc" in r)
    assert json.loads((tmp_path / "ft" / "summary.json").read_text()) == s


ER0 = Path(__file__).resolve().parent.parent / "runs" / "er-0"


# The tiny random model cannot stand in here: its frozen projections are
# untrained, so the brain's output barely reaches the logits (8 pairs stayed
# at chance after 1000 steps). The claim is about pretrained models, so test
# one. runs/ is git-ignored, and CPU is too slow for 100 steps of the real model.
@pytest.mark.skipif(not (ER0 / "ckpt.pt").exists() or not torch.cuda.is_available(),
                    reason="needs runs/er-0 and a GPU")
def test_pretrained_brain_alone_can_memorise_a_short_list(tmp_path):
    from tokenizers import Tokenizer
    from clm.prep import EOT
    from clm.vocab_task import word_pool
    gpu = torch.device("cuda")
    model, _ = load_pretrained(ER0, ER0.parent.parent / "configs" / "stage1.json", gpu)
    tok = Tokenizer.from_file(str(ER0.parent.parent / "data" / "fineweb" / "tokenizer.json"))
    cfg = FinetuneConfig(pairs=128, steps=100, quiz_every=50)
    s = run(model, *make_pairs(word_pool(tok), 128, seed=0), cfg, tmp_path / "ft", gpu,
            eot=tok.token_to_id(EOT))
    assert s["final_acc"] >= 0.5 > s["start_acc"]


def _fake_run(tmp_path, region, fingerprint=None, status="complete"):
    model = build_model(MCFG, "er", 0, region)
    d = tmp_path / "er-0"
    d.mkdir()
    torch.save({"arm": "er", "seed": 0, "step": 6, "model": model.state_dict()}, d / "ckpt.pt")
    (d / "results.json").write_text(json.dumps({
        "status": status, "arm": "er", "seed": 0,
        "fingerprints": [fingerprint or model.brain.fingerprint]}))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"model": {k: getattr(MCFG, k) for k in MCFG.__dataclass_fields__},
                               "train": {"region": str(region)}}))
    return d, cfg, model


def test_load_pretrained_restores_the_weights(region, tmp_path):
    d, cfg, model = _fake_run(tmp_path, region)
    loaded, res = load_pretrained(d, cfg, CPU)
    for k, v in model.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), k
    assert (res["arm"], res["seed"]) == ("er", 0)


@pytest.mark.parametrize("kw", [{"fingerprint": "not-this-wiring"}, {"status": "diverged"}])
def test_load_pretrained_refuses_a_mismatched_or_unfinished_run(region, tmp_path, kw):
    d, cfg, _ = _fake_run(tmp_path, region, **kw)
    with pytest.raises(RuntimeError):
        load_pretrained(d, cfg, CPU)


def test_sequential_records_every_list_after_every_phase(region, tmp_path):
    model = build_model(MCFG, "er", 0, region)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    lists = make_lists(POOL, k=3, m=8, seed=0)
    s = run_sequential(model, lists, small_cfg(steps=10, quiz_every=5), tmp_path / "seq", CPU, eot=0)
    assert len(s["acc_matrix"]) == 3 and all(len(row) == 3 for row in s["acc_matrix"])
    m = s["acc_matrix"]
    assert s["final_mean"] == pytest.approx(sum(m[2]) / 3)
    assert s["backward_transfer"] == pytest.approx(((m[2][0] - m[0][0]) + (m[2][1] - m[1][1])) / 2)
    log = [json.loads(l) for l in (tmp_path / "seq" / "log.jsonl").read_text().splitlines()]
    assert len(log) == 1 + 30 and log[-1]["step"] == 30 and [r["phase"] for r in log[1:]] == [0] * 10 + [1] * 10 + [2] * 10
    assert [r["step"] for r in log if "quiz" in r] == [0, 5, 10, 15, 20, 25, 30]
    after = model.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]) != (k in TRAINABLE), k
