import json
import math

import numpy as np
import pytest
import torch

import clm.train as T
from clm.graphs import make_wiring
from clm.model import ModelConfig
from clm.train import TrainConfig, lr_at, run

CPU = torch.device("cpu")


def test_lr_warms_up_then_decays_to_the_floor():
    c = TrainConfig(lr=1e-3, min_lr=1e-4, warmup=10)
    assert lr_at(0, 100, c) == pytest.approx(1e-4)
    assert lr_at(9, 100, c) == pytest.approx(1e-3)
    assert lr_at(10, 100, c) == pytest.approx(1e-3)
    assert lr_at(99, 100, c) == pytest.approx(1e-4)
    after = [lr_at(s, 100, c) for s in range(10, 100)]
    assert all(a >= b for a, b in zip(after, after[1:]))


@pytest.fixture
def tiny(tmp_path, region):
    rng = np.random.default_rng(0)
    for name, n in (("train.bin", 40_000), ("val.bin", 2_000)):
        rng.integers(0, 64, n).astype(np.uint16).tofile(tmp_path / name)
    mcfg = ModelConfig(vocab=64, ctx=16, d_model=16, n_layer=3, n_head=2, leak=0.5, k_io=8)
    tcfg = TrainConfig(region=str(region), train_bin=str(tmp_path / "train.bin"),
                       val_bin=str(tmp_path / "val.bin"), tokens=6 * 8 * 16, batch=8, micro=4,
                       warmup=2, eval_every=3, eval_tokens=512, ckpt_every=100)
    return mcfg, tcfg, tmp_path


def test_interrupted_and_resumed_run_ends_bit_identical(tiny):
    mcfg, tcfg, tmp = tiny
    run("real", 0, mcfg, tcfg, tmp / "straight", CPU)
    run("real", 0, mcfg, tcfg, tmp / "resumed", CPU, max_steps=3)
    assert not (tmp / "resumed" / "results.json").exists()
    run("real", 0, mcfg, tcfg, tmp / "resumed", CPU)
    a = torch.load(tmp / "straight" / "ckpt.pt")["model"]
    b = torch.load(tmp / "resumed" / "ckpt.pt")["model"]
    for k in a:
        assert torch.equal(a[k], b[k]), k
    ra, rb = (json.loads((tmp / d / "results.json").read_text()) for d in ("straight", "resumed"))
    assert ra["final_val_loss"] == rb["final_val_loss"]


def test_completed_run_reports_what_the_analysis_checks(tiny):
    mcfg, tcfg, tmp = tiny
    run("shuf", 1, mcfg, tcfg, tmp / "r", CPU)
    r = json.loads((tmp / "r" / "results.json").read_text())
    assert (r["status"], r["arm"], r["seed"]) == ("complete", "shuf", 1)
    assert r["tokens_seen"] == 6 * 8 * 16 and r["steps"] == 6
    assert r["fingerprints"] == [make_wiring(tcfg.region, "shuf", 1, 8).fingerprint()]
    assert math.isfinite(r["final_val_loss"]) and 0 < r["final_val_loss"] < 10
    log = [json.loads(l) for l in (tmp / "r" / "log.jsonl").read_text().splitlines()]
    assert [e["step"] for e in log] == list(range(6))
    assert [e["step"] for e in log if "val_loss" in e] == [2, 5]
    assert log[-1]["val_loss"] == r["final_val_loss"]


def test_dense_run_records_the_fully_connected_wiring(tiny):
    mcfg, tcfg, tmp = tiny
    run("dense", 0, mcfg, tcfg, tmp / "d", CPU)
    assert (json.loads((tmp / "d" / "results.json").read_text())["fingerprints"]
            == [make_wiring(tcfg.region, "dense", 0, 8).fingerprint()])


def test_non_finite_loss_stops_the_run_and_records_divergence(tiny, monkeypatch):
    mcfg, tcfg, tmp = tiny
    real_build = T.build_model

    def poisoned(*a, **k):
        m = real_build(*a, **k)
        forward = m.forward
        m.forward = lambda x, y=None: (forward(x, y)[0], forward(x, y)[1] * float("nan"))
        return m

    monkeypatch.setattr(T, "build_model", poisoned)
    with pytest.raises(SystemExit) as e:
        run("real", 0, mcfg, tcfg, tmp / "bad", CPU)
    assert e.value.code == 2
    st = json.loads((tmp / "bad" / "status.json").read_text())
    assert st["status"] == "diverged" and st["step"] == 0
    assert not (tmp / "bad" / "results.json").exists()


def test_resume_refuses_a_checkpoint_from_another_arm(tiny):
    mcfg, tcfg, tmp = tiny
    run("real", 0, mcfg, tcfg, tmp / "x", CPU, max_steps=2)
    with pytest.raises(RuntimeError, match="checkpoint"):
        run("shuf", 0, mcfg, tcfg, tmp / "x", CPU)


def test_config_file_with_an_unknown_key_is_refused(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"model": {"leak": 0.5, "typo": 1}, "train": {}}))
    with pytest.raises(TypeError):
        T.load_config(p)
