# Vocabulary-List Fine-Tune Pilot Implementation Plan

> **For agentic workers:** implement this plan task-by-task, in order, checking off each step as it lands. Steps use checkbox (`- [ ]`) syntax for tracking. Dispatch a fresh subagent per task where the tasks are independent (see dispatching-parallel-agents), otherwise execute inline with a review checkpoint after each task.

**Goal:** Build a task that makes a pretrained Stage 1 model memorise a list of random word pairs using only its brain's synapses, and pilot it on one `er` model to choose the list size, learning rate and training length for a later 15-model run.

**Architecture:** `src/clm/vocab_task.py` makes the word pairs, the training sequences and the quiz (pure functions, no model). `src/clm/finetune.py` loads a finished Stage 1 run, freezes every parameter outside `model.brain`, trains on the task and quizzes recall every 50 steps. `scripts/pilot_vocab.py` runs a grid of settings on `runs/er-0` only and writes `results/pilot_vocab.json`.

**Tech Stack:** Python 3.14 (laptop `.venv`, system site-packages), torch 2.14, numpy, `tokenizers`, pytest.

---

## Background the implementer needs

- Read `HANDOVER.md` first. Stage 1 is finished: 15 runs in `runs/<arm>-<seed>/`, each with `ckpt.pt` (keys `arm`, `seed`, `step`, `model`, `opt`) and `results.json` (`status`, `arm`, `seed`, `fingerprints`, ...). The primary verdict was `X_WORSE` (real wiring slightly worse than shuffled).
- The model (`src/clm/model.py`): six layers share one `SharedBrain` (`src/clm/block.py`). The brain's trainable parameters are `brain.values` (one strength per edge) and `brain.bias` (one per neuron). Each layer has its own `w_in`/`w_out` projections into and out of the brain. **The brain's state resets for every token**, so the brain cannot carry information between tokens; attention does that.
- **Why the task looks like this.** Recall must come from stored weights, not from copying earlier text, or the test measures attention instead of the wiring. So: (1) only `brain.*` trains, everything else is frozen; (2) within one sequence each cue appears at most once, so attention cannot find the answer earlier in the same sequence.
- **Pilot discipline (from Stage 1).** Pilots run on the `er` arm only, so task choices cannot be tuned toward `real` vs `shuf`. `scripts/pilot_vocab.py` enforces this. Do not run `clm.finetune` on any other arm during this plan. Skip `er-1`: it had a gradient spike (max grad norm 83.5) during pretraining.
- **Word pool.** The tokenizer (`data/fineweb/tokenizer.json`, committed, byte-level BPE, 8,192 tokens, `<|eot|>` is id 0) has 3,394 tokens matching `Ġ[a-z]{4,10}` (a leading-space lowercase word of 4–10 letters). Cues and answers are disjoint, so at most 1,697 pairs. Pilot list sizes are multiples of 64: 128, 512, 1536.
- **Sequence format.** `[eot, cue1, ans1, cue2, ans2, ...]` with 64 pairs per sequence: 129 tokens, under the model's 256 context. The target at each cue position is its answer; every other target is `-100` (ignored).
- **Measure.** Quiz accuracy: every cue asked exactly once, teacher-forced, in a fixed order; a hit means the model's top guess at the cue position is the answer. Chance is 1/N once the model knows the answer set. Also record steps to reach 50% and 90%.
- Laptop GPU: RTX 3050, 4 GB. Run `nvidia-smi` before any GPU job; wait if a game or other job holds the GPU. Cap memory with `--mem-fraction 0.75` (the internal panel link failed once under a full-VRAM test). Jobs over 30 minutes go in `systemd-run --user` (the harness kills longer background jobs).
- `load_pretrained` rebuilds the wiring and checks it against the pretraining fingerprint. The pod used numpy 2.5.4; the laptop has 2.5.3 (system site-packages). Checked 2026-10-23: the laptop rebuilds all 15 Stage 1 wirings with matching fingerprints, so no numpy change is needed. Task 3 Step 4 stays as a fallback.
- Test command: `/home/nate/Projects/connectome-lm/.venv/bin/python -m pytest /home/nate/Projects/connectome-lm/tests -q` (114 pass before this plan).

## File structure

- Create `src/clm/vocab_task.py`: word pool, pairs, training batches, quiz batches, hit counting.
- Create `src/clm/finetune.py`: `FinetuneConfig`, freezing, quiz, training loop, loading a finished run, CLI.
- Create `scripts/pilot_vocab.py`: the `er`-only pilot grid.
- Create `tests/test_vocab_task.py`, `tests/test_finetune.py`.
- Create `notes/pilot_vocab.md` after the pilot (measurements and the choice).

---

### Task 1: The vocabulary-list task

**Files:**
- Create: `src/clm/vocab_task.py`
- Test: `tests/test_vocab_task.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vocab_task.py
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from clm.vocab_task import IGNORE, correct, make_pairs, quiz_batches, train_batch, word_pool

TOKENIZER = Path(__file__).resolve().parent.parent / "data" / "fineweb" / "tokenizer.json"
EOT = 0


def test_word_pool_is_single_token_lowercase_words_and_big_enough():
    tok = Tokenizer.from_file(str(TOKENIZER))
    pool = word_pool(tok)
    assert len(pool) >= 2 * 1536
    assert len(set(pool.tolist())) == len(pool)
    for i in pool[:200]:
        w = tok.decode([int(i)])
        assert w[0] == " " and w[1:].isalpha() and w[1:].islower() and 4 <= len(w) - 1 <= 10


def test_pairs_are_disjoint_and_fixed_by_seed():
    pool = np.arange(100, 400)
    c, a = make_pairs(pool, 128, seed=0)
    assert len(c) == len(a) == 128
    assert not set(c.tolist()) & set(a.tolist())
    c2, a2 = make_pairs(pool, 128, seed=0)
    assert np.array_equal(c, c2) and np.array_equal(a, a2)
    assert not np.array_equal(c, make_pairs(pool, 128, seed=1)[0])
    with pytest.raises(ValueError):
        make_pairs(pool, 151, seed=0)


def test_train_batch_targets_only_answers_and_never_repeats_a_cue_in_a_row():
    c, a = make_pairs(np.arange(100, 400), 128, seed=0)
    answer_of = dict(zip(c.tolist(), a.tolist()))
    x, y = train_batch(c, a, batch=8, per_seq=64, seed=0, step=5, eot=EOT)
    assert x.shape == y.shape == (8, 129)
    assert (x[:, 0] == EOT).all()
    for row_x, row_y in zip(x.tolist(), y.tolist()):
        cues = row_x[1::2]
        assert len(set(cues)) == 64
        assert row_y[1::2] == [answer_of[q] for q in cues]
        assert row_x[2::2] == row_y[1::2]
        assert set(row_y[0::2]) == {IGNORE}
    x2, _ = train_batch(c, a, batch=8, per_seq=64, seed=0, step=5, eot=EOT)
    assert torch.equal(x, x2)
    assert not torch.equal(x, train_batch(c, a, 8, 64, 0, 6, EOT)[0])


def test_quiz_asks_every_cue_exactly_once():
    c, a = make_pairs(np.arange(100, 400), 128, seed=0)
    x, y = quiz_batches(c, a, per_seq=64, seed=0, eot=EOT)
    assert x.shape == (2, 129)
    assert sorted(x[:, 1::2].reshape(-1).tolist()) == sorted(c.tolist())
    with pytest.raises(ValueError):
        quiz_batches(c[:100], a[:100], per_seq=64, seed=0, eot=EOT)


def test_correct_counts_top_guesses_at_cue_positions_only():
    y = torch.tensor([[IGNORE, 5, IGNORE, 7]])
    logits = torch.zeros(1, 4, 10)
    logits[0, 1, 5] = 1.0     # right
    logits[0, 3, 2] = 1.0     # wrong
    logits[0, 0, 9] = 1.0     # ignored position
    assert correct(logits, y) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/nate/Projects/connectome-lm/.venv/bin/python -m pytest /home/nate/Projects/connectome-lm/tests/test_vocab_task.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'clm.vocab_task'`.

- [ ] **Step 3: Write the implementation**

```python
# src/clm/vocab_task.py
"""The vocabulary-list task: memorise N random word pairings.

Cue and answer words are single tokens, so a pairing is one token predicting
another. Pairings are random, so English gives no hint; the model has to store
them. Within one sequence every cue appears once, so attention cannot copy an
answer from earlier in the same sequence: recall has to come from weights.
"""
import re

import numpy as np
import torch

WORD = re.compile(r"Ġ[a-z]{4,10}")
IGNORE = -100


def word_pool(tok):
    """Ids of single-token lowercase words (4-10 letters, leading space)."""
    return np.array([i for i in range(tok.get_vocab_size())
                     if WORD.fullmatch(tok.id_to_token(i) or "")], dtype=np.int64)


def make_pairs(pool, n, seed):
    """n (cue, answer) id pairs; cues and answers are disjoint."""
    if 2 * n > len(pool):
        raise ValueError(f"{n} pairs need {2 * n} words; the pool has {len(pool)}")
    words = np.random.default_rng(seed).permutation(pool)[:2 * n]
    return words[:n], words[n:]


def _pack(cues, answers, idx, eot):
    """eot, then cue/answer for each index in idx. Targets are the answer at
    each cue position and IGNORE everywhere else."""
    x = np.empty(1 + 2 * len(idx), np.int64)
    x[0] = eot
    x[1::2], x[2::2] = cues[idx], answers[idx]
    y = np.full_like(x, IGNORE)
    y[1::2] = answers[idx]
    return x, y


def _stack(rows):
    return (torch.from_numpy(np.stack([r[0] for r in rows])),
            torch.from_numpy(np.stack([r[1] for r in rows])))


def train_batch(cues, answers, batch, per_seq, seed, step, eot):
    rng = np.random.default_rng([seed, step])
    return _stack([_pack(cues, answers, rng.choice(len(cues), per_seq, replace=False), eot)
                   for _ in range(batch)])


def quiz_batches(cues, answers, per_seq, seed, eot):
    """Every cue exactly once, in a fixed order, per_seq cues per sequence."""
    if len(cues) % per_seq:
        raise ValueError(f"{len(cues)} pairs do not split into sequences of {per_seq}")
    # 2**31 keeps the quiz order apart from every training step's stream
    order = np.random.default_rng([seed, 2**31]).permutation(len(cues))
    return _stack([_pack(cues, answers, order[i:i + per_seq], eot)
                   for i in range(0, len(order), per_seq)])


def correct(logits, y):
    """Number of cue positions whose top guess is the answer."""
    m = y != IGNORE
    return int((logits.argmax(-1)[m] == y[m]).sum())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/nate/Projects/connectome-lm/.venv/bin/python -m pytest /home/nate/Projects/connectome-lm/tests/test_vocab_task.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/nate/Projects/connectome-lm add src/clm/vocab_task.py tests/test_vocab_task.py
git -C /home/nate/Projects/connectome-lm commit -m "vocab_task: word pairs, training batches and quiz for the brain-only memory task

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Brain-only fine-tuning

**Files:**
- Create: `src/clm/finetune.py`
- Test: `tests/test_finetune.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_finetune.py
import json

import numpy as np
import pytest
import torch

from clm.finetune import FinetuneConfig, load_pretrained, run
from clm.model import ModelConfig, build_model
from clm.vocab_task import make_pairs

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


def test_brain_alone_can_memorise_a_short_list(region, tmp_path):
    s = run(build_model(MCFG, "er", 0, region), *make_pairs(POOL, 8, seed=0),
            small_cfg(steps=300, quiz_every=50), tmp_path / "ft", CPU, eot=0)
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/nate/Projects/connectome-lm/.venv/bin/python -m pytest /home/nate/Projects/connectome-lm/tests/test_finetune.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'clm.finetune'`.

- [ ] **Step 3: Write the implementation**

```python
# src/clm/finetune.py
"""Fine-tune only the brain's synapses on the vocabulary-list task.

    python -m clm.finetune --run runs/er-0 --config configs/stage1.json \
        --out runs/pilot-vocab/er-0-n512 --pairs 512 --lr 1e-3 --steps 1500

Everything outside model.brain is frozen, so whatever the model learns is
stored in the wiring's synapse strengths and biases. Writes log.jsonl (one
line per step, with quiz_acc on quiz steps) and summary.json.
"""
import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from clm.model import build_model
from clm.prep import EOT
from clm.train import ROOT, _autocast, _write, load_config
from clm.vocab_task import IGNORE, correct, make_pairs, quiz_batches, train_batch, word_pool


@dataclasses.dataclass(frozen=True)
class FinetuneConfig:
    pairs: int = 512
    per_seq: int = 64
    batch: int = 16
    lr: float = 1e-3
    warmup: int = 20
    steps: int = 1500
    quiz_every: int = 50
    clip: float = 1.0
    task_seed: int = 0        # same list and order for every model, so arms pair cleanly


def freeze_all_but_brain(model):
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("brain."))
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def quiz(model, qx, qy, micro, device):
    model.eval()
    hits = 0
    for i in range(0, len(qx), micro):
        x, y = qx[i:i + micro].to(device), qy[i:i + micro].to(device)
        with _autocast(device):
            logits, _ = model(x)
        hits += correct(logits, y)
    model.train()
    return hits / int((qy != IGNORE).sum())


def first_step_at(curve, level):
    return next((step for step, acc in curve if acc >= level), None)


def run(model, cues, answers, fcfg, out, device, eot):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.to(device)
    params = freeze_all_but_brain(model)
    opt = torch.optim.AdamW(params, lr=fcfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    qx, qy = quiz_batches(cues, answers, fcfg.per_seq, fcfg.task_seed, eot)
    curve = [(0, quiz(model, qx, qy, fcfg.batch, device))]
    with open(out / "log.jsonl", "w") as log:
        log.write(json.dumps({"step": 0, "quiz_acc": curve[0][1]}) + "\n")
        for step in range(fcfg.steps):
            t0 = time.time()
            for g in opt.param_groups:
                g["lr"] = fcfg.lr * min(1.0, (step + 1) / fcfg.warmup)
            x, y = train_batch(cues, answers, fcfg.batch, fcfg.per_seq, fcfg.task_seed, step, eot)
            x, y = x.to(device), y.to(device)
            with _autocast(device):
                logits, _ = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                                   ignore_index=IGNORE)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, fcfg.clip).item()
            if not (math.isfinite(loss.item()) and math.isfinite(gnorm)):
                raise RuntimeError(f"non-finite loss {loss.item()} or grad norm {gnorm} at step {step}")
            opt.step()
            opt.zero_grad(set_to_none=True)
            rec = {"step": step + 1, "loss": loss.item(), "grad_norm": gnorm,
                   "seconds": round(time.time() - t0, 3)}
            if (step + 1) % fcfg.quiz_every == 0 or step + 1 == fcfg.steps:
                rec["quiz_acc"] = quiz(model, qx, qy, fcfg.batch, device)
                curve.append((step + 1, rec["quiz_acc"]))
            log.write(json.dumps(rec) + "\n")
            log.flush()
    summary = {"pairs": len(cues), "chance": 1 / len(cues),
               "start_acc": curve[0][1], "final_acc": curve[-1][1],
               "steps_to_50": first_step_at(curve, 0.5), "steps_to_90": first_step_at(curve, 0.9),
               "config": dataclasses.asdict(fcfg)}
    _write(out / "summary.json", summary)
    return summary


def load_pretrained(run_dir, config_path, device):
    """A finished Stage 1 run, rebuilt and checked against its wiring fingerprint."""
    run_dir = Path(run_dir)
    res = json.loads((run_dir / "results.json").read_text())
    if res.get("status") != "complete":
        raise RuntimeError(f"{run_dir} has status {res.get('status')!r}, not 'complete'")
    mcfg, tcfg = load_config(config_path)
    model = build_model(mcfg, res["arm"], res["seed"], ROOT / tcfg.region)
    if [model.brain.fingerprint] != res["fingerprints"]:
        raise RuntimeError(f"rebuilt {res['arm']}-{res['seed']} wiring does not match {run_dir}'s "
                           "fingerprint (numpy version drift?)")
    state = torch.load(run_dir / "ckpt.pt", map_location="cpu")
    if (state["arm"], state["seed"]) != (res["arm"], res["seed"]):
        raise RuntimeError(f"{run_dir}/ckpt.pt is {state['arm']}-{state['seed']}, "
                           f"not {res['arm']}-{res['seed']}")
    model.load_state_dict(state["model"])
    return model.to(device), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=FinetuneConfig.pairs)
    ap.add_argument("--lr", type=float, default=FinetuneConfig.lr)
    ap.add_argument("--steps", type=int, default=FinetuneConfig.steps)
    ap.add_argument("--mem-fraction", type=float)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mem_fraction and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    model, _ = load_pretrained(args.run, args.config, device)
    tok = Tokenizer.from_file(str(ROOT / "data" / "fineweb" / "tokenizer.json"))
    fcfg = FinetuneConfig(pairs=args.pairs, lr=args.lr, steps=args.steps)
    cues, answers = make_pairs(word_pool(tok), fcfg.pairs, fcfg.task_seed)
    s = run(model, cues, answers, fcfg, args.out, device, tok.token_to_id(EOT))
    print(json.dumps({k: s[k] for k in ("pairs", "start_acc", "final_acc", "steps_to_50", "steps_to_90")}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/nate/Projects/connectome-lm/.venv/bin/python -m pytest /home/nate/Projects/connectome-lm/tests/test_finetune.py -q`
Expected: `6 passed`.

If `test_brain_alone_can_memorise_a_short_list` fails: do not loosen the threshold. Rerun it once with `steps=1000` in a scratch script to tell "slow" from "cannot". Report both numbers to Nate before changing anything; a brain that cannot memorise 8 pairs through frozen projections would change the whole design.

- [ ] **Step 5: Run the full suite**

Run: `/home/nate/Projects/connectome-lm/.venv/bin/python -m pytest /home/nate/Projects/connectome-lm/tests -q`
Expected: `125 passed` (114 old + 5 + 6).

- [ ] **Step 6: Commit**

```bash
git -C /home/nate/Projects/connectome-lm add src/clm/finetune.py tests/test_finetune.py
git -C /home/nate/Projects/connectome-lm commit -m "finetune: brain-only training on the vocab task, quiz curve, fingerprint-checked loading

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Pilot script and a smoke run on the real checkpoint

**Files:**
- Create: `scripts/pilot_vocab.py`

- [ ] **Step 1: Write the script**

```python
# scripts/pilot_vocab.py
"""Pilot the vocabulary-list task on one er model.

    python scripts/pilot_vocab.py --pairs 512 --lrs 3e-4 1e-3 3e-3

Pilots touch only the er arm, so task settings cannot be tuned toward real
vs shuf. Each setting runs clm.finetune in its own process; settings with a
summary.json are skipped. Writes results/pilot_vocab.json.
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/er-0")
    ap.add_argument("--pairs", type=int, nargs="+", required=True)
    ap.add_argument("--lrs", type=float, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--mem-fraction", type=float, default=0.75)
    args = ap.parse_args()
    run_dir = ROOT / args.run
    arm = json.loads((run_dir / "results.json").read_text())["arm"]
    if arm != "er":
        sys.exit(f"{args.run} is arm {arm!r}; pilots use the er arm only")
    table_path = ROOT / "results" / "pilot_vocab.json"
    table = json.loads(table_path.read_text()) if table_path.exists() else {}
    for n, lr in itertools.product(args.pairs, args.lrs):
        name = f"{run_dir.name}-n{n}-lr{lr:g}-s{args.steps}"
        out = ROOT / "runs" / "pilot-vocab" / name
        if not (out / "summary.json").exists():
            code = subprocess.run([str(PY), "-m", "clm.finetune", "--run", str(run_dir),
                                   "--config", str(ROOT / "configs" / "stage1.json"),
                                   "--out", str(out), "--pairs", str(n), "--lr", str(lr),
                                   "--steps", str(args.steps),
                                   "--mem-fraction", str(args.mem_fraction)]).returncode
            if code != 0:
                sys.exit(f"{name}: clm.finetune exited {code}")
        s = json.loads((out / "summary.json").read_text())
        secs = [json.loads(l)["seconds"] for l in (out / "log.jsonl").read_text().splitlines()[1:]]
        row = {k: s[k] for k in ("pairs", "chance", "start_acc", "final_acc", "steps_to_50", "steps_to_90")}
        row.update(lr=lr, steps=args.steps, sec_per_step=round(sum(secs) / len(secs), 3))
        table[name] = row
        table_path.write_text(json.dumps(table, indent=2) + "\n")
        print(name, json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Check the script refuses a non-er run**

Run: `/home/nate/Projects/connectome-lm/.venv/bin/python /home/nate/Projects/connectome-lm/scripts/pilot_vocab.py --run runs/real-0 --pairs 128 --lrs 1e-3`
Expected: exits non-zero with `runs/real-0 is arm 'real'; pilots use the er arm only`. No directory `runs/pilot-vocab/real-0-*` appears.

- [ ] **Step 3: Smoke run, 20 steps, on the real er-0 checkpoint**

Check `nvidia-smi` first; wait if another job holds the GPU.

Run: `/home/nate/Projects/connectome-lm/.venv/bin/python -m clm.finetune --run /home/nate/Projects/connectome-lm/runs/er-0 --config /home/nate/Projects/connectome-lm/configs/stage1.json --out /tmp/clm-smoke --pairs 128 --lr 1e-3 --steps 20 --mem-fraction 0.75`
Expected: one JSON line with `"pairs": 128`, `start_acc` near 0 (chance is 0.0078), no traceback. Read `/tmp/clm-smoke/log.jsonl` for `seconds` per step; note it for Task 4.

- [ ] **Step 4: If Step 3 fails on the fingerprint check**

The error reads `rebuilt er-0 wiring does not match ... (numpy version drift?)`. Install the pod's numpy into the venv and rerun Step 3:
`uv pip install --python /home/nate/Projects/connectome-lm/.venv/bin/python numpy==2.5.4`
If it still fails, stop and report to Nate. Do not bypass the check: a wrong wiring would make every result meaningless.

- [ ] **Step 5: Commit**

```bash
git -C /home/nate/Projects/connectome-lm add scripts/pilot_vocab.py
git -C /home/nate/Projects/connectome-lm commit -m "pilot_vocab: er-only grid over list size and learning rate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Pilot phase 1, learning rate at 512 pairs

- [ ] **Step 1: Estimate the time**

Time = 3 settings × 1,500 steps × (seconds per step from Task 3 Step 3), plus about 10% for quizzes. If the estimate is over 4 hours, report it to Nate before starting.

- [ ] **Step 2: Launch**

Check `nvidia-smi` first. Then:

```bash
systemd-run --user --unit clm-pilot-vocab-1 --working-directory=/home/nate/Projects/connectome-lm /home/nate/Projects/connectome-lm/.venv/bin/python /home/nate/Projects/connectome-lm/scripts/pilot_vocab.py --pairs 512 --lrs 3e-4 1e-3 3e-3
```

Follow progress with `journalctl --user -u clm-pilot-vocab-1 -f`.

- [ ] **Step 3: Read the result**

Read `/home/nate/Projects/connectome-lm/results/pilot_vocab.json`. Pick the learning rate by this rule, written before the result:
- The highest `final_acc` wins. If two are within 0.02 of each other, take the smaller learning rate (more stable).
- **Stop rule:** if every `final_acc` is below 2 × chance (0.0039 at 512 pairs), stop. Report to Nate: the brain cannot store the list through frozen projections. The options are a longer run, or unfreezing `w_in`/`w_out`, which would make it no longer a brain-only test. That is his decision.

---

### Task 5: Pilot phase 2, list size, then record the choice

- [ ] **Step 1: Launch with the chosen learning rate**

Check `nvidia-smi` first. Replace `<LR>` with the Task 4 choice:

```bash
systemd-run --user --unit clm-pilot-vocab-2 --working-directory=/home/nate/Projects/connectome-lm /home/nate/Projects/connectome-lm/.venv/bin/python /home/nate/Projects/connectome-lm/scripts/pilot_vocab.py --pairs 128 1536 --lrs <LR>
```

- [ ] **Step 2: Choose the list size and training length**

Rules, written before the result:
- **List size:** the largest of 128, 512 and 1536 where `er-0` ends between 0.30 and 0.90 accuracy. That leaves room for a better or worse arm in either direction. If none land in that band, report the three accuracies to Nate.
- **Training length:** keep 1,500 steps if the chosen setting's accuracy is still rising over its last 300 steps (read `log.jsonl`). Otherwise use the first quiz step after which accuracy gains less than 0.01 per 300 steps.
- **Headline metric for the main run:** propose `final_acc` at the chosen length. It has one number per model and no threshold choice, and it pairs by seed like Stage 1. Report `steps_to_50` as secondary.

- [ ] **Step 3: Write `notes/pilot_vocab.md`**

Include: the full table from `results/pilot_vocab.json`, seconds per step, the chosen learning rate, list size and step count, which rule picked each, and the projected time and cost for all 15 models (laptop hours, and Vast cost at about US$0.27 per GPU-hour). Say plainly that the pilot used `er-0` only and read no `real` or `shuf` model.

- [ ] **Step 4: Commit**

```bash
git -C /home/nate/Projects/connectome-lm add results/pilot_vocab.json notes/pilot_vocab.md
git -C /home/nate/Projects/connectome-lm commit -m "Vocab-list pilot on er-0: chosen lr, list size and length

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Report to Nate**

Report the table, the choices and the projected cost. The 15-model run needs its own pre-registration (decision rules, SESOI, what each verdict leads to) and Nate's approval before it starts. That is a separate plan.
