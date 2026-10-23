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
from clm.vocab_task import (IGNORE, correct, make_lists, make_pairs, make_twin_pairs, quiz_batches,
                            train_batch, word_pool)
from clm.word_sim import cached_similarity


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



def run_sequential(model, lists, fcfg, out, device, eot):
    """Learn lists one after another, fcfg.steps each, quizzing every list.

    acc_matrix[i][j] is list j's accuracy at the end of phase i. Summary
    follows Lopez-Paz & Ranzato (2017): final_mean is the mean of the last
    row; backward_transfer is the mean, over all lists but the last, of
    (final accuracy - accuracy right after its own phase); negative means
    forgetting."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.to(device)
    params = freeze_all_but_brain(model)
    opt = torch.optim.AdamW(params, lr=fcfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    quizzes = [quiz_batches(c, a, fcfg.per_seq, fcfg.task_seed, eot) for c, a in lists]

    def quiz_all():
        return [quiz(model, qx, qy, fcfg.batch, device) for qx, qy in quizzes]

    acc_matrix, step = [], 0
    with open(out / "log.jsonl", "w") as log:
        log.write(json.dumps({"step": 0, "phase": -1, "quiz": quiz_all()}) + "\n")
        for phase, (cues, answers) in enumerate(lists):
            for k in range(fcfg.steps):
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
                step += 1
                rec = {"step": step, "phase": phase, "loss": loss.item(), "grad_norm": gnorm,
                       "seconds": round(time.time() - t0, 3)}
                if (k + 1) % fcfg.quiz_every == 0 or k + 1 == fcfg.steps:
                    rec["quiz"] = quiz_all()
                log.write(json.dumps(rec) + "\n")
                log.flush()
            acc_matrix.append(rec["quiz"])
    last = acc_matrix[-1]
    summary = {"lists": len(lists), "pairs_per_list": len(lists[0][0]), "chance": 1 / len(lists[0][0]),
               "acc_matrix": acc_matrix, "final_mean": sum(last) / len(last),
               "backward_transfer": (sum(last[j] - acc_matrix[j][j] for j in range(len(lists) - 1))
                                     / max(1, len(lists) - 1)),
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
    ap.add_argument("--task", choices=("random", "twins", "sequential"), default="random")
    ap.add_argument("--pairs", type=int, default=FinetuneConfig.pairs,
                    help="random/twins: list size; sequential: pairs per list")
    ap.add_argument("--lists", type=int, default=4, help="sequential only")
    ap.add_argument("--lr", type=float, default=FinetuneConfig.lr)
    ap.add_argument("--steps", type=int, default=FinetuneConfig.steps,
                    help="random/twins: total; sequential: per list")
    ap.add_argument("--mem-fraction", type=float)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mem_fraction and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    model, _ = load_pretrained(args.run, args.config, device)
    tok = Tokenizer.from_file(str(ROOT / "data" / "fineweb" / "tokenizer.json"))
    fcfg = FinetuneConfig(pairs=args.pairs, lr=args.lr, steps=args.steps)
    pool, eot = word_pool(tok), tok.token_to_id(EOT)
    if args.task == "sequential":
        s = run_sequential(model, make_lists(pool, args.lists, fcfg.pairs, fcfg.task_seed), fcfg,
                           args.out, device, eot)
        print(json.dumps({k: s[k] for k in ("lists", "pairs_per_list", "final_mean", "backward_transfer")}))
        return
    if args.task == "twins":
        sim = cached_similarity(ROOT / "data" / "fineweb" / "train.bin", pool, tok.get_vocab_size(),
                                ROOT / "data" / "fineweb" / "pool_sim.npz")
        cues, answers = make_twin_pairs(pool, sim, fcfg.pairs, fcfg.task_seed)
    else:
        cues, answers = make_pairs(pool, fcfg.pairs, fcfg.task_seed)
    s = run(model, cues, answers, fcfg, args.out, device, eot)
    print(json.dumps({k: s[k] for k in ("pairs", "start_acc", "final_acc", "steps_to_50", "steps_to_90")}))


if __name__ == "__main__":
    main()
