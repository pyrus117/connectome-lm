"""Stage 2C-3 (PREREG_stage2c3.md): memory-on consolidation consistency (4a) and context-gated overlay writes (4b).

    .venv/bin/python scripts/stage2c3.py --runs runs2

Reads the write weight from results/stage2c_lists.json, runs the three consolidation variants (4a) and three
context-gated write variants (4b) on the finished COUNT-M seed 0 for task seeds 0, 1, 2, and writes
results/stage2c3.json.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from clm.consolidate_fix import run_consolidate_variant, run_write_gated
from clm.data import eval_windows, open_tokens
from clm.fastwrite_arms import ArmConfig
from clm.prep import EOT
from clm.train import ROOT
from clm.vocab_task import make_lists, make_pairs, word_pool
from stage2c import LISTS, PAIRS, TASK_SEEDS, fresh, load

PREREG = ROOT / "PREREG_stage2c3.md"
UNSEEN_PAIRS = 128

CONSOLIDATE_VARIANTS = {
    "original_teacher": dict(teacher_mode="original", freeze_head=False, memory_kl_weight=0.0),
    "original_teacher_memory_consistency": dict(teacher_mode="original", freeze_head=False, memory_kl_weight=1.0),
    "rolling_teacher_memory_consistency": dict(teacher_mode="rolling", freeze_head=False, memory_kl_weight=1.0),
}

WRITE_VARIANTS = {
    "write_gated_k3_c0": dict(gate_k=3, max_hcount=0),
    "write_gated_k2_c0": dict(gate_k=2, max_hcount=0),
    "write_gated_k2_c9": dict(gate_k=2, max_hcount=9),
}


def make_unseen_list(pool, lists, n_pairs, seed):
    """Held-out list of `n_pairs` (cue, answer) pairs drawn from `pool` words not used in `lists`."""
    used = set(np.concatenate([c for c, _ in lists] + [a for _, a in lists]).tolist())
    rest = np.array([w for w in pool if int(w) not in used], dtype=np.int64)
    return make_pairs(rest, n_pairs, seed + 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "runs2")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--mem-fraction", type=float, default=0.75)
    ap.add_argument("--steps", type=int, default=ArmConfig.steps)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(TASK_SEEDS))
    args = ap.parse_args()
    if not PREREG.exists():
        sys.exit("PREREG_stage2c3.md is missing: register the stage before running it")
    registered = args.out / "stage2c_lists.json"
    if not registered.exists():
        sys.exit(f"{registered} is missing: Stage 2C must have run first (it fixes the write weight)")
    weight = json.loads(registered.read_text())["write_weight"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    tok = Tokenizer.from_file(str(ROOT / "data/fineweb/tokenizer.json"))
    pool, eot, vocab = word_pool(tok), tok.token_to_id(EOT), tok.get_vocab_size()
    val = open_tokens(ROOT / "data/fineweb/val.bin")
    starts = eval_windows(len(val), 512, 2_097_152)
    lm_windows, anchor_windows = starts[:2048:32], starts[2048::32]
    count_model, reader, _ = load(args.runs / "count-M-0", device)

    runs = {}
    for seed in args.seeds:
        lists = make_lists(pool, LISTS, PAIRS, seed)
        unseen = make_unseen_list(pool, lists, UNSEEN_PAIRS, seed)
        cfg = ArmConfig(steps=args.steps, task_seed=seed)
        runs[str(seed)] = {"consolidate": {}, "write_gated": {}}
        for name, kw in WRITE_VARIANTS.items():
            t0 = time.time()
            r = run_write_gated(fresh(count_model), reader, lists, weight, vocab, eot, cfg, device, val, lm_windows, **kw)
            r["seconds"] = round(time.time() - t0, 2)
            runs[str(seed)]["write_gated"][name] = r
            print(f"seed {seed} {name}: final_mean {r['final_mean']:.3f} backward_transfer {r['backward_transfer']:.3f} "
                  f"lm_loss_on {r['lm_loss_memory_on'][-1]:.3f} (off {r['lm_loss_memory_off']:.3f})", flush=True)
        for name, kw in CONSOLIDATE_VARIANTS.items():
            t0 = time.time()
            r = run_consolidate_variant(fresh(count_model), reader, lists, weight, vocab, eot, cfg, device, val,
                                        lm_windows, anchor_windows, unseen_list=unseen, **kw)
            r["seconds"] = round(time.time() - t0)
            runs[str(seed)]["consolidate"][name] = r
            print(f"seed {seed} {name}: off_mean {r['final_mean']:.3f} on_mean {r['with_memory']['final_mean']:.3f} "
                  f"unseen {r['unseen_start']:.3f}->{r['unseen_with_memory'][-1]:.3f} "
                  f"lm_loss {r['lm_loss'][-1]:.3f}", flush=True)
    out = {"prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(), "lists": LISTS, "pairs_per_list": PAIRS,
           "unseen_pairs": UNSEEN_PAIRS, "steps_per_list": args.steps, "write_weight": weight, "runs": runs}
    (args.out / "stage2c3.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
