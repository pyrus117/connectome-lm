"""Stage 2C-2 (PREREG_stage2c2.md): the consolidation arm of Stage 2C with its teacher and head repaired.

    .venv/bin/python scripts/stage2c2.py --runs runs2

Reads the write weight from results/stage2c_lists.json, runs three variants on the finished COUNT-M seed 0 for task
seeds 0, 1, 2 and writes results/stage2c2_lists.json. The registered consolidate and write arms are not rerun; the
analysis note takes them from results/stage2c_lists.json.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from clm.consolidate_fix import run_consolidate_variant
from clm.data import eval_windows, open_tokens
from clm.fastwrite_arms import ArmConfig
from clm.prep import EOT
from clm.train import ROOT
from clm.vocab_task import make_lists, word_pool
from stage2c import LISTS, PAIRS, TASK_SEEDS, fresh, load

PREREG = ROOT / "PREREG_stage2c2.md"
VARIANTS = {"original_teacher": dict(teacher_mode="original", freeze_head=False),
            "original_teacher_frozen_head": dict(teacher_mode="original", freeze_head=True),
            "rolling_teacher_frozen_head": dict(teacher_mode="rolling", freeze_head=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "runs2")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--mem-fraction", type=float, default=0.75)
    ap.add_argument("--steps", type=int, default=ArmConfig.steps)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(TASK_SEEDS))
    args = ap.parse_args()
    if not PREREG.exists():
        sys.exit("PREREG_stage2c2.md is missing: register the stage before running it")
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
    lm_windows, anchor_windows = starts[:2048:32], starts[2048::32]       # the same windows as stage2c.py
    count_model, reader, _ = load(args.runs / "count-M-0", device)

    runs = {}
    for seed in args.seeds:
        lists = make_lists(pool, LISTS, PAIRS, seed)
        cfg = ArmConfig(steps=args.steps, task_seed=seed)
        runs[str(seed)] = {}
        for name, kw in VARIANTS.items():
            t0 = time.time()
            r = run_consolidate_variant(fresh(count_model), reader, lists, weight, vocab, eot, cfg, device, val,
                                        lm_windows, anchor_windows, **kw)
            r["seconds"] = round(time.time() - t0)
            runs[str(seed)][name] = r
            print(f"seed {seed} {name}: final_mean {r['final_mean']:.3f} backward_transfer {r['backward_transfer']:.3f} "
                  f"language loss {r['lm_loss'][-1]:.3f} (start {r['lm_loss_start']:.3f}) "
                  f"teacher recall on its own lists {[round(r['teacher_matrix'][i][i], 3) for i in range(LISTS)]}", flush=True)
    out = {"prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(), "lists": LISTS, "pairs_per_list": PAIRS,
           "steps_per_list": args.steps, "write_weight": weight, "runs": runs}
    (args.out / "stage2c2_lists.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
