"""Stage 2N driver: which 64e-3 AdamW group kills the loop? The gained all-depth looped model with the plain COUNT read,
size M, in two arms completing the 2x2 factorial with Stage 2M's SHARED and GAINLOW (both have loop_gain at 8e-3):
- HEADLOW: head.weight and loop_gain at 8e-3, embeddings (embed, value_embed, ve_gate) at 64e-3 (P24).
- EMBEDLOW: embeddings and loop_gain at 8e-3, only head.weight at 64e-3 (P25).
Per-arm futility stop after seed 0. PREREG_stage2n.md. Resumable.

    .venv/bin/python scripts/stage2n_driver.py
"""
import json
import subprocess
import sys

import stage2_0_driver as base
from clm import rules2n as R
from stage2_0_driver import OUT, ROOT

base.CONFIG = "configs/stage2n.json"
HEADLOW, EMBEDLOW = "loop_count_headlow", "loop_count_embedlow"
ARMS = {HEADLOW: ["--gain-mult", "1.0", "--head-mult", "1.0"],
        EMBEDLOW: ["--adam-mult", "1.0", "--head-mult", "8.0"]}


def train(arm, seed, args):
    out = OUT / f"{arm}-M-{seed}"
    if base.train("M", seed, out, args) is None:                  # diverged: rerun once from scratch
        out.rename(out.with_name(out.name + "-diverged1"))
        if base.train("M", seed, out, args) is None:
            sys.exit(f"DIVERGED TWICE {out}; reported, no verdict")
    return json.loads((out / "results.json").read_text())


def main():
    r0 = {arm: train(arm, 0, a) for arm, a in ARMS.items()}
    s0 = {arm: R.summary(r["val_loss_by_depth"]) for arm, r in r0.items()}
    kept = []
    for arm in ARMS:
        if R.keep_arm(s0[arm]):
            print(f"FUTILITY {arm} continue", flush=True)
            kept.append(arm)
        else:
            print(f"FUTILITY {arm} stop", flush=True)
    for seed in (1, 2):
        for arm in kept:
            train(arm, seed, ARMS[arm])
    subprocess.run([sys.executable, str(ROOT / "scripts" / "stage2n_analyze.py")], check=True, cwd=ROOT, env=base.child_env())
    result = json.loads((ROOT / "results" / "stage2n.json").read_text())
    print("P24", result["p24_status"], "P25", result["p25_status"], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
