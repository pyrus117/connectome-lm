"""Stage 2L driver: the gained all-depth looped model (Stage 2H's `both`) with and without the COUNT read, on the
separate Muon and AdamW rates, size M, three seeds, with Stage 2H's futility stop after seed 0. The COUNT read is the
one Stage 2K2 carried forward (results/stage2k2.json). PREREG_stage2l.md. Resumable like the others.

    .venv/bin/python scripts/stage2l_driver.py
"""
import json
import subprocess
import sys

import stage2_0_driver as base
from clm import rules2h as R
from stage2_0_driver import OUT, ROOT

base.CONFIG = "configs/stage2l.json"
READS = {"count": [], "gated_low": ["--count-gate", "--gate-mult", "1.0"], "scaled_low": ["--count-scale", "--gate-mult", "1.0"],
         "oscaled_low": ["--count-scale-orders", "--gate-mult", "1.0"]}
LOOP, BOTH = "loop_gain_all_sep", "loop_gain_all_count_sep"


def train(arm, seed, args):
    out = OUT / f"{arm}-M-{seed}"
    if base.train("M", seed, out, args) is None:                  # diverged: rerun once from scratch
        out.rename(out.with_name(out.name + "-diverged1"))
        if base.train("M", seed, out, args) is None:
            sys.exit(f"DIVERGED TWICE {out}; reported, no verdict")
    return json.loads((out / "results.json").read_text())


def main():
    carry = json.loads((ROOT / "results" / "stage2k2.json").read_text())["carry_forward"]
    (OUT / "stage2l_choice.json").write_text(json.dumps({"read": carry, "args": READS[carry]}, indent=2) + "\n")
    print("READ", carry, flush=True)
    args = {LOOP: ["--memory", "none"], BOTH: READS[carry]}
    r0 = {arm: train(arm, 0, a) for arm, a in args.items()}
    s = {arm: R.summary(r["val_loss_by_depth"]) for arm, r in r0.items()}
    if R.futile(s[BOTH], s[LOOP]):                                # PREREG_stage2l.md section 3; prints no numbers
        print("FUTILITY stop", flush=True)
    else:
        print("FUTILITY continue", flush=True)
        for seed in (1, 2):
            for arm, a in args.items():
                train(arm, seed, a)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "stage2l_analyze.py")], check=True, cwd=ROOT, env=base.child_env())
    result = json.loads((ROOT / "results" / "stage2l.json").read_text())
    print("P21", result["p21_status"], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
