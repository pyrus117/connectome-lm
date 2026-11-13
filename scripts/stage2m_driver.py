"""Stage 2M driver: why the loop passes were inert in Stage 2L. The gained all-depth looped model with the plain COUNT
read, size M, three seeds, in two arms: (GAINLOW) the separate-rate recipe with only the loop gain's AdamW rate put back
at 8e-3, and (SHARED) Stage 2H's one shared rate. Futility stop after seed 0. PREREG_stage2m.md. Resumable like the others.

    .venv/bin/python scripts/stage2m_driver.py
"""
import json
import subprocess
import sys

import stage2_0_driver as base
from clm import rules2m as R
from stage2_0_driver import OUT, ROOT

base.CONFIG = "configs/stage2m.json"
GAINLOW, SHARED = "loop_count_gainlow", "loop_count_shared"
ARMS = {GAINLOW: ["--gain-mult", "1.0"], SHARED: ["--adam-mult", "1.0"]}


def train(arm, seed, args):
    out = OUT / f"{arm}-M-{seed}"
    if base.train("M", seed, out, args) is None:                  # diverged: rerun once from scratch
        out.rename(out.with_name(out.name + "-diverged1"))
        if base.train("M", seed, out, args) is None:
            sys.exit(f"DIVERGED TWICE {out}; reported, no verdict")
    return json.loads((out / "results.json").read_text())


def main():
    r0 = {arm: train(arm, 0, a) for arm, a in ARMS.items()}
    s = {arm: R.summary(r["val_loss_by_depth"]) for arm, r in r0.items()}
    if R.futile(s[GAINLOW], s[SHARED]):                           # PREREG_stage2m.md section 3; prints no numbers
        print("FUTILITY stop", flush=True)
    else:
        print("FUTILITY continue", flush=True)
        for seed in (1, 2):
            for arm, a in ARMS.items():
                train(arm, seed, a)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "stage2m_analyze.py")], check=True, cwd=ROOT, env=base.child_env())
    result = json.loads((ROOT / "results" / "stage2m.json").read_text())
    print("P22", result["p22_status"], "P23", result["p23_status"], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
