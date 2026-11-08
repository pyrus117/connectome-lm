"""Stage 2H driver (follow-up 5A): the looped model with a per-pass gain and the loss on every depth ("both"), against
Stage 2D's looped model ("reference"), plus two single-seed ablations ("gain" only, "all" only). PREREG_stage2h.md.
Order: both seed 0, the futility check, then (unless it stops) reference and both seeds 1 and 2, then the ablations, then
the analysis. Resumable: a finished run is skipped, an interrupted one resumes from its checkpoint.

    .venv/bin/python scripts/stage2h_driver.py
"""
import json
import subprocess
import sys

import stage2_0_driver as base
from clm import rules2h as R
from stage2_0_driver import OUT, ROOT

base.CONFIG = "configs/stage2h.json"
ENV = base.child_env()
STATE = OUT / "stage2h"
ARMS = {"reference": ([], "loop-M-{}"),
        "both": (["--loop-gain", "--loop-loss", "all", "--micro", "4"], "loop_gain_all-M-{}"),
        "gain": (["--loop-gain"], "loop_gain-M-{}"),
        "all": (["--loop-loss", "all", "--micro", "4"], "loop_all-M-{}")}


def train(arm, seed):
    args, name = ARMS[arm]
    out = OUT / name.format(seed)
    if base.train("M", seed, out, args) is None:                       # diverged: rerun once from scratch
        out.rename(out.with_name(out.name + "-diverged1"))
        if base.train("M", seed, out, args) is None:
            sys.exit(f"DIVERGED TWICE {out}; reported, no verdict")
    return json.loads((out / "results.json").read_text())


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    new0, ref0 = train("both", 0), train("reference", 0)
    stop = R.futile(R.summary(new0["val_loss_by_depth"]), R.summary(ref0["val_loss_by_depth"]))
    (STATE / "futility.json").write_text(json.dumps({"stop": stop}) + "\n")
    print("FUTILITY", "stop" if stop else "continue", flush=True)
    if not stop:
        for seed in (1, 2):
            train("both", seed)
            train("reference", seed)
    train("gain", 0)
    train("all", 0)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "stage2h_analyze.py")], check=True, cwd=ROOT, env=ENV)
    result = json.loads((ROOT / "results" / "stage2h.json").read_text())
    print("P17", result["p17_status"], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
