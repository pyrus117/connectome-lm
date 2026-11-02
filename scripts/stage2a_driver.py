"""Stage 2A driver: the COUNT learning-rate pilot, three COUNT-M seeds, the
leakage diagnostic, and the paired verdict against the frozen Stage 2.0 B0-M
runs. Follows PREREG_stage2a.md exactly. Resumable like stage2_0_driver.

    .venv/bin/python scripts/stage2a_driver.py
"""
import json
import subprocess
import sys
from pathlib import Path

import stage2_0_driver as base
from stage2_0_driver import OUT, PILOT_TOKENS, ROOT

CONFIG = "configs/stage2a.json"
base.CONFIG = CONFIG                                     # base.train reads its module's CONFIG
PILOT = OUT / "pilot_count"
GRID = [4e-3, 8e-3, 16e-3]
SEEDS = (0, 1, 2)


def run(size, seed, out, extra=()):
    return base.train(size, seed, out, extra)


def pilot(lr):
    r = run("M", 0, PILOT / f"count-{lr:g}", ["--lr", str(lr), "--tokens", str(PILOT_TOKENS)])
    return None if r is None else r["final_val_loss"]


def main():
    PILOT.mkdir(parents=True, exist_ok=True)
    scores = {lr: pilot(lr) for lr in GRID}
    finite = {k: v for k, v in scores.items() if v is not None}
    if not finite:
        sys.exit("every COUNT pilot rate diverged")
    best = min(finite, key=finite.get)
    if best in (GRID[0], GRID[-1]):                      # optimum on the grid edge: one step further, once
        step = best / 2 if best == GRID[0] else best * 2
        scores[step] = pilot(step)
        finite = {k: v for k, v in scores.items() if v is not None}
        best = min(finite, key=finite.get)
    (PILOT / "choice.json").write_text(json.dumps(
        {"lr": best, "pilot": [{"lr": k, "val_loss": v} for k, v in sorted(scores.items())]}, indent=2) + "\n")
    print("CHOICE count lr", best, flush=True)
    args = ["--lr", str(best)]
    b0_args = ["--lr", "0.008", "--optimizer", "muon", "--memory", "none", "--prereg", "PREREG_stage2_0.md"]

    def seeds(which):
        for seed in which:
            out = OUT / f"count-M-{seed}"
            if run("M", seed, out, args) is None:        # diverged: rerun once from scratch
                out.rename(out.with_name(out.name + "-diverged1"))
                if run("M", seed, out, args) is None:
                    sys.exit(f"DIVERGED TWICE {out}; reported, no verdict")

    def analyse():
        names = [str(p) for p in sorted(OUT.glob("count-M-[0-9]")) + sorted(OUT.glob("b0-M-[0-9]"))
                 + [OUT / "count-noloo-M-0"] if (p / "results.json").exists()]
        subprocess.run([sys.executable, str(ROOT / "scripts" / "bucket_eval.py"), *names], check=True,
                       cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
        subprocess.run([sys.executable, str(ROOT / "scripts" / "stage2a_analyze.py")], check=True,
                       cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
        return json.loads((ROOT / "results" / "stage2a.json").read_text())

    seeds(SEEDS)
    run("M", 0, OUT / "count-noloo-M-0", [*args, "--no-loo"])
    result = analyse()
    if result["status"] == "NEEDS_SEEDS":                # PREREG_stage2a.md section 5
        seeds((3, 4))
        for seed in (3, 4):
            run("M", seed, OUT / f"b0-M-{seed}", b0_args)
        result = analyse()
    print("VERDICT", result["verdict"], "gate 1", result["gate1"], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
