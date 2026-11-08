"""Stage 2.0 driver: the optimizer/learning-rate pilot on B0-M, then the size ladder.

Follows PREREG_stage2_0.md exactly. Resumable: a run with results.json is
skipped, an interrupted run resumes from its checkpoint.

    .venv/bin/python scripts/stage2_0_driver.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WINDOWS = os.name == "nt"
PY = ROOT / ".venv" / ("Scripts/python.exe" if WINDOWS else "bin/python")


def child_env(**extra):
    env = {"PYTHONPATH": str(ROOT / "src"), **extra}
    if WINDOWS:     # torch needs SYSTEMROOT and the CUDA DLL search path; expandable_segments is Linux-only
        env.update({k: os.environ[k] for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP") if k in os.environ})
    else:
        env.update({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PATH": "/usr/bin:/bin"})
    return env
CONFIG = "configs/stage2_0.json"
OUT = ROOT / "runs2"
PILOT_TOKENS = 50_000_000
GRID = {"adamw": [1e-3, 2e-3, 4e-3], "muon": [2e-3, 4e-3, 8e-3]}
LADDER = [(size, seed) for size in ("M", "S", "L") for seed in (0, 1, 2)]


def train(size, seed, out, extra=()):
    out = Path(out)
    if (out / "results.json").exists():
        return json.loads((out / "results.json").read_text())
    cmd = [str(PY), "-m", "clm.s2train", "--config", CONFIG, "--size", size, "--seed", str(seed),
           "--out", str(out), *extra]
    print("RUN", " ".join(cmd[2:]), flush=True)
    code = subprocess.run(cmd, cwd=ROOT, env=child_env()).returncode
    if code == 2:
        print(f"DIVERGED {out}", flush=True)
        return None
    if code != 0:
        sys.exit(f"{out} failed with exit code {code}")
    return json.loads((out / "results.json").read_text())


def pilot(opt, lr):
    r = train("M", 0, OUT / "pilot" / f"{opt}-{lr:g}",
              ["--optimizer", opt, "--lr", str(lr), "--tokens", str(PILOT_TOKENS)])
    return None if r is None else r["final_val_loss"]


def main():
    scores = {(opt, lr): pilot(opt, lr) for opt, lrs in GRID.items() for lr in lrs}
    finite = {k: v for k, v in scores.items() if v is not None}
    if not finite:
        sys.exit("every pilot run diverged")
    best_opt, best_lr = min(finite, key=finite.get)
    lrs = GRID[best_opt]
    if best_lr in (lrs[0], lrs[-1]):          # optimum on the grid edge: one step further, once
        step = best_lr / 2 if best_lr == lrs[0] else best_lr * 2
        scores[(best_opt, step)] = pilot(best_opt, step)
        finite = {k: v for k, v in scores.items() if v is not None}
        best_opt, best_lr = min(finite, key=finite.get)
    choice = {"optimizer": best_opt, "lr": best_lr,
              "pilot": [{"optimizer": o, "lr": l, "val_loss": v} for (o, l), v in sorted(scores.items())]}
    (OUT / "pilot" / "choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print("CHOICE", choice["optimizer"], choice["lr"], flush=True)
    for size, seed in LADDER:
        out, args = OUT / f"b0-{size}-{seed}", ["--optimizer", best_opt, "--lr", str(best_lr)]
        if train(size, seed, out, args) is None:          # diverged: rerun once from scratch
            out.rename(out.with_name(out.name + "-diverged1"))
            if train(size, seed, out, args) is None:
                print(f"DIVERGED TWICE {out}; reported, fitted without it", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
