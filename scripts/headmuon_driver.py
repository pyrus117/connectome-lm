"""Per-head Muon side test, following PREREG_headmuon.md. Resumable like stage2_0_driver.

    .venv/bin/python scripts/headmuon_driver.py
"""
import json
import statistics
import sys
from pathlib import Path

from stage2_0_driver import OUT, PILOT_TOKENS, train

HERE = OUT / "headmuon"
GRID = [4e-3, 8e-3, 16e-3]
FUSED_LR = 8e-3
FUSED_SEED0 = OUT / "pilot" / "muon-0.008"


def run(opt, lr, seed):
    out = FUSED_SEED0 if (opt, lr, seed) == ("muon", FUSED_LR, 0) else HERE / f"{opt}-{lr:g}-s{seed}"
    r = train("M", seed, out, ["--optimizer", opt, "--lr", str(lr), "--tokens", str(PILOT_TOKENS)])
    if r is None:
        return None
    return {"loss": r["final_val_loss"], "s_per_step": r["train_seconds"] / r["steps"]}


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    sweep = {lr: run("muon_head", lr, 0) for lr in GRID}
    if all(v is None for v in sweep.values()):
        sys.exit("every muon_head rate diverged")
    best = min((lr for lr in sweep if sweep[lr]), key=lambda lr: sweep[lr]["loss"])
    if best in (GRID[0], GRID[-1]):
        step = best / 2 if best == GRID[0] else best * 2
        sweep[step] = run("muon_head", step, 0)
        best = min((lr for lr in sweep if sweep[lr]), key=lambda lr: sweep[lr]["loss"])
    print("CHOICE muon_head", best, flush=True)
    head = [sweep[best]] + [run("muon_head", best, s) for s in (1, 2)]
    fused = [run("muon", FUSED_LR, s) for s in (0, 1, 2)]
    if any(r is None for r in head + fused):
        sys.exit("a comparison run diverged; reported, no verdict")
    d = [h["loss"] - f["loss"] for h, f in zip(head, fused)]
    mean = statistics.fmean(d)
    if mean <= -0.01 and all(x < 0 for x in d):
        verdict = "BETTER"
    elif mean >= 0.01 and all(x > 0 for x in d):
        verdict = "WORSE"
    else:
        verdict = "NO DIFFERENCE"
    out = {"verdict": verdict, "muon_head_lr": best, "fused_lr": FUSED_LR,
           "sweep_seed0": {f"{lr:g}": v for lr, v in sorted(sweep.items())},
           "muon_head": head, "muon": fused, "d": d, "mean_d": mean,
           "sd_muon_head": statistics.stdev(h["loss"] for h in head),
           "sd_muon": statistics.stdev(f["loss"] for f in fused)}
    (HERE / "verdict.json").write_text(json.dumps(out, indent=2) + "\n")
    print("VERDICT", verdict, json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
