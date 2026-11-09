"""PREREG_stage2h.md section 5: P17 (the looped model with a per-pass gain and the loss on every depth against Stage 2D's
looped model, best-depth validation loss, three seeds), the two secondary contrasts and the ablations. Writes
results/stage2h.json.

    .venv/bin/python scripts/stage2h_analyze.py
"""
import hashlib
import json
import statistics
import sys

import torch

import stage2a_analyze as A
from clm import rules2h as R

ROOT, RUNS = A.ROOT, A.RUNS
STATE = RUNS / "stage2h" / "futility.json"
SEEDS = (0, 1, 2)
# arm: (run directory, results arm, loop_gain, loop_loss, micro)
ARMS = {"reference": ("loop-M-{}", "loop", False, "sampled", 8),
        "both": ("loop_gain_all-M-{}", "loop_gain_all", True, "all", 4),
        "gain": ("loop_gain-M-{}", "loop_gain", True, "sampled", 8),
        "all": ("loop_all-M-{}", "loop_all", False, "all", 4)}
SHARED = {"optimizer": "muon", "lr": 0.008, "adam_mult": 1.0, "loop_max": 6, "arch": "loop", "memory": "none"}


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def load(arm, seed):
    """A finished run, checked against its arm. Stage 2D's seed 0 predates this stage: it carries Stage 2D's
    pre-registration hash and has no loop_gain, loop_loss or adam_mult setting (they default to off, sampled and 1.0, one shared rate)."""
    name, results_arm, gain, loss, micro = ARMS[arm]
    own = "PREREG_stage2d.md" if (arm, seed) == ("reference", 0) else "PREREG_stage2h.md"
    r = A.load(RUNS / name.format(seed) / "results.json", results_arm, sha(own))
    want = {**SHARED, "loop_gain": gain, "loop_loss": loss, "micro": micro}
    got = {k: r["train"].get(k, False if k == "loop_gain" else "sampled" if k == "loop_loss" else 1.0 if k == "adam_mult" else None) for k in want}
    if got != want:
        sys.exit(f"{name.format(seed)} trained with {got}, expected {want}")
    if len(r["val_loss_by_depth"]) != 6:
        sys.exit(f"{name.format(seed)} reports {len(r['val_loss_by_depth'])} depths, expected 6")
    return r


def gain_stats(arm, seed):
    g = torch.load(RUNS / ARMS[arm][0].format(seed) / "ckpt.pt", map_location="cpu")["model"]["loop_gain"].float()
    return [{"pass": r + 1, "mean": g[r].mean().item(), "std": g[r].std().item(), "min": g[r].min().item(),
             "max": g[r].max().item(), "distance_from_one": (g[r] - 1).norm().item()} for r in range(g.shape[0])]


def main():
    stopped = json.loads(STATE.read_text())["stop"] if STATE.exists() else False
    main_seeds = (0,) if stopped else SEEDS
    runs = {"reference": {s: load("reference", s) for s in main_seeds}, "both": {s: load("both", s) for s in main_seeds}}
    for arm in ("gain", "all"):
        if (RUNS / ARMS[arm][0].format(0) / "results.json").exists():
            runs[arm] = {0: load(arm, 0)}
    summ = {arm: {s: R.summary(r["val_loss_by_depth"]) for s, r in rs.items()} for arm, rs in runs.items()}
    out = {"futility_stop": stopped, "sesoi": R.SESOI, "seeds": list(main_seeds),
           "val_loss_by_depth": {arm: {s: r["val_loss_by_depth"] for s, r in rs.items()} for arm, rs in runs.items()},
           "summary": summ}
    if stopped:
        out["p17_status"], out["p17"] = "NOT_SUPPORTED_FUTILITY", False
        out["seed0_difference_best_loss"] = summ["both"][0]["best_loss"] - summ["reference"][0]["best_loss"]
    else:
        out["contrasts"] = {key: R.contrast(summ["both"], summ["reference"], key) for key in ("best_loss", "depth1", "deepest")}
        out["mean_d"] = {key: statistics.fmean(summ["both"][s][key] - summ["reference"][s][key] for s in SEEDS)
                         for key in out["contrasts"]}
        out["d_by_seed"] = {key: {s: summ["both"][s][key] - summ["reference"][s][key] for s in SEEDS} for key in out["contrasts"]}
        out["p17"] = R.p17(summ["both"], summ["reference"])
        out["p17_status"] = out["contrasts"]["best_loss"]
    b0 = [json.loads((RUNS / f"b0-M-{s}" / "results.json").read_text())["final_val_loss"] for s in SEEDS]
    out["b0_m_final_val_loss"], out["b0_m_mean"] = b0, statistics.fmean(b0)
    out["mean_gains"] = {arm: [statistics.fmean(v["gains"][i] for v in sm.values()) for i in range(5)] for arm, sm in summ.items()}
    out["learned_gain"] = {arm: {s: gain_stats(arm, s) for s in rs} for arm, rs in runs.items() if arm in ("both", "gain")}
    out["cost"] = {arm: {"s_per_step": statistics.fmean(r["train_seconds"] / r["steps"] for r in rs.values()),
                         "peak_vram_mib": max(r["peak_vram_mib"] for r in rs.values())} for arm, rs in runs.items()}
    (ROOT / "results" / "stage2h.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("futility_stop", "p17_status", "summary", "mean_d") if k in out}, indent=2))
    return out


if __name__ == "__main__":
    main()
