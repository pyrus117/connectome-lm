"""Stage 2I (PREREG_stage2i.md): deployable exit rules against fixed depth on the three Stage 2H looped models
(`runs2/loop_gain_all-M-{0,1,2}`). Per seed it fits two linear probes on the last 4,096 windows of train.bin,
sets every threshold on that slice, then scores each rule's executed frontier on the validation windows.
Writes results/stage2i_seed<s>.json (skipped if present) and, once all seeds exist, results/stage2i.json.

    .venv/bin/python scripts/stage2i_analyze.py
"""
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

from clm import depth_eval as E
from clm import depth_law as D
from clm import halting as H
from clm.data import open_tokens
from clm.s2loop import build_loop
from clm.s2model import B0Config
from clm.s2train import S2TrainConfig
from clm.train import ROOT

SEEDS = (0, 1, 2)
K0, TAU, EFFORTS = 0.2, 10 / math.log(100), list(range(11))
QUANTILES = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97)
FIT_WINDOWS, STRIDE, ALPHAS = 4096, 8, [1e1, 1e2, 1e3, 1e4, 1e5]
RULES = ("change", "entropy", "margin", "probe_small", "probe_state")
CANDIDATES = RULES[1:]                       # the rules P18 is about; "change" repeats Stage 2D's


def load_model(run, device):
    res = json.loads((run / "results.json").read_text())
    mcfg, tcfg = B0Config(**res["model"]), S2TrainConfig(**res["train"])
    if not (tcfg.loop_gain and tcfg.loop_loss == "all"):
        sys.exit(f"{run.name} is not a gained all-depth run")
    model = build_loop(mcfg, res["seed"], tcfg.loop_max, gain=True).to(device)
    model.load_state_dict(torch.load(run / "ckpt.pt", map_location=device)["model"])
    return model, mcfg, tcfg


def fit_probes(small, state, gains):
    """One ridge per pass 1..R-1 for each probe variant; returns both lists and a record of the fits."""
    probes, record = {"probe_small": [], "probe_state": []}, {"probe_small": [], "probe_state": []}
    for r in range(small.shape[0]):
        y = gains[r].astype(np.float64)
        for name, X in (("probe_small", small[r]), ("probe_state", np.concatenate([small[r], state[r].astype(np.float32)], axis=1))):
            p = H.ridge_fit(X, y, ALPHAS)
            probes[name].append(p)
            tail = y[int(len(y) * 0.75):]
            record[name].append({"pass": r + 1, "alpha": p["alpha"], "holdout_mse": p["holdout_mse"][p["alpha"]],
                                 "holdout_var_of_gain": float(tail.var()), "samples": int(len(y))})
    return probes, record


def thresholds(small):
    """Per quantile, the threshold that makes that share of fit-slice (pass, token) values stop (passes 1..R-1 pooled)."""
    pooled = {"entropy": small[:, :, 0].ravel(), "margin": small[:, :, 1].ravel(), "change": small[:, :, 2].ravel()}
    return {q: {rule: H.level_for_share(rule, v, q) for rule, v in pooled.items()} for q in QUANTILES}


def analyze_seed(seed, device):
    run = ROOT / "runs2" / f"loop_gain_all-M-{seed}"
    model, mcfg, tcfg = load_model(run, device)
    val = open_tokens(ROOT / tcfg.val_bin)
    train = open_tokens(ROOT / tcfg.train_bin)
    starts = (len(train) - 1 - FIT_WINDOWS * mcfg.ctx) + np.arange(FIT_WINDOWS, dtype=np.int64) * mcfg.ctx
    small, state, gains = E.collect_fit(model, train, starts, mcfg.ctx, tcfg.micro, device, STRIDE)
    print(f"seed {seed}: fit slice {small.shape[1]} samples", flush=True)
    probes, record = fit_probes(small, state, gains)
    theta = thresholds(small)
    del state

    predictors = {k: H.probe_predictor(v, k == "probe_state") for k, v in probes.items()}
    sig = E.collect_signals(model, val, mcfg.ctx, tcfg.eval_tokens, tcfg.micro, device, predictors)
    args = (val, mcfg.ctx, tcfg.eval_tokens, tcfg.micro, device)
    R, n = sig["loss"].shape
    fixed = [(r, float(sig["loss"][r - 1].mean())) for r in range(1, R + 1)]
    if abs(fixed[-1][1] - json.loads((run / "results.json").read_text())["final_val_loss"]) > 2e-3:
        sys.exit("re-evaluated depth-6 loss differs from results.json")

    frontiers = {}
    for rule in RULES:
        levels = ([(q, theta[q][rule]) for q in QUANTILES] if rule in ("change", "entropy", "margin")
                  else [(b, float(D.price(b, K0, TAU))) for b in EFFORTS])
        pts = []
        for tag, level in levels:
            depths = H.exit_depths(H.stop_matrix(rule, sig, level))
            pts.append({"level_tag": tag, "level": level, "mean_depth": float(depths.mean()),
                        "executed_loss": E.executed_loss(model, *args, depths)})
            print(f"  {rule} {tag}: depth {pts[-1]['mean_depth']:.2f} loss {pts[-1]['executed_loss']:.4f}", flush=True)
        frontiers[rule] = pts
    pts = []
    for b in EFFORTS:
        depths = D.oracle_depths(sig["loss"], D.price(b, K0, TAU))
        pts.append({"level_tag": b, "level": float(D.price(b, K0, TAU)), "mean_depth": float(depths.mean()),
                    "executed_loss": E.executed_loss(model, *args, depths)})
    frontiers["oracle"] = pts

    scores = {r: H.region_scores([(p["mean_depth"], p["executed_loss"]) for p in f], fixed) for r, f in frontiers.items()}
    out = {"seed": seed, "tokens": n, "fixed_depth": fixed, "frontiers": frontiers, "region_scores": scores,
           "probe_fits": record, "thresholds": {str(q): v for q, v in theta.items()}, "fit_samples": int(small.shape[1])}
    (ROOT / "results" / f"stage2i_seed{seed}.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def summarize(per_seed):
    by = {r: {s: per_seed[s]["region_scores"][r] for s in SEEDS} for r in RULES + ("oracle",)}
    oracle_gap = (statistics.fmean(by["oracle"][s]["gap"] for s in SEEDS)
                  if all(by["oracle"][s]["gap"] is not None for s in SEEDS) else None)
    rows = {}
    for r in RULES:
        gaps = [by[r][s]["gap"] for s in SEEDS]
        scored = all(g is not None for g in gaps)
        rows[r] = {"verdict": H.beats_fixed(by[r]), "gap_by_seed": gaps, "mean_gap": statistics.fmean(gaps) if scored else None,
                   "share_of_oracle_gap": statistics.fmean(gaps) / oracle_gap if scored and oracle_gap else None}
    p18 = any(rows[r]["verdict"] == "X_BETTER" for r in CANDIDATES)
    return {"region": H.REGION, "sesoi": 0.01, "oracle_mean_gap": oracle_gap, "rules": rows,
            "p18_supported": p18, "winners": [r for r in CANDIDATES if rows[r]["verdict"] == "X_BETTER"]}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_seed = {}
    for s in SEEDS:
        f = ROOT / "results" / f"stage2i_seed{s}.json"
        per_seed[s] = json.loads(f.read_text()) if f.exists() else analyze_seed(s, device)
    out = summarize(per_seed)
    (ROOT / "results" / "stage2i.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
