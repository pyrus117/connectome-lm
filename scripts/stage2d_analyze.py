"""Stage 2D analysis (PREREG_stage2d.md): per-token loss at every depth of the trained
looped model, the exact optimal-depth frontier, the effort law P9, the executed
frontiers (oracle depths and a free-exit rule), and the RL gate. Writes
results/stage2d.json, or results/<name> if a second argument is given (Stage 2H
repeats it on a gained model: loop_gain is read from the run's own settings).

    .venv/bin/python scripts/stage2d_analyze.py [runs2/loop-M-0 [stage2d.json]]
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
from clm.analyze import SESOI
from clm.data import open_tokens
from clm.s2loop import build_loop
from clm.s2model import B0Config
from clm.s2train import S2TrainConfig
from clm.train import ROOT

K0, TAU, EFFORTS = 0.2, 10 / math.log(100), list(range(11))          # price 0.2 nats/pass at b = 0, 0.002 at b = 10
THETA_QUANTILES = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97)


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs2" / "loop-M-0")
    res = json.loads((run / "results.json").read_text())
    mcfg, tcfg = B0Config(**res["model"]), S2TrainConfig(**res["train"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_loop(mcfg, res["seed"], tcfg.loop_max, gain=tcfg.loop_gain).to(device)
    model.load_state_dict(torch.load(run / "ckpt.pt", map_location=device)["model"])
    val = open_tokens(ROOT / tcfg.val_bin)
    args = (val, mcfg.ctx, tcfg.eval_tokens, tcfg.micro, device)
    R = tcfg.loop_max

    losses, change = E.collect(model, *args)
    n = losses.shape[1]
    mean_by_depth = losses.mean(axis=1)
    if abs(mean_by_depth[-1] - res["final_val_loss"]) > 2e-3:
        sys.exit(f"re-evaluated loss at depth {R} {mean_by_depth[-1]:.4f} differs from results.json {res['final_val_loss']:.4f}")
    b0 = [json.loads((ROOT / "runs2" / f"b0-M-{s}" / "results.json").read_text())["final_val_loss"] for s in (0, 1, 2)]
    fixed = [{"depth": r, "layers_run": 4 + 2 * r, "mean_loss": float(mean_by_depth[r - 1])} for r in range(1, R + 1)]

    oracle = []
    for row in D.frontier(losses, EFFORTS, K0, TAU):
        depths = D.oracle_depths(losses, row["price"])
        oracle.append({**row, "executed_loss": E.executed_loss(model, *args, depths),
                       "share_at_depth": [float((depths == r).mean()) for r in range(1, R + 1)]})
        oracle[-1]["cross_token_effect"] = oracle[-1]["executed_loss"] - row["mean_loss"]
        print(f"effort {row['effort']:.0f}: depth {row['mean_depth']:.2f} predicted {row['mean_loss']:.4f} "
              f"executed {oracle[-1]['executed_loss']:.4f}", flush=True)

    thetas = np.quantile(change, THETA_QUANTILES)
    free = []
    for q, theta in zip(THETA_QUANTILES, thetas):
        depths = E.free_exit_depths(change, theta)
        free.append({"quantile": q, "theta": float(theta), "mean_depth": float(depths.mean()),
                     "executed_loss": E.executed_loss(model, *args, depths)})
        print(f"free exit q{q}: depth {free[-1]['mean_depth']:.2f} loss {free[-1]['executed_loss']:.4f}", flush=True)

    free_pts = [(f["mean_depth"], f["executed_loss"]) for f in free]
    fixed_pts = [(f["depth"], f["mean_loss"]) for f in fixed]
    gaps_free, gaps_fixed = [], []
    for o in oracle:
        lf, lx = D.interpolate_loss(free_pts, o["mean_depth"]), D.interpolate_loss(fixed_pts, o["mean_depth"])
        o["free_exit_loss_at_same_depth"], o["fixed_depth_loss_at_same_depth"] = lf, lx
        if lf is not None:
            gaps_free.append(lf - o["executed_loss"])
        if lx is not None:
            gaps_fixed.append(lx - o["executed_loss"])
    gap_free = statistics.fmean(gaps_free) if gaps_free else None
    out = {
        "run": run.name, "tokens": n, "max_depth": R, "price": {"k0": K0, "tau": TAU},
        "fixed_depth": fixed, "b0_m_final_val_loss": b0, "b0_m_mean": statistics.fmean(b0),
        "marginal_gains": D.marginal_gains(mean_by_depth).tolist(),
        "geometric_decay": D.geometric_decay(D.marginal_gains(mean_by_depth)),
        "oracle": oracle, "free_exit": free,
        "effort_law_predicted": D.effort_law(oracle, R),
        "p9_supported": None,
        "cross_token_effect_mean": statistics.fmean(o["cross_token_effect"] for o in oracle),
        "oracle_vs_free_exit_gap": gap_free,
        "oracle_vs_fixed_depth_gap": statistics.fmean(gaps_fixed) if gaps_fixed else None,
        "rl_gate": "OPEN" if gap_free is not None and gap_free >= SESOI else "CLOSED",
    }
    law = out["effort_law_predicted"]
    out["p9_supported"] = bool(law["r2"] is not None and law["r2"] >= 0.9 and law["slope"] > 0)
    (ROOT / "results" / (sys.argv[2] if len(sys.argv) > 2 else "stage2d.json")).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("fixed_depth", "geometric_decay", "effort_law_predicted", "p9_supported",
                                          "cross_token_effect_mean", "oracle_vs_free_exit_gap",
                                          "oracle_vs_fixed_depth_gap", "rl_gate")}, indent=2))


if __name__ == "__main__":
    main()
