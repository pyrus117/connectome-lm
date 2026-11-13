"""PREREG_stage2m.md section 3: did the extra passes of the looped model earn anything once the loop gain trains at 8e-3
(P22), and with every parameter at one shared rate (P23)? Both arms carry the plain COUNT read. The decision is the
loop's benefit inside each run (depth-1 loss minus best-depth loss) on every seed; the paired contrasts against Stage 2L
and Stage 2H are described only. Rules in clm.rules2m. Writes results/stage2m.json.

    .venv/bin/python scripts/stage2m_analyze.py
"""
import hashlib
import json
import statistics
import sys

import torch

import stage2a_analyze as A
from clm import rules2h as H
from clm import rules2m as R
from clm.analyze import SESOI

ROOT, RUNS = A.ROOT, A.RUNS
GAINLOW, SHARED = "loop_count_gainlow", "loop_count_shared"
SHARED_SETTINGS = {"optimizer": "muon", "lr": 0.008, "arch": "loop", "loop_gain": True, "loop_loss": "all", "micro": 4,
                   "memory": "count"}
WANT = {GAINLOW: {**SHARED_SETTINGS, "adam_mult": 8.0, "gain_mult": 1.0},
        SHARED: {**SHARED_SETTINGS, "adam_mult": 1.0, "gain_mult": None}}


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def gain_rows(run):
    """Mean |gain| of each pass in a finished run's final checkpoint, or None if the checkpoint is not kept."""
    ckpt = RUNS / run / "ckpt.pt"
    if not ckpt.exists():
        return None
    g = torch.load(ckpt, map_location="cpu")["model"]["loop_gain"].float()
    return g.abs().mean(dim=1).tolist()


def main():
    seeds = [s for s in range(3) if all((RUNS / f"{n}-M-{s}" / "results.json").exists() for n in WANT)]
    if 0 not in seeds:
        sys.exit("no seed 0 of both arms")
    runs = {n: {s: A.load(RUNS / f"{n}-M-{s}" / "results.json", "loop_gain_all_count", sha("PREREG_stage2m.md")) for s in seeds}
            for n in WANT}
    for n, rs in runs.items():
        for s, r in rs.items():
            got = {k: r["train"].get(k) for k in WANT[n]}
            if got != WANT[n]:
                sys.exit(f"{n}-M-{s} trained with {got}, expected {WANT[n]}")
            if len(r["val_loss_by_depth"]) != 6:
                sys.exit(f"{n}-M-{s} has {len(r['val_loss_by_depth'])} depth readouts")
    summ = {n: {s: H.summary(r["val_loss_by_depth"]) for s, r in rs.items()} for n, rs in runs.items()}
    benefit = {n: {s: R.benefit(v) for s, v in d.items()} for n, d in summ.items()}
    futility = len(seeds) == 1
    out = {"seeds": seeds, "futility_stop": futility, "sesoi": SESOI, "summary": summ, "benefit": benefit}
    if futility:
        out["p22_status"] = out["p23_status"] = "NOT_SUPPORTED_FUTILITY"
    else:
        out["p22_status"] = "SUPPORTED" if R.alive(summ[GAINLOW]) else "NOT_SUPPORTED"
        out["p23_status"] = "SUPPORTED" if R.alive(summ[SHARED]) else "NOT_SUPPORTED"
    refs = {"stage2l_loop_count": ("loop_gain_all_count_sep", "loop_gain_all_count", "PREREG_stage2l.md"),
            "stage2h_loop": ("loop_gain_all", "loop_gain_all", "PREREG_stage2h.md"),
            "count_separate": ("count_decoupled_q", "count", "PREREG_stage2k.md"),
            "count_shared": ("count", "count", None)}
    ref = {k: {s: A.load(RUNS / f"{d}-M-{s}" / "results.json", arm, sha(p) if p else None) for s in seeds}
           for k, (d, arm, p) in refs.items()}
    ref_loop = {k: {s: H.summary(r["val_loss_by_depth"]) for s, r in ref[k].items()} for k in ("stage2l_loop_count", "stage2h_loop")}
    ref_plain = {k: {s: H.summary([r["final_val_loss"]]) for s, r in ref[k].items()} for k in ("count_separate", "count_shared")}
    out["reference_benefit"] = {k: {s: R.benefit(v) for s, v in d.items()} for k, d in ref_loop.items()}
    if not futility:
        pairs = {"gainlow_vs_stage2l_loop_count": (GAINLOW, ref_loop["stage2l_loop_count"]),
                 "shared_vs_stage2h_loop": (SHARED, ref_loop["stage2h_loop"]),
                 "gainlow_vs_count_separate": (GAINLOW, ref_plain["count_separate"]),
                 "shared_vs_count_shared": (SHARED, ref_plain["count_shared"]),
                 "gainlow_vs_stage2h_loop": (GAINLOW, ref_loop["stage2h_loop"])}
        out["described_contrasts"] = {k: {"verdict": H.contrast(summ[n], other, "best_loss"),
                                          "mean_d": statistics.fmean(summ[n][s]["best_loss"] - other[s]["best_loss"] for s in seeds)}
                                      for k, (n, other) in pairs.items()}
        out["described_contrasts"]["shared_vs_gainlow"] = {
            "verdict": H.contrast(summ[SHARED], summ[GAINLOW], "best_loss"),
            "mean_d": statistics.fmean(summ[SHARED][s]["best_loss"] - summ[GAINLOW][s]["best_loss"] for s in seeds)}
    out["mean_gain_by_depth"] = {n: [statistics.fmean(summ[n][s]["gains"][i] for s in seeds) for i in range(5)] for n in WANT}
    out["gain_rows_mean_abs"] = {n: {s: gain_rows(f"{n}-M-{s}") for s in seeds} for n in WANT}
    out["gain_rows_mean_abs"]["stage2l_loop_count"] = {s: gain_rows(f"loop_gain_all_count_sep-M-{s}") for s in seeds}
    out["cost"] = {n: {"s_per_step": statistics.fmean(r["train_seconds"] / r["steps"] for r in rs.values()),
                       "peak_vram_mib": max(r["peak_vram_mib"] for r in rs.values()),
                       "host_rss_peak_mib": max(r["host_rss_peak_mib"] for r in rs.values())} for n, rs in runs.items()}
    (ROOT / "results" / "stage2m.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in out if k in ("p22_status", "p23_status", "benefit", "described_contrasts", "futility_stop")}, indent=2))
    return out


if __name__ == "__main__":
    main()
