"""PREREG_stage2n.md section 3: which 64e-3 AdamW group kills the loop? Completes the 2x2 factorial with Stage 2M's
SHARED (head 8e-3, embeddings 8e-3) and GAINLOW (head 64e-3, embeddings 64e-3), holding loop_gain at 8e-3 in all four
cells:
- HEADLOW (P24): head.weight at 8e-3, embeddings (embed, value_embed, ve_gate) at 64e-3.
- EMBEDLOW (P25): embeddings at 8e-3, head.weight at 64e-3.
Rules in clm.rules2n. Writes results/stage2n.json.

    .venv/bin/python scripts/stage2n_analyze.py
"""
import hashlib
import json
import statistics
import sys

import torch

import stage2a_analyze as A
from clm import rules2h as H
from clm import rules2n as R
from clm.analyze import SESOI

ROOT, RUNS = A.ROOT, A.RUNS
HEADLOW, EMBEDLOW = "loop_count_headlow", "loop_count_embedlow"
SHARED_SETTINGS = {"optimizer": "muon", "lr": 0.008, "arch": "loop", "loop_gain": True, "loop_loss": "all", "micro": 4,
                   "memory": "count"}
WANT = {HEADLOW: {**SHARED_SETTINGS, "adam_mult": 8.0, "gain_mult": 1.0, "head_mult": 1.0},
        EMBEDLOW: {**SHARED_SETTINGS, "adam_mult": 1.0, "gain_mult": None, "head_mult": 8.0}}


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def ckpt_stats(run):
    """Mean |loop_gain| per pass and L2 norm of head.weight in a finished run's final checkpoint."""
    ckpt = RUNS / run / "ckpt.pt"
    if not ckpt.exists():
        return None, None
    m = torch.load(ckpt, map_location="cpu")["model"]
    g = m["loop_gain"].float().abs().mean(dim=1).tolist()
    h = float(m["head.weight"].float().norm().item())
    return g, h


def main():
    seeds = {n: [s for s in range(3) if (RUNS / f"{n}-M-{s}" / "results.json").exists()] for n in WANT}
    for n, ss in seeds.items():
        if ss not in ([0], [0, 1, 2]):
            sys.exit(f"{n} has seeds {ss}, expected [0] or [0, 1, 2]")
    runs = {n: {s: A.load(RUNS / f"{n}-M-{s}" / "results.json", "loop_gain_all_count", sha("PREREG_stage2n.md")) for s in seeds[n]}
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
    p24 = R.status(summ[HEADLOW])
    p25 = R.status(summ[EMBEDLOW])
    refs = {"stage2m_shared": ("loop_count_shared", "loop_gain_all_count", "PREREG_stage2m.md"),
            "stage2m_gainlow": ("loop_count_gainlow", "loop_gain_all_count", "PREREG_stage2m.md")}
    ref = {k: {s: A.load(RUNS / f"{d}-M-{s}" / "results.json", arm, sha(p)) for s in range(3)}
           for k, (d, arm, p) in refs.items()}
    ref_summ = {k: {s: H.summary(r["val_loss_by_depth"]) for s, r in ref[k].items()} for k in refs}
    contrasts = {}
    for label, arm in (("headlow", HEADLOW), ("embedlow", EMBEDLOW)):
        if seeds[arm] == [0, 1, 2]:
            for rlabel, rkey in (("shared", "stage2m_shared"), ("gainlow", "stage2m_gainlow")):
                other = ref_summ[rkey]
                contrasts[f"{label}_vs_{rlabel}"] = {
                    "verdict": H.contrast(summ[arm], other, "best_loss"),
                    "mean_d": statistics.fmean(summ[arm][s]["best_loss"] - other[s]["best_loss"] for s in (0, 1, 2)),
                }
    if seeds[HEADLOW] == [0, 1, 2] and seeds[EMBEDLOW] == [0, 1, 2]:
        contrasts["headlow_vs_embedlow"] = {
            "verdict": H.contrast(summ[HEADLOW], summ[EMBEDLOW], "best_loss"),
            "mean_d": statistics.fmean(summ[HEADLOW][s]["best_loss"] - summ[EMBEDLOW][s]["best_loss"] for s in (0, 1, 2)),
        }
    gain_rows, head_norms = {}, {}
    for n in WANT:
        gain_rows[n], head_norms[n] = {}, {}
        for s in seeds[n]:
            g, h = ckpt_stats(f"{n}-M-{s}")
            gain_rows[n][s], head_norms[n][s] = g, h
    out = {
        "seeds": seeds,
        "sesoi": SESOI,
        "p24_status": p24,
        "p25_status": p25,
        "summary": summ,
        "benefit": benefit,
        "reference_benefit": {k: {s: R.benefit(v) for s, v in d.items()} for k, d in ref_summ.items()},
        "described_contrasts": contrasts,
        "mean_gain_by_depth": {n: [statistics.fmean(summ[n][s]["gains"][i] for s in seeds[n]) for i in range(5)] for n in WANT},
        "gain_rows_mean_abs": gain_rows,
        "head_weight_norm": head_norms,
        "cost": {n: {"s_per_step": statistics.fmean(r["train_seconds"] / r["steps"] for r in rs.values()),
                     "peak_vram_mib": max(r["peak_vram_mib"] for r in rs.values()),
                     "host_rss_peak_mib": max(r["host_rss_peak_mib"] for r in rs.values())} for n, rs in runs.items()},
    }
    (ROOT / "results" / "stage2n.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("p24_status", "p25_status", "benefit", "described_contrasts")}, indent=2))
    return out


if __name__ == "__main__":
    main()
