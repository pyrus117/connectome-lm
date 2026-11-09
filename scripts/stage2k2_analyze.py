"""PREREG_stage2k2.md section 3: three candidate COUNT reads with their gate parameters on a slower rate, each against
plain COUNT on five seeds (P20), and the read later stages carry forward. Rules in clm.rules2k2. Writes results/stage2k2.json.

    .venv/bin/python scripts/stage2k2_analyze.py
"""
import hashlib
import json
import statistics
import sys

import stage2a_analyze as A
from clm import rules2k2 as R
from clm.analyze import SESOI, verdict
from clm.countread import BUCKET_LABELS
from stage2b_analyze import per_class

ROOT, RUNS = A.ROOT, A.RUNS
SEEDS = range(5)
NAMES = {"count": "count_decoupled_q", "gated_low": "count_gated_low", "scaled_low": "count_scaled_low",
         "oscaled_low": "count_oscaled_low"}
ARM = {"count": "count", "gated_low": "count_gated", "scaled_low": "count_scaled", "oscaled_low": "count_oscaled"}
HIGH = {"gated_high": ("count_gated_sep", "count_gated"), "scaled_high": ("count_scaled_sep", "count_scaled")}
RATES = {"optimizer": "muon", "lr": 0.008, "adam_mult": 8.0}


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def main():
    sha_k, sha_k2 = sha("PREREG_stage2k.md"), sha("PREREG_stage2k2.md")
    runs = {a: {s: A.load(RUNS / f"{NAMES[a]}-M-{s}" / "results.json", ARM[a],
                          sha_k if a == "count" and s < 3 else sha_k2) for s in SEEDS} for a in NAMES}
    for a, rs in runs.items():
        want = {**RATES, "gate_mult": None if a == "count" else 1.0}
        for s, r in rs.items():
            got = {k: r["train"].get(k) for k in want}
            if got != want:
                sys.exit(f"{NAMES[a]}-M-{s} trained with {got}, expected {want}")
    high = {a: {s: A.load(RUNS / f"{n}-M-{s}" / "results.json", arm, sha_k)["final_val_loss"] for s in range(3)}
            for a, (n, arm) in HIGH.items()}
    loss = {a: {s: r["final_val_loss"] for s, r in rs.items()} for a, rs in runs.items()}
    classes = {a: per_class(NAMES[a], list(SEEDS)) for a in NAMES}
    verdicts = R.versus_count(loss)
    cands = R.CANDIDATES
    pairs = {f"{x}_vs_{y}": (x, y) for i, x in enumerate(cands) for y in cands[i + 1:]}
    d = lambda x, y: {s: loss[x][s] - loss[y][s] for s in SEEDS}
    sec = lambda rs: statistics.fmean(r["train_seconds"] / r["steps"] for r in rs.values())
    out = {
        "sesoi": SESOI, "seeds": list(SEEDS), "loss": loss, "p20": R.p20(loss), "carry_forward": R.carry_forward(loss),
        "verdicts_vs_count": verdicts,
        "verdicts_among_candidates": {k: verdict(loss[x], loss[y]) for k, (x, y) in pairs.items()},
        "mean_d_vs_count": {a: statistics.fmean(d(a, "count").values()) for a in cands},
        "d_by_seed_vs_count": {a: d(a, "count") for a in cands},
        "count_seed_sd": statistics.stdev(loss["count"].values()),
        "class_gaps_vs_count": {lab: {a: statistics.fmean(classes[a][lab]) - statistics.fmean(classes["count"][lab])
                                      for a in cands} for lab in BUCKET_LABELS},
        "class_losses_mean": {a: {lab: statistics.fmean(c[lab]) for lab in BUCKET_LABELS} for a, c in classes.items()},
        "gate_table_mean": [[statistics.fmean(runs["gated_low"][s]["count_gate"][o][i] for s in SEEDS) for i in range(8)]
                            for o in range(4)],
        "absent_norm_mean": [statistics.fmean(runs["gated_low"][s]["count_absent_norm"][o] for s in SEEDS) for o in range(4)],
        "scale_by_seed": {a: {s: runs[a][s]["count_scale"] for s in SEEDS} for a in ("scaled_low", "oscaled_low")},
        "high_rate_stage2k_loss": high,
        "low_minus_high_rate_by_seed": {k: {s: loss[low][s] - high[hk][s] for s in range(3)}
                                        for k, low, hk in (("gated", "gated_low", "gated_high"), ("scaled", "scaled_low", "scaled_high"))},
        "cost": {"s_per_step": {a: sec(rs) for a, rs in runs.items()},
                 "peak_vram_mib": {a: max(r["peak_vram_mib"] for r in rs.values()) for a, rs in runs.items()},
                 "host_rss_peak_mib": {a: max(r["host_rss_peak_mib"] for r in rs.values()) for a, rs in runs.items()}},
    }
    (ROOT / "results" / "stage2k2.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("verdicts_vs_count", "p20", "carry_forward", "mean_d_vs_count", "count_seed_sd")}, indent=2))
    return out


if __name__ == "__main__":
    main()
