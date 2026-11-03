"""PREREG_stage2b.md section 5: COUNT + Kenyon backoff against COUNT and against the random-bucket
control, paired by seed, with the per-history-count-bucket pattern (P4). Writes results/stage2b.json.

    .venv/bin/python scripts/stage2b_analyze.py
status NEEDS_SEEDS asks the driver for seeds 3 and 4 (an INCONCLUSIVE Kenyon-against-COUNT result whose mean
gap is under 0.02 nats), as in Stage 2A.
"""
import hashlib
import json
import statistics
import sys
from pathlib import Path

import stage2a_analyze as A
from clm.analyze import SESOI, verdict
from clm.countread import BUCKET_LABELS

ROOT = A.ROOT
RUNS = A.RUNS
PREREG_B = ROOT / "PREREG_stage2b.md"
HISTORY = "history_4_tokens"
RARE, COMMON = ("not stored", "1-9"), ("100-999", ">=1000")
P4_GAIN, P4_FLAT = 0.01, 0.01


def per_class(arm, seeds):
    """{class: [loss per seed]} for 4-token histories, from each run's bucket.json."""
    out = {lab: [] for lab in BUCKET_LABELS}
    for s in seeds:
        b = json.loads((RUNS / f"{arm}-M-{s}" / "bucket.json").read_text())["by_history_bucket"][HISTORY]
        for lab in BUCKET_LABELS:
            out[lab].append(b[lab]["loss"])
    return out


def main():
    sha_a = hashlib.sha256(A.PREREG.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(PREREG_B.read_bytes()).hexdigest()
    seeds = [s for s in range(5) if all((RUNS / f"{a}-M-{s}" / "results.json").exists()
                                        for a in ("count", "count_kenyon", "count_random"))]
    if len(seeds) < 3:
        sys.exit(f"need three seeds of every arm, found {seeds}")
    runs = {"count": {s: A.load(RUNS / f"count-M-{s}" / "results.json", "count", sha_a) for s in seeds},
            "kenyon": {s: A.load(RUNS / f"count_kenyon-M-{s}" / "results.json", "count_kenyon", sha_b) for s in seeds},
            "random": {s: A.load(RUNS / f"count_random-M-{s}" / "results.json", "count_random", sha_b) for s in seeds}}
    lrs = {arm: {r["train"]["lr"] for r in rs.values()} for arm, rs in runs.items()}
    if len({lr for v in lrs.values() for lr in v}) != 1:
        sys.exit(f"arms trained at different learning rates: {lrs}")
    loss = {arm: {s: r["final_val_loss"] for s, r in rs.items()} for arm, rs in runs.items()}
    mean_d = lambda a, b: statistics.fmean(loss[a][s] - loss[b][s] for s in seeds)
    verdicts = {"kenyon_vs_count": verdict(loss["kenyon"], loss["count"]),
                "random_vs_count": verdict(loss["random"], loss["count"]),
                "kenyon_vs_random": verdict(loss["kenyon"], loss["random"])}
    needs = (verdicts["kenyon_vs_count"] == "INCONCLUSIVE" and abs(mean_d("kenyon", "count")) < 2 * SESOI
             and len(seeds) == 3)

    classes = {arm: per_class({"count": "count", "kenyon": "count_kenyon", "random": "count_random"}[arm], seeds)
               for arm in runs}
    mean = lambda arm, lab: statistics.fmean(classes[arm][lab])
    gap = {lab: {"kenyon_minus_count": mean("kenyon", lab) - mean("count", lab),
                 "random_minus_count": mean("random", lab) - mean("count", lab)} for lab in BUCKET_LABELS}
    p4 = (all(gap[l]["kenyon_minus_count"] <= -P4_GAIN and
              gap[l]["kenyon_minus_count"] - gap[l]["random_minus_count"] <= -P4_GAIN for l in RARE)
          and all(abs(gap[l]["kenyon_minus_count"]) < P4_FLAT for l in COMMON))
    sec = lambda rs: statistics.fmean(r["train_seconds"] / r["steps"] for r in rs.values())
    out = {
        "verdict": verdicts["kenyon_vs_count"], "verdicts": verdicts,
        "status": "NEEDS_SEEDS" if needs else "DONE", "seeds": seeds, "sesoi": SESOI,
        "loss": loss, "mean_d": {"kenyon_vs_count": mean_d("kenyon", "count"),
                                 "random_vs_count": mean_d("random", "count"),
                                 "kenyon_vs_random": mean_d("kenyon", "random")},
        "p4": {"supported": bool(p4), "rare_classes": list(RARE), "common_classes": list(COMMON),
               "gain_threshold": P4_GAIN, "flat_threshold": P4_FLAT, "class_gaps": gap},
        "class_losses_mean": {arm: {lab: mean(arm, lab) for lab in BUCKET_LABELS} for arm in runs},
        "cost": {"s_per_step": {arm: sec(rs) for arm, rs in runs.items()},
                 "time_ratio_kenyon_vs_count": sec(runs["kenyon"]) / sec(runs["count"]),
                 "lookup_seconds_mean": {arm: statistics.fmean(r["lookup_seconds_mean"] for r in rs.values())
                                         for arm, rs in runs.items()},
                 "peak_vram_mib": {arm: max(r["peak_vram_mib"] for r in rs.values()) for arm, rs in runs.items()},
                 "host_rss_peak_mib": {arm: max(r["host_rss_peak_mib"] for r in rs.values())
                                       for arm, rs in runs.items()}},
    }
    (ROOT / "results" / "stage2b.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("verdicts", "status", "mean_d", "p4", "cost")}, indent=2))
    return out


if __name__ == "__main__":
    main()
