"""PREREG_stage2k.md section 5: the gated COUNT read (P19) and the scale-only control against plain COUNT, all on
separate Muon and AdamW rates, and what the three verdicts say about structure versus scale. Rules in clm.rules2k.
Writes results/stage2k.json.

    .venv/bin/python scripts/stage2k_analyze.py
status NEEDS_SEEDS asks the driver for seeds 3 and 4 of every arm (decided on three seeds only).
"""
import hashlib
import json
import statistics
import sys

import stage2a_analyze as A
from clm import rules2k as R
from clm.analyze import SESOI, verdict
from clm.countread import BUCKET_LABELS
from stage2b_analyze import per_class

ROOT, RUNS = A.ROOT, A.RUNS
NAMES = {"count": "count_decoupled_q", "gated": "count_gated_sep", "scaled": "count_scaled_sep"}
ARM = {"count": "count", "gated": "count_gated", "scaled": "count_scaled"}
PAIRS = {"gated_vs_count": ("gated", "count"), "gated_vs_scaled": ("gated", "scaled"), "scaled_vs_count": ("scaled", "count")}
RATES = {"optimizer": "muon", "lr": 0.008, "adam_mult": 8.0}


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def main():
    have = lambda arm, s: (RUNS / f"{NAMES[arm]}-M-{s}" / "results.json").exists()
    seeds = [s for s in range(5) if all(have(a, s) for a in NAMES)]
    if len(seeds) < 3:
        sys.exit(f"need three seeds of count, gated and scaled, found {seeds}")
    runs = {a: {s: A.load(RUNS / f"{NAMES[a]}-M-{s}" / "results.json", ARM[a], sha("PREREG_stage2k.md")) for s in seeds}
            for a in NAMES}
    laptop = {s: json.loads((RUNS / f"count_decoupled-M-{s}" / "results.json").read_text())["final_val_loss"]
              for s in (0, 1, 2) if (RUNS / f"count_decoupled-M-{s}" / "results.json").exists()}
    for a, rs in runs.items():
        for s, r in rs.items():
            got = {k: r["train"][k] for k in RATES}
            if got != RATES:
                sys.exit(f"{NAMES[a]}-M-{s} trained with {got}, expected {RATES}")
    loss = {a: {s: r["final_val_loss"] for s, r in rs.items()} for a, rs in runs.items()}
    classes = {a: per_class(NAMES[a], seeds) for a in NAMES}
    b0 = per_class("b0", [s for s in seeds if (RUNS / f"b0-M-{s}" / "bucket.json").exists()])
    verdicts = {k: verdict(loss[x], loss[y]) for k, (x, y) in PAIRS.items()}
    status = "NEEDS_SEEDS" if R.wants_seeds(loss) else "DONE"

    def table_mean(tables):
        return [[statistics.fmean(t[o][i] for t in tables) for i in range(len(tables[0][0]))] for o in range(len(tables[0]))]

    sec = lambda rs: statistics.fmean(r["train_seconds"] / r["steps"] for r in rs.values())
    out = {
        "status": status, "seeds": seeds, "sesoi": SESOI, "verdicts": verdicts, "p19": R.p19(loss), "reading": R.reading(loss),
        "loss": loss, "count_on_laptop_stage2f": laptop,
        "machine_check_q_minus_laptop": {s: loss["count"][s] - laptop[s] for s in laptop},
        "mean_d":{k: R.mean_d(loss, x, y) for k, (x, y) in PAIRS.items()},
        "d_by_seed": {k: {s: loss[x][s] - loss[y][s] for s in seeds} for k, (x, y) in PAIRS.items()},
        "class_gaps": {lab: {k: R.gap(classes, x, y, lab) for k, (x, y) in PAIRS.items()} for lab in BUCKET_LABELS},
        "class_losses_mean": {a: {lab: statistics.fmean(c[lab]) for lab in BUCKET_LABELS if c[lab]}
                              for a, c in (classes | {"b0_shared_rate": b0}).items()},
        "gate_table_mean": table_mean([r["count_gate"] for r in runs["gated"].values()]),
        "absent_norm_mean": [statistics.fmean(r["count_absent_norm"][o] for r in runs["gated"].values()) for o in range(4)],
        "scale_by_seed": {s: r["count_scale"] for s, r in runs["scaled"].items()},
        "cost": {"s_per_step": {a: sec(rs) for a, rs in runs.items()},
                 "peak_vram_mib": {a: max(r["peak_vram_mib"] for r in rs.values()) for a, rs in runs.items()},
                 "host_rss_peak_mib": {a: max(r["host_rss_peak_mib"] for r in rs.values()) for a, rs in runs.items()}},
    }
    (ROOT / "results" / "stage2k.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("verdicts", "status", "mean_d", "p19", "reading", "scale_by_seed")}, indent=2))
    return out


if __name__ == "__main__":
    main()
