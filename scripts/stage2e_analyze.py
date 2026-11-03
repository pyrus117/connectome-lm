"""PREREG_stage2e.md section 5: the gated COUNT read against COUNT (P10), Kenyon buckets under the gate against the
gated read (P11), and the gated random-bucket control when it ran. Rules in clm.rules2e. Writes results/stage2e.json.

    .venv/bin/python scripts/stage2e_analyze.py
status NEEDS_RANDOM asks the driver for the gated random-bucket arm (decided on three seeds only);
NEEDS_SEEDS asks for seeds 3 and 4 of every arm run so far.
"""
import hashlib
import json
import statistics
import sys

import stage2a_analyze as A
from clm import rules2e as R
from clm.analyze import SESOI, verdict
from clm.countread import BUCKET_LABELS
from stage2b_analyze import per_class

ROOT, RUNS = A.ROOT, A.RUNS
PREREG_E = ROOT / "PREREG_stage2e.md"
NAMES = {"count": "count", "gated": "count_gated", "kenyon": "count_kenyon_gated", "random": "count_random_gated"}


def main():
    sha = {"count": hashlib.sha256(A.PREREG.read_bytes()).hexdigest(),
           "e": hashlib.sha256(PREREG_E.read_bytes()).hexdigest()}
    have = lambda arm, s: (RUNS / f"{NAMES[arm]}-M-{s}" / "results.json").exists()
    seeds = [s for s in range(5) if all(have(a, s) for a in ("count", "gated", "kenyon"))]
    if len(seeds) < 3:
        sys.exit(f"need three seeds of count, gated and kenyon, found {seeds}")
    random_seeds = [s for s in seeds if have("random", s)]
    if random_seeds and random_seeds != seeds:
        sys.exit(f"the random-bucket arm has seeds {random_seeds}, the others {seeds}")
    arms = ["count", "gated", "kenyon"] + (["random"] if random_seeds else [])
    runs = {a: {s: A.load(RUNS / f"{NAMES[a]}-M-{s}" / "results.json", NAMES[a], sha["count" if a == "count" else "e"])
                for s in seeds} for a in arms}
    lrs = {r["train"]["lr"] for rs in runs.values() for r in rs.values()}
    if len(lrs) != 1:
        sys.exit(f"arms trained at different learning rates: {lrs}")
    loss = {a: {s: r["final_val_loss"] for s, r in rs.items()} for a, rs in runs.items()}
    classes = {a: per_class(NAMES[a], seeds) for a in arms}
    b0 = per_class("b0", [s for s in seeds if (RUNS / f"b0-M-{s}" / "bucket.json").exists()])

    pairs = {"gated_vs_count": ("gated", "count"), "kenyon_vs_gated": ("kenyon", "gated"),
             "kenyon_vs_count": ("kenyon", "count")}
    if "random" in arms:
        pairs |= {"random_vs_gated": ("random", "gated"), "kenyon_vs_random": ("kenyon", "random")}
    verdicts = {k: verdict(loss[x], loss[y]) for k, (x, y) in pairs.items()}
    want_random = len(seeds) == 3 and "random" not in arms and R.run_random(loss, classes)
    want_seeds = R.needs_seeds(loss, "gated", "count") or R.needs_seeds(loss, "kenyon", "gated")
    status = "NEEDS_RANDOM" if want_random else "NEEDS_SEEDS" if want_seeds else "DONE"

    mean = lambda c, lab: statistics.fmean(c[lab]) if c[lab] else None

    def relative(table):
        """Gates over the mean of the gates the read used; a gate still exactly 1.0 (its start) got no gradient."""
        used = [v for row in table for v in row if v != 1.0]
        m = statistics.fmean(used) if used else 1.0
        return [[v / m if v != 1.0 else None for v in row] for row in table]

    def table_mean(tables):
        return [[None if any(t[o][i] is None for t in tables) else statistics.fmean(t[o][i] for t in tables)
                 for i in range(len(tables[0][0]))] for o in range(len(tables[0]))]
    gates = {a: {"count_gate_mean": table_mean([r["count_gate"] for r in runs[a].values()]),
                 "count_gate_relative_mean": table_mean([relative(r["count_gate"]) for r in runs[a].values()]),
                 "count_absent_norm_mean": [statistics.fmean(r["count_absent_norm"][o] for r in runs[a].values())
                                            for o in range(len(next(iter(runs[a].values()))["count_absent_norm"]))]}
             for a in arms if a != "count"}
    sec = lambda rs: statistics.fmean(r["train_seconds"] / r["steps"] for r in rs.values())
    out = {
        "verdict": verdicts["gated_vs_count"], "verdicts": verdicts, "status": status, "seeds": seeds,
        "sesoi": SESOI, "loss": loss, "mean_d": {k: R.mean_d(loss, x, y) for k, (x, y) in pairs.items()},
        "p10": verdicts["gated_vs_count"] == "X_BETTER", "p10a": R.p10a(classes), "p10b": R.p10b(classes),
        "p11": R.p11(classes), "p11_similarity": R.p11_similarity(classes) if "random" in arms else None,
        "random_arm": "run" if "random" in arms else ("requested" if want_random else "not needed"),
        "class_gaps": {lab: {k: R.gap(classes, x, y, lab) for k, (x, y) in pairs.items()} for lab in BUCKET_LABELS},
        "class_losses_mean": {a: {lab: mean(c, lab) for lab in BUCKET_LABELS} for a, c in (classes | {"b0": b0}).items()},
        "gates": gates,
        "cost": {"s_per_step": {a: sec(rs) for a, rs in runs.items()},
                 "lookup_seconds_mean": {a: statistics.fmean(r["lookup_seconds_mean"] for r in rs.values())
                                         for a, rs in runs.items()},
                 "peak_vram_mib": {a: max(r["peak_vram_mib"] for r in rs.values()) for a, rs in runs.items()},
                 "host_rss_peak_mib": {a: max(r["host_rss_peak_mib"] for r in rs.values()) for a, rs in runs.items()}},
    }
    (ROOT / "results" / "stage2e.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("verdicts", "status", "mean_d", "p10", "p10a", "p10b", "p11",
                                          "p11_similarity", "random_arm")}, indent=2))
    return out


if __name__ == "__main__":
    main()
