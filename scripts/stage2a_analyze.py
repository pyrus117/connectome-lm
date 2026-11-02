"""PREREG_stage2a.md section 5: the paired verdict of COUNT-M against the frozen
Stage 2.0 B0-M runs, with the run costs, the leakage diagnostic and the
per-bucket losses. Writes results/stage2a.json.

    .venv/bin/python scripts/stage2a_analyze.py
Prints and stores `status`: DONE, or NEEDS_SEEDS when the pre-registered rule
asks for seeds 3 and 4 (an INCONCLUSIVE result whose mean gap is under 0.02 nats).
"""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

from clm.analyze import SESOI, verdict
from clm.countread import BUCKET_LABELS

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs2"
PREREG = ROOT / "PREREG_stage2a.md"
TOKENS = 12_207 * 32 * 512            # what a 200M-token run records: whole steps


def load(path, arm, prereg_sha=None):
    if not path.exists():
        sys.exit(f"missing {path}")
    r = json.loads(path.read_text())
    problems = []
    if r.get("status") != "complete" or not math.isfinite(r["final_val_loss"]):
        problems.append(f"status {r.get('status')}, loss {r['final_val_loss']}")
    if r["arm"] != arm:
        problems.append(f"arm {r['arm']!r}, expected {arm!r}")
    if r["tokens_seen"] != TOKENS:
        problems.append(f"tokens_seen {r['tokens_seen']}")
    if r.get("git_dirty"):
        problems.append("dirty tree")
    if prereg_sha and r["prereg_sha256"] != prereg_sha:
        problems.append("pre-registration hash differs from PREREG_stage2a.md")
    if problems:
        sys.exit(f"{path}: {'; '.join(problems)}")
    return r


def train_tail(run, n=100):
    log = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]
    return statistics.fmean(e["loss"] for e in log[-n:])


def bucket_means(names):
    """Mean loss per bucket over runs that have a bucket.json, for 3- and 4-token histories."""
    out = {}
    for k in (3, 4):
        key = f"history_{k}_tokens"
        out[key] = {}
        for lab in BUCKET_LABELS:
            vals = [json.loads((RUNS / n / "bucket.json").read_text())["by_history_bucket"][key][lab]["loss"]
                    for n in names if (RUNS / n / "bucket.json").exists()]
            out[key][lab] = statistics.fmean(v for v in vals if v is not None) if vals else None
    return out


def main():
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    seeds = [s for s in range(5) if (RUNS / f"count-M-{s}" / "results.json").exists()]
    if len(seeds) < 3:
        sys.exit(f"need three COUNT-M runs, found seeds {seeds}")
    count = {s: load(RUNS / f"count-M-{s}" / "results.json", "count", sha) for s in seeds}
    b0 = {s: load(RUNS / f"b0-M-{s}" / "results.json", "b0") for s in seeds}
    c_loss, b_loss = ({s: r["final_val_loss"] for s, r in rs.items()} for rs in (count, b0))
    d = [c_loss[s] - b_loss[s] for s in seeds]
    mean = statistics.fmean(d)
    v = verdict(c_loss, b_loss)
    needs_seeds = v == "INCONCLUSIVE" and abs(mean) < 2 * SESOI and len(seeds) == 3
    sec = lambda rs: statistics.fmean(r["train_seconds"] / r["steps"] for r in rs.values())
    pilot = json.loads((RUNS / "pilot" / "muon-0.008" / "results.json").read_text())["final_val_loss"]
    exponent = math.log(pilot / b0[0]["final_val_loss"]) / math.log(4)       # PREREG_stage2a.md section 5
    giveback = statistics.fmean(b_loss.values()) * ((sec(count) / sec(b0)) ** exponent - 1)
    noloo = load(RUNS / "count-noloo-M-0" / "results.json", "count_noloo", sha) \
        if (RUNS / "count-noloo-M-0" / "results.json").exists() else None
    ngram = json.loads((ROOT / "results" / "stage2a_ngram.json").read_text())
    out = {
        "verdict": v, "status": "NEEDS_SEEDS" if needs_seeds else "DONE",
        "gate1": "PENDING" if needs_seeds else "PASS" if v == "X_BETTER" else "FAIL",
        "equal_wallclock": {"token_exponent": exponent, "time_giveback_nats": giveback,
                            "gap_nats": mean + giveback,
                            "flag_for_nate": v == "X_BETTER" and mean + giveback > -SESOI},
        "seeds": seeds, "sesoi": SESOI, "count": c_loss, "b0": b_loss,
        "d": dict(zip(seeds, d)), "mean_d": mean, "sd_d": statistics.stdev(d),
        "sd_count": statistics.stdev(c_loss.values()), "sd_b0": statistics.stdev(b_loss.values()),
        "count_lr": next(iter(count.values()))["train"]["lr"],
        "cost": {"s_per_step_count": sec(count), "s_per_step_b0": sec(b0),
                 "time_ratio": sec(count) / sec(b0),
                 "peak_vram_mib_count": max(r["peak_vram_mib"] for r in count.values()),
                 "peak_vram_mib_b0": max(r["peak_vram_mib"] for r in b0.values()),
                 "host_rss_peak_mib_count": max(r["host_rss_peak_mib"] for r in count.values()),
                 "lookup_seconds_mean": statistics.fmean(r["lookup_seconds_mean"] for r in count.values())},
        "train_vs_val": {f"count-M-{s}": {"train_last100": train_tail(RUNS / f"count-M-{s}"),
                                          "val": c_loss[s]} for s in seeds},
        "leakage_diagnostic": None if noloo is None else {
            "val_loss_noloo_seed0": noloo["final_val_loss"], "val_loss_loo_seed0": c_loss[0],
            "train_last100_noloo": train_tail(RUNS / "count-noloo-M-0"),
            "train_last100_loo": train_tail(RUNS / "count-M-0")},
        "pure_ngram_val_loss": ngram["val_loss"],
        "buckets": {"count": bucket_means([f"count-M-{s}" for s in seeds]),
                    "b0": bucket_means([f"b0-M-{s}" for s in seeds])},
    }
    (ROOT / "results" / "stage2a.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("verdict", "status", "gate1", "d", "mean_d", "cost", "equal_wallclock")}, indent=2))
    return out


if __name__ == "__main__":
    main()
