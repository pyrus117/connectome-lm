"""PREREG_stage2g.md section 4: gate 1 again, COUNT-M on separate rates (Stage 2F's runs) against B0-M on its best
rates (B0-M on separate rates if the pilot asked for it, else the Stage 2.0 runs). Writes results/stage2g.json.

    .venv/bin/python scripts/stage2g_analyze.py
"""
import hashlib
import json
import statistics
import sys

import stage2a_analyze as A
from clm.analyze import SESOI, verdict

ROOT, RUNS = A.ROOT, A.RUNS
CHOICE = RUNS / "pilot_2g" / "choice.json"


def main():
    choice = json.loads(CHOICE.read_text())
    sha = {k: hashlib.sha256((ROOT / f"PREREG_stage{k}.md").read_bytes()).hexdigest() for k in ("2f", "2g")}
    seeds = (0, 1, 2)
    count = {s: A.load(RUNS / f"count_decoupled-M-{s}" / "results.json", "count", sha["2f"]) for s in seeds}
    shared = {s: A.load(RUNS / f"b0-M-{s}" / "results.json", "b0") for s in seeds}
    ref, sep = shared, None
    if choice["separate"]:
        sep = {s: A.load(RUNS / f"b0_separate-M-{s}" / "results.json", "b0", sha["2g"]) for s in seeds}
        want = {"lr": 8e-3, "adam_mult": choice["adam_mult"]}
        for s, r in sep.items():
            if {k: r["train"][k] for k in want} != want:
                sys.exit(f"seed {s} B0 run trained at {r['train']['lr']}, x{r['train']['adam_mult']}, not {want}")
        ref = sep
    loss = lambda rs: {s: r["final_val_loss"] for s, r in rs.items()}
    c_loss, r_loss, b_loss = loss(count), loss(ref), loss(shared)
    v = verdict(c_loss, r_loss)
    mean = lambda x, y: statistics.fmean(x[s] - y[s] for s in x)
    out = {"verdict": v, "gate1": v == "X_BETTER", "sesoi": SESOI, "b0_reference": "separate" if sep else "shared",
           "adam_lr": choice["adam"], "loss": {"count_separate": c_loss, "b0_reference": r_loss, "b0_shared": b_loss},
           "mean_d_count_vs_b0_reference": mean(c_loss, r_loss),
           "d_by_seed": {s: c_loss[s] - r_loss[s] for s in seeds},
           "b0_separate_vs_shared": ({"verdict": verdict(r_loss, b_loss), "mean_d": mean(r_loss, b_loss)} if sep else None),
           "pilot": choice["pilot"],
           "cost": {"s_per_step": {n: statistics.fmean(r["train_seconds"] / r["steps"] for r in rs.values())
                                   for n, rs in (("count_separate", count), ("b0_reference", ref))}}}
    (ROOT / "results" / "stage2g.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("verdict", "gate1", "b0_reference", "mean_d_count_vs_b0_reference",
                                          "b0_separate_vs_shared")}, indent=2))
    return out


if __name__ == "__main__":
    main()
