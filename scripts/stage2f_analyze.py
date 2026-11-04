"""PREREG_stage2f.md section 4: COUNT-M with separate Muon and AdamW rates against COUNT-M with one shared rate
(Stage 2A's runs). Writes results/stage2f.json.

    .venv/bin/python scripts/stage2f_analyze.py
status NEEDS_SEEDS asks the driver for seeds 3 and 4 of both arms (an INCONCLUSIVE contrast whose mean gap is under
2 x SESOI, rule in clm.rules2e.needs_seeds).
"""
import hashlib
import json
import statistics
import sys

import stage2a_analyze as A
from clm import rules2e
from clm.analyze import SESOI, verdict

ROOT, RUNS = A.ROOT, A.RUNS
PREREG_F = ROOT / "PREREG_stage2f.md"
CHOICE = RUNS / "pilot_2f" / "choice.json"


def main():
    choice = json.loads(CHOICE.read_text())
    if not choice["confirm"]:
        sys.exit("the pilot did not ask for a confirmation run; there is nothing to analyse")
    sha = {"count": hashlib.sha256(A.PREREG.read_bytes()).hexdigest(), "f": hashlib.sha256(PREREG_F.read_bytes()).hexdigest()}
    have = lambda name, s: (RUNS / f"{name}-M-{s}" / "results.json").exists()
    seeds = [s for s in range(5) if have("count", s) and have("count_decoupled", s)]
    if len(seeds) < 3:
        sys.exit(f"need three seeds of count and count_decoupled, found {seeds}")
    runs = {"count": {s: A.load(RUNS / f"count-M-{s}" / "results.json", "count", sha["count"]) for s in seeds},
            "decoupled": {s: A.load(RUNS / f"count_decoupled-M-{s}" / "results.json", "count", sha["f"]) for s in seeds}}
    want = {"lr": choice["muon"], "adam_mult": choice["adam_mult"]}
    for s, r in runs["decoupled"].items():
        if {k: r["train"][k] for k in want} != want:
            sys.exit(f"seed {s} decoupled run trained at {r['train']['lr']}, x{r['train']['adam_mult']}, not {want}")
    if {r["train"].get("adam_mult", 1.0) for r in runs["count"].values()} != {1.0}:       # Stage 2A runs predate the key
        sys.exit("a COUNT run has a separate Adam rate")
    loss = {a: {s: r["final_val_loss"] for s, r in rs.items()} for a, rs in runs.items()}
    v = verdict(loss["decoupled"], loss["count"])
    status = "NEEDS_SEEDS" if rules2e.needs_seeds(loss, "decoupled", "count") else "DONE"
    sec = lambda rs: statistics.fmean(r["train_seconds"] / r["steps"] for r in rs.values())
    out = {"verdict": v, "p12": v == "X_BETTER", "status": status, "seeds": seeds, "sesoi": SESOI,
           "muon_lr": choice["muon"], "adam_lr": choice["adam"], "loss": loss,
           "mean_d": rules2e.mean_d(loss, "decoupled", "count"),
           "d_by_seed": {s: loss["decoupled"][s] - loss["count"][s] for s in seeds},
           "b0_control": choice.get("b0_control"), "pilot": choice["pilot"],
           "cost": {"s_per_step": {a: sec(rs) for a, rs in runs.items()},
                    "peak_vram_mib": {a: max(r["peak_vram_mib"] for r in rs.values()) for a, rs in runs.items()}}}
    (ROOT / "results" / "stage2f.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("verdict", "p12", "status", "mean_d", "d_by_seed")}, indent=2))
    return out


if __name__ == "__main__":
    main()
