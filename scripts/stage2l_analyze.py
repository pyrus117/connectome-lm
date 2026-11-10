"""PREREG_stage2l.md section 3: does the COUNT read help the gained all-depth looped model (P21)? The looped model with
the read against without it on best-depth loss, how much each of the two additions is worth alone and together, and
the futility case. Rules in clm.rules2h (Stage 2H's, reused). Writes results/stage2l.json.

    .venv/bin/python scripts/stage2l_analyze.py
"""
import hashlib
import json
import statistics
import sys

import stage2a_analyze as A
from clm import rules2h as R
from clm.analyze import SESOI

ROOT, RUNS = A.ROOT, A.RUNS
LOOP, BOTH = "loop_gain_all_sep", "loop_gain_all_count_sep"
SUFFIX = {"count": "", "gated_low": "_gated", "scaled_low": "_scaled", "oscaled_low": "_oscaled"}
REF = {"count": "count_decoupled_q", "gated_low": "count_gated_low", "scaled_low": "count_scaled_low",
       "oscaled_low": "count_oscaled_low"}
REF_ARM = {"count": "count", "gated_low": "count_gated", "scaled_low": "count_scaled", "oscaled_low": "count_oscaled"}
SHARED = {"optimizer": "muon", "lr": 0.008, "adam_mult": 8.0, "arch": "loop", "loop_gain": True, "loop_loss": "all", "micro": 4}


def main():
    sha = hashlib.sha256((ROOT / "PREREG_stage2l.md").read_bytes()).hexdigest()
    carry = json.loads((RUNS / "stage2l_choice.json").read_text())["read"]
    seeds = [s for s in range(3) if (RUNS / f"{BOTH}-M-{s}" / "results.json").exists()]
    if 0 not in seeds:
        sys.exit("no seed 0 of the combined arm")
    arms = {LOOP: "loop_gain_all", BOTH: "loop_gain_all_count" + SUFFIX[carry]}
    runs = {n: {s: A.load(RUNS / f"{n}-M-{s}" / "results.json", arm, sha) for s in seeds} for n, arm in arms.items()}
    for n, rs in runs.items():
        want = {**SHARED, "memory": "none" if n == LOOP else "count"}
        for s, r in rs.items():
            got = {k: r["train"][k] for k in want}
            if got != want:
                sys.exit(f"{n}-M-{s} trained with {got}, expected {want}")
            if len(r["val_loss_by_depth"]) != 6:
                sys.exit(f"{n}-M-{s} has {len(r['val_loss_by_depth'])} depth readouts")
    summ = {n: {s: R.summary(r["val_loss_by_depth"]) for s, r in rs.items()} for n, rs in runs.items()}
    ref_runs = {s: A.load(RUNS / f"{REF[carry]}-M-{s}" / "results.json", REF_ARM[carry],
                          hashlib.sha256((ROOT / ("PREREG_stage2k.md" if carry == "count" and s < 3 else "PREREG_stage2k2.md")).read_bytes()).hexdigest())
                for s in range(3)}
    ref = {s: r["final_val_loss"] for s, r in ref_runs.items()}
    b0 = {s: json.loads((RUNS / f"b0_separate-M-{s}" / "results.json").read_text())["final_val_loss"]
          for s in range(3) if (RUNS / f"b0_separate-M-{s}" / "results.json").exists()}
    futility = len(seeds) == 1
    out = {"read": carry, "seeds": seeds, "futility_stop": futility, "sesoi": SESOI,
           "summary": summ, "reference_read_loss": ref, "b0_separate_laptop_loss": b0}
    if futility:
        out["p21_status"] = "NOT_SUPPORTED_FUTILITY"
        out["seed0_best_loss_difference"] = summ[BOTH][0]["best_loss"] - summ[LOOP][0]["best_loss"]
    else:
        out["contrasts_vs_loop"] = {k: R.contrast(summ[BOTH], summ[LOOP], k) for k in ("best_loss", "depth1", "deepest")}
        out["mean_d_vs_loop"] = {k: statistics.fmean(summ[BOTH][s][k] - summ[LOOP][s][k] for s in seeds)
                                 for k in ("best_loss", "depth1", "deepest")}
        out["d_by_seed_vs_loop"] = {s: summ[BOTH][s]["best_loss"] - summ[LOOP][s]["best_loss"] for s in seeds}
        out["p21"] = R.p17(summ[BOTH], summ[LOOP])
        out["p21_status"] = out["contrasts_vs_loop"]["best_loss"]
        ref_s = {s: R.summary([ref[s]]) for s in seeds}
        out["both_vs_reference_read"] = {"verdict": R.contrast(summ[BOTH], ref_s, "best_loss"),
                                         "mean_d": statistics.fmean(summ[BOTH][s]["best_loss"] - ref[s] for s in seeds)}
        mean = lambda xs: statistics.fmean(xs)
        out["effects"] = {
            "read_added_to_loop": mean(summ[BOTH][s]["best_loss"] - summ[LOOP][s]["best_loss"] for s in seeds),
            "loop_added_to_read": mean(summ[BOTH][s]["best_loss"] - ref[s] for s in seeds),
            "read_added_to_b0": mean(ref[s] - b0[s] for s in seeds) if len(b0) == 3 else None,
            "loop_added_to_b0": mean(summ[LOOP][s]["best_loss"] - b0[s] for s in seeds) if len(b0) == 3 else None}
    out["mean_gain_by_depth"] = {n: [statistics.fmean(summ[n][s]["gains"][i] for s in seeds) for i in range(5)] for n in arms}
    out["cost"] = {n: {"s_per_step": statistics.fmean(r["train_seconds"] / r["steps"] for r in rs.values()),
                       "peak_vram_mib": max(r["peak_vram_mib"] for r in rs.values()),
                       "host_rss_peak_mib": max(r["host_rss_peak_mib"] for r in rs.values())} for n, rs in runs.items()}
    (ROOT / "results" / "stage2l.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in out if k in ("p21_status", "contrasts_vs_loop", "mean_d_vs_loop", "effects", "futility_stop")}, indent=2))
    return out


if __name__ == "__main__":
    main()
