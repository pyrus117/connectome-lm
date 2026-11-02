"""PREREG_stage2_0.md §5: SESOI from B0-M seed spread, and the S/M/L power-law ladder fit.

Writes results/stage2_0.json. Run after all nine ladder runs report status "complete".
"""
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = ("S", "M", "L")
SEEDS = (0, 1, 2)


def load():
    runs = {}
    for size in SIZES:
        for seed in SEEDS:
            path = ROOT / "runs2" / f"b0-{size}-{seed}" / "results.json"
            if not path.exists():
                sys.exit(f"missing {path}")
            r = json.loads(path.read_text())
            if r.get("status") != "complete" or not math.isfinite(r["final_val_loss"]):
                sys.exit(f"{path}: status {r.get('status')}, loss {r['final_val_loss']}")
            if r.get("git_dirty"):
                sys.exit(f"{path}: run recorded a dirty tree")
            runs[(size, seed)] = r
    return runs


def fit_power_law(ns, losses):
    """Least squares on log L = log a - b log N; returns a, b, max abs residual in nats."""
    xs = [math.log(n) for n in ns]
    ys = [math.log(v) for v in losses]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = math.exp(my - slope * mx)
    b = -slope
    resid = max(abs(a * n ** -b - v) for n, v in zip(ns, losses))
    return a, b, resid


def main():
    runs = load()
    per_size = {}
    for size in SIZES:
        losses = [runs[(size, s)]["final_val_loss"] for s in SEEDS]
        secs = [runs[(size, s)]["train_seconds"] for s in SEEDS]
        per_size[size] = {
            "non_embedding_params": runs[(size, 0)]["non_embedding_params"],
            "losses": losses,
            "mean": statistics.fmean(losses),
            "sd": statistics.stdev(losses),
            "train_seconds_mean": statistics.fmean(secs),
            "peak_vram_mib": runs[(size, 0)]["peak_vram_mib"],
        }
    sd_m = per_size["M"]["sd"]
    sesoi = max(0.01, 2 * sd_m)
    ns = [per_size[s]["non_embedding_params"] for s in SIZES]
    means = [per_size[s]["mean"] for s in SIZES]
    a, b, resid = fit_power_law(ns, means)

    def size_multiplier(gain, base="M"):
        """Non-embedding size multiple B0 needs to lower its loss by `gain` nats from `base`."""
        target = per_size[base]["mean"] - gain
        return (per_size[base]["mean"] / target) ** (1 / b) if gain < per_size[base]["mean"] else math.inf

    out = {
        "per_size": per_size,
        "sesoi": {"sd_M": sd_m, "two_sd_M": 2 * sd_m, "sesoi": sesoi,
                  "floor_binds": sesoi == 0.01},
        "ladder_fit": {"form": "L = a * N^-b, N = non-embedding params, fit on size means",
                       "a": a, "b": b, "max_abs_residual_nats": resid},
        "size_multiplier_at_M": {f"{g:.2f}": size_multiplier(g) for g in (0.01, 0.02, 0.05, 0.10)},
    }
    dst = ROOT / "results" / "stage2_0.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
