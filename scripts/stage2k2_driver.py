"""Stage 2K2 driver: the gated read, the one-scalar control and a per-order scalar with the gate parameters on a slower
AdamW rate (8e-3), against plain COUNT, five seeds each. COUNT seeds 0 to 2 are Stage 2K's runs (skipped here); seeds 3
and 4 are new. PREREG_stage2k2.md. Resumable like the others.

    .venv/bin/python scripts/stage2k2_driver.py
"""
import json
import subprocess
import sys

import stage2_0_driver as base
from stage2_0_driver import OUT, ROOT

base.CONFIG = "configs/stage2k.json"
PREREG = ["--prereg", "PREREG_stage2k2.md"]
ARMS = {"count_decoupled_q": PREREG,
        "count_gated_low": ["--count-gate", "--gate-mult", "1.0", *PREREG],
        "count_scaled_low": ["--count-scale", "--gate-mult", "1.0", *PREREG],
        "count_oscaled_low": ["--count-scale-orders", "--gate-mult", "1.0", *PREREG]}


def main():
    for seed in range(5):
        for arm, args in ARMS.items():
            out = OUT / f"{arm}-M-{seed}"
            if arm == "count_decoupled_q" and seed < 3:           # Stage 2K's runs, trained under PREREG_stage2k.md
                if not (out / "results.json").exists():
                    sys.exit(f"missing Stage 2K run {out}")
                continue
            if base.train("M", seed, out, args) is None:          # diverged: rerun once from scratch
                out.rename(out.with_name(out.name + "-diverged1"))
                if base.train("M", seed, out, args) is None:
                    sys.exit(f"DIVERGED TWICE {out}; reported, no verdict")
    names = [str(p) for arm in ARMS for p in sorted(OUT.glob(f"{arm}-M-[0-9]")) if (p / "results.json").exists()]
    for script, extra in (("bucket_eval.py", names), ("stage2k2_analyze.py", [])):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), *extra], check=True, cwd=ROOT, env=base.child_env())
    result = json.loads((ROOT / "results" / "stage2k2.json").read_text())
    print("VERDICTS", result["verdicts_vs_count"], "P20", result["p20"], "carry_forward", result["carry_forward"], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
