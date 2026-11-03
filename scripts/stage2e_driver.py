"""Stage 2E driver: the gated COUNT read and Kenyon buckets under the gate, three seeds each at COUNT's learning
rate; then the gated random-bucket control and seeds 3 and 4 only when the analysis asks. PREREG_stage2e.md.
Resumable like the others.

    .venv/bin/python scripts/stage2e_driver.py
"""
import json
import subprocess
import sys

import stage2_0_driver as base
from stage2_0_driver import OUT, ROOT

base.CONFIG = "configs/stage2e.json"
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
ARMS = {"count_gated": ["--count-gate", "--bucket-dir", ""],
        "count_kenyon_gated": ["--count-gate", "--bucket-dir", "data/count_kenyon"],
        "count_random_gated": ["--count-gate", "--bucket-dir", "data/count_randbucket"],
        "count": ["--bucket-dir", "", "--prereg", "PREREG_stage2a.md"]}      # COUNT's own seeds 3 and 4 only


def main():
    lr = json.loads((OUT / "pilot_count" / "choice.json").read_text())["lr"]          # COUNT's rate from Stage 2A

    def runs(arms, seeds):
        for seed in seeds:
            for arm in arms:
                out, args = OUT / f"{arm}-M-{seed}", ["--lr", str(lr), *ARMS[arm]]
                if base.train("M", seed, out, args) is None:     # diverged: rerun once from scratch
                    out.rename(out.with_name(out.name + "-diverged1"))
                    if base.train("M", seed, out, args) is None:
                        sys.exit(f"DIVERGED TWICE {out}; reported, no verdict")

    def analyse():
        names = [str(p) for arm in ("count", "count_gated", "count_kenyon_gated", "count_random_gated")
                 for p in sorted(OUT.glob(f"{arm}-M-[0-9]")) if (p / "results.json").exists()]
        for script, extra in (("bucket_eval.py", names), ("stage2e_analyze.py", [])):
            subprocess.run([sys.executable, str(ROOT / "scripts" / script), *extra], check=True, cwd=ROOT, env=ENV)
        return json.loads((ROOT / "results" / "stage2e.json").read_text())

    main_arms = ["count_gated", "count_kenyon_gated"]
    if (OUT / "count_random_gated-M-0").exists():               # a resumed driver that already ran the control
        main_arms.append("count_random_gated")
    runs(main_arms, (0, 1, 2))
    result = analyse()
    if result["status"] == "NEEDS_RANDOM":                       # PREREG_stage2e.md section 5
        main_arms.append("count_random_gated")
        runs(["count_random_gated"], (0, 1, 2))
        result = analyse()
    if result["status"] == "NEEDS_SEEDS":
        runs(main_arms + ["count"], (3, 4))
        result = analyse()
    print("VERDICT", result["verdict"], "P10a", result["p10a"], "P10b", result["p10b"], "P11", result["p11"],
          "P11s", result["p11_similarity"], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
