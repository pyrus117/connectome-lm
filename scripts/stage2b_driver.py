"""Stage 2B driver: COUNT with the Kenyon backoff and COUNT with the random-bucket control, three seeds each
at the COUNT learning rate, then the paired analysis. Follows PREREG_stage2b.md. Resumable like the others.

    .venv/bin/python scripts/stage2b_driver.py
"""
import json
import subprocess
import sys

import stage2_0_driver as base
from stage2_0_driver import OUT, ROOT

base.CONFIG = "configs/stage2b.json"
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
ARMS = {"count_kenyon": "data/count_kenyon", "count_random": "data/count_randbucket"}


def main():
    lr = json.loads((OUT / "pilot_count" / "choice.json").read_text())["lr"]          # COUNT's rate from Stage 2A
    count_args = ["--lr", str(lr), "--prereg", "PREREG_stage2a.md", "--bucket-dir", ""]

    def runs(seeds):
        for seed in seeds:
            for arm, directory in ARMS.items():
                out = OUT / f"{arm}-M-{seed}"
                args = ["--lr", str(lr), "--bucket-dir", directory]
                if base.train("M", seed, out, args) is None:     # diverged: rerun once from scratch
                    out.rename(out.with_name(out.name + "-diverged1"))
                    if base.train("M", seed, out, args) is None:
                        sys.exit(f"DIVERGED TWICE {out}; reported, no verdict")

    def analyse():
        names = [str(p) for p in sorted(OUT.glob("count-M-[0-9]")) + sorted(OUT.glob("count_kenyon-M-[0-9]"))
                 + sorted(OUT.glob("count_random-M-[0-9]")) if (p / "results.json").exists()]
        for script, extra in (("bucket_eval.py", names), ("stage2b_analyze.py", [])):
            subprocess.run([sys.executable, str(ROOT / "scripts" / script), *extra], check=True, cwd=ROOT, env=ENV)
        return json.loads((ROOT / "results" / "stage2b.json").read_text())

    runs((0, 1, 2))
    result = analyse()
    if result["status"] == "NEEDS_SEEDS":                        # PREREG_stage2b.md section 5
        runs((3, 4))
        for seed in (3, 4):
            out = OUT / f"count-M-{seed}"
            base.train("M", seed, out, count_args)               # COUNT's own seeds 3 and 4, if 2A did not run them
        result = analyse()
    print("VERDICT", result["verdict"], "P4", result["p4"]["supported"], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
