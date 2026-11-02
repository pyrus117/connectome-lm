"""Stage 2D driver: train the looped model once, then run the analysis.
Follows PREREG_stage2d.md. Resumable like the other drivers.

    .venv/bin/python scripts/stage2d_driver.py
"""
import subprocess
import sys

import stage2_0_driver as base
from stage2_0_driver import OUT, ROOT

base.CONFIG = "configs/stage2d.json"
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}


def main():
    out = OUT / "loop-M-0"
    if base.train("M", 0, out) is None:                  # diverged: rerun once from scratch
        out.rename(out.with_name(out.name + "-diverged1"))
        if base.train("M", 0, out) is None:
            sys.exit(f"DIVERGED TWICE {out}; reported, no analysis")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "stage2d_analyze.py"), str(out)], check=True,
                   cwd=ROOT, env=ENV)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
