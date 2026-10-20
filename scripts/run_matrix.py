"""Run all 15 (arm, seed) runs of the frozen config, skipping completed ones.

Order is seed-major so an interruption leaves whole seeds complete. Prints
only completion status, never a loss: losses are read once, by clm.analyze,
after every run is done.
"""
import json
import subprocess
import sys
from pathlib import Path

from clm.model import RUN_ARMS

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
CONFIG = ROOT / "configs" / "stage1.json"


def main():
    if subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                      capture_output=True, text=True).stdout.strip():
        sys.exit("working tree is dirty; commit before production runs")
    failed = []
    for seed in (0, 1, 2):
        for arm in RUN_ARMS:
            out = ROOT / "runs" / f"{arm}-{seed}"
            res = out / "results.json"
            if res.exists() and json.loads(res.read_text()).get("status") == "complete":
                print(f"{arm}-{seed}: already complete")
                continue
            code = subprocess.run([str(PY), "-m", "clm.train", "--arm", arm, "--seed", str(seed),
                                   "--config", str(CONFIG), "--out", str(out)]).returncode
            print(f"{arm}-{seed}: exit {code}")
            if code != 0:
                failed.append(f"{arm}-{seed}")
    if failed:
        sys.exit(f"failed runs: {failed}")


if __name__ == "__main__":
    main()
