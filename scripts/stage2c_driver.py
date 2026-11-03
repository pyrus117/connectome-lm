"""Stage 2C driver: the sequential-list arms, then the twins test (PREREG_stage2c.md).
Models are the finished B0-M and COUNT-M seed 0 runs under STAGE2_RUNS (default <repo>/runs2).

    STAGE2_RUNS=/path/to/runs2 .venv/bin/python scripts/stage2c_driver.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("STAGE2_RUNS", ROOT / "runs2"))
ENV = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def main():
    for stage in ("lists", "twins"):
        out = ROOT / "results" / f"stage2c_{stage}.json"
        if out.exists():
            print(f"{out.name} exists, skipping {stage}", flush=True)
            continue
        code = subprocess.run([sys.executable, str(ROOT / "scripts" / "stage2c.py"), stage, "--runs", str(RUNS)],
                              cwd=ROOT, env=ENV).returncode
        if code != 0:
            sys.exit(f"stage2c.py {stage} failed with exit code {code}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
