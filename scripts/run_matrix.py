"""Run all 15 (arm, seed) runs of the frozen config, skipping completed ones.

Order is seed-major so an interruption leaves whole seeds complete. With
--gpus N, N workers share one queue and each is pinned to one GPU, so the
cards stay busy even though the dense runs are shorter than the others.
Prints only completion status, never a loss: losses are read once, by
clm.analyze, after every run is done.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from clm.model import RUN_ARMS

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
CONFIG = ROOT / "configs" / "stage1.json"


def is_complete(out):
    res = out / "results.json"
    return res.exists() and json.loads(res.read_text()).get("status") == "complete"


def worker(gpu, jobs, failed, lock):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    while True:
        try:
            arm, seed = jobs.get_nowait()
        except queue.Empty:
            return
        out = ROOT / "runs" / f"{arm}-{seed}"
        code = subprocess.run([str(PY), "-m", "clm.train", "--arm", arm, "--seed", str(seed),
                               "--config", str(CONFIG), "--out", str(out)], env=env).returncode
        with lock:
            print(f"{arm}-{seed}: exit {code} (gpu {gpu})", flush=True)
            if code != 0:
                failed.append(f"{arm}-{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=1)
    args = ap.parse_args()
    if subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                      capture_output=True, text=True).stdout.strip():
        sys.exit("working tree is dirty; commit before production runs")
    jobs = queue.Queue()
    for seed in (0, 1, 2):
        for arm in RUN_ARMS:
            if is_complete(ROOT / "runs" / f"{arm}-{seed}"):
                print(f"{arm}-{seed}: already complete")
            else:
                jobs.put((arm, seed))
    failed, lock = [], threading.Lock()
    threads = [threading.Thread(target=worker, args=(g, jobs, failed, lock)) for g in range(args.gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        sys.exit(f"failed runs: {sorted(failed)}")


if __name__ == "__main__":
    main()
