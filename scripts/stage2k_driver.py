"""Stage 2K driver: the gated COUNT read and the scale-only control on separate Muon (8e-3) and AdamW (64e-3) rates,
three seeds each, against plain COUNT rerun on the same machine and rates; seeds 3 and 4 of all three arms only when
the analysis asks. PREREG_stage2k.md. Resumable like the others.

    .venv/bin/python scripts/stage2k_driver.py
"""
import json
import subprocess
import sys

import stage2_0_driver as base
from stage2_0_driver import OUT, ROOT

base.CONFIG = "configs/stage2k.json"
ARMS = {"count_decoupled_q": [],                       # plain COUNT rerun on q-pc, the paired reference (Stage 2F's runs were on the laptop)
        "count_gated_sep": ["--count-gate"],
        "count_scaled_sep": ["--count-scale"]}


def main():
    def runs(arms, seeds):
        for seed in seeds:
            for arm in arms:
                out, args = OUT / f"{arm}-M-{seed}", ARMS[arm]
                if base.train("M", seed, out, args) is None:        # diverged: rerun once from scratch
                    out.rename(out.with_name(out.name + "-diverged1"))
                    if base.train("M", seed, out, args) is None:
                        sys.exit(f"DIVERGED TWICE {out}; reported, no verdict")

    def analyse():
        names = [str(p) for arm in ARMS for p in sorted(OUT.glob(f"{arm}-M-[0-9]")) if (p / "results.json").exists()]
        for script, extra in (("bucket_eval.py", names), ("stage2k_analyze.py", [])):
            subprocess.run([sys.executable, str(ROOT / "scripts" / script), *extra], check=True, cwd=ROOT,
                           env=base.child_env())
        return json.loads((ROOT / "results" / "stage2k.json").read_text())

    runs(list(ARMS), (0, 1, 2))
    result = analyse()
    if result["status"] == "NEEDS_SEEDS":                          # PREREG_stage2k.md section 5
        done = ROOT / "results" / "stage2k.json"                   # an untracked file would make the next runs record git_dirty
        (OUT / "stage2k_three_seeds.json").write_text(done.read_text())
        done.unlink()
        runs(list(ARMS), (3, 4))
        result = analyse()
    print("VERDICT gated_vs_count", result["verdicts"]["gated_vs_count"], "reading", result["reading"], "P19", result["p19"],
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
