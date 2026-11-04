"""Stage 2G driver: B0-M on separate Muon and AdamW rates, then gate 1 again against COUNT-M on separate rates
(Stage 2F). Pilot: B0-M seed 0, 50M tokens, Muon 8e-3, AdamW 32e-3 and 128e-3 (the shared 8e-3 and the 64e-3 cell
are already measured), edge extension once. Three seeds of B0 at the best AdamW rate only if it beats the shared rate by
the SESOI. PREREG_stage2g.md. Resumable.

    .venv/bin/python scripts/stage2g_driver.py
"""
import json
import subprocess
import sys

import stage2_0_driver as base
from clm import rules2g as R
from stage2_0_driver import OUT, PILOT_TOKENS, ROOT

base.CONFIG = "configs/stage2g.json"
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
PILOT = OUT / "pilot_2g"


def loss_of(path):
    return json.loads((path / "results.json").read_text())["final_val_loss"]


def pilot(adam):
    r = base.train("M", 0, PILOT / f"b0-a{adam:g}", ["--lr", str(R.MUON), "--adam-mult", str(adam / R.MUON),
                                                     "--tokens", str(PILOT_TOKENS)])
    return None if r is None else r["final_val_loss"]


def main():
    PILOT.mkdir(parents=True, exist_ok=True)
    scores = {R.SHARED: loss_of(OUT / "pilot" / "muon-0.008"),            # Stage 2.0's B0 pilot
              64e-3: loss_of(OUT / "pilot_2f" / "b0-m0.008-a0.064")}      # Stage 2F's B0 control
    for adam in R.GRID:
        if adam not in scores:
            scores[adam] = pilot(adam)
    win = R.best(scores)
    if win is None:
        sys.exit("every Stage 2G pilot diverged")
    for adam in R.extension(win, scores):
        scores[adam] = pilot(adam)
    win, separate = R.best(scores), R.separate(scores)
    choice = {"adam": win, "adam_mult": win / R.MUON, "separate": separate,
              "pilot": [{"adam": a, "val_loss": v} for a, v in sorted(scores.items())]}
    (PILOT / "choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print("CHOICE adam", win, "separate", separate, flush=True)
    if separate:
        args = ["--lr", str(R.MUON), "--adam-mult", str(win / R.MUON)]
        for seed in (0, 1, 2):
            out = OUT / f"b0_separate-M-{seed}"
            if base.train("M", seed, out, args) is None:                  # diverged: rerun once from scratch
                out.rename(out.with_name(out.name + "-diverged1"))
                if base.train("M", seed, out, args) is None:
                    sys.exit(f"DIVERGED TWICE {out}; reported, no verdict")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "stage2g_analyze.py")], check=True, cwd=ROOT, env=ENV)
    result = json.loads((ROOT / "results" / "stage2g.json").read_text())
    print("VERDICT", result["verdict"], "gate1", result["gate1"], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
