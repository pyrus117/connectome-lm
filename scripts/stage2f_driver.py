"""Stage 2F driver: separate learning rates for COUNT's block matrices (Muon) and its embeddings, head and gates
(AdamW). A 3 x 4 pilot grid on COUNT-M seed 0 (the shared-rate cells come from Stage 2A), edge extension once, and
only if the best separate-rate cell beats the best shared-rate cell by the SESOI: three seeds at 200M tokens against
Stage 2A's COUNT runs, a B0 control pilot, and seeds 3 and 4 when the analysis asks. PREREG_stage2f.md. Resumable.

    .venv/bin/python scripts/stage2f_driver.py
"""
import json
import subprocess
import sys

import stage2_0_driver as base
from clm import rules2f as R
from stage2_0_driver import OUT, PILOT_TOKENS, ROOT

base.CONFIG = "configs/stage2f.json"
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
PILOT = OUT / "pilot_2f"
COUNT_ARGS = ["--lr", "0.032", "--bucket-dir", "", "--prereg", "PREREG_stage2a.md"]     # Stage 2A's recipe, seeds 3 and 4


def pilot(muon, adam, memory="count"):
    out = PILOT / f"{'count' if memory == 'count' else 'b0'}-m{muon:g}-a{adam:g}"
    r = base.train("M", 0, out, ["--lr", str(muon), "--adam-mult", str(adam / muon), "--memory", memory,
                                 "--tokens", str(PILOT_TOKENS)])
    return None if r is None else r["final_val_loss"]


def main():
    PILOT.mkdir(parents=True, exist_ok=True)
    stage2a = json.loads((OUT / "pilot_count" / "choice.json").read_text())["pilot"]
    scores = {(p["lr"], p["lr"]): p["val_loss"] for p in stage2a}       # the shared-rate cells, run in Stage 2A
    imported = set(scores)
    for cell in R.grid():
        if cell not in scores:
            scores[cell] = pilot(*cell)
    win = R.best(scores)
    if win is None:
        sys.exit("every Stage 2F pilot diverged")
    for cell in R.extensions(win, scores):                              # one round past the edge the winner sits on
        scores[cell] = pilot(*cell)
    win, confirm = R.best(scores), R.confirm(scores)
    choice = {"muon": win[0], "adam": win[1], "adam_mult": win[1] / win[0], "confirm": confirm,
              "pilot": [{"muon": m, "adam": a, "val_loss": v, "source": "stage2a" if (m, a) in imported else "stage2f"}
                        for (m, a), v in sorted(scores.items())]}
    if confirm:
        choice["b0_control"] = {"muon": win[0], "adam": win[1], "val_loss": pilot(*win, memory="none")}
    (PILOT / "choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print("CHOICE muon", win[0], "adam", win[1], "confirm", confirm, flush=True)
    if not confirm:
        print("VERDICT NO_PILOT_GAIN; the shared rate stays", flush=True)
        print("DONE", flush=True)
        return

    def runs(arm, args, seeds):
        for seed in seeds:
            out = OUT / f"{arm}-M-{seed}"
            if base.train("M", seed, out, args) is None:                # diverged: rerun once from scratch
                out.rename(out.with_name(out.name + "-diverged1"))
                if base.train("M", seed, out, args) is None:
                    sys.exit(f"DIVERGED TWICE {out}; reported, no verdict")

    def analyse():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "stage2f_analyze.py")], check=True, cwd=ROOT, env=ENV)
        return json.loads((ROOT / "results" / "stage2f.json").read_text())

    dargs = ["--lr", str(win[0]), "--adam-mult", str(win[1] / win[0])]
    runs("count_decoupled", dargs, (0, 1, 2))
    result = analyse()
    if result["status"] == "NEEDS_SEEDS":                               # PREREG_stage2f.md section 4
        runs("count_decoupled", dargs, (3, 4))
        runs("count", COUNT_ARGS, (3, 4))
        result = analyse()
    print("VERDICT", result["verdict"], "P12", result["p12"], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
