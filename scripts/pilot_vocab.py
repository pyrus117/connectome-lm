"""Pilot the vocabulary-list task on one er model.

    python scripts/pilot_vocab.py --pairs 512 --lrs 3e-4 1e-3 3e-3
    python scripts/pilot_vocab.py --task twins --pairs 1536 --lrs 3e-4
    python scripts/pilot_vocab.py --task sequential --lists 4 --pairs 384 --lrs 3e-4 --steps 300

Pilots touch only the er arm, so task settings cannot be tuned toward real
vs shuf. Each setting runs clm.finetune in its own process; settings with a
summary.json are skipped. Writes results/pilot_vocab.json.
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/er-0")
    ap.add_argument("--pairs", type=int, nargs="+", required=True)
    ap.add_argument("--lrs", type=float, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--task", choices=("random", "twins", "sequential"), default="random")
    ap.add_argument("--lists", type=int, default=4)
    ap.add_argument("--mem-fraction", type=float, default=0.75)
    args = ap.parse_args()
    run_dir = ROOT / args.run
    arm = json.loads((run_dir / "results.json").read_text())["arm"]
    if arm != "er":
        sys.exit(f"{args.run} is arm {arm!r}; pilots use the er arm only")
    table_path = ROOT / "results" / "pilot_vocab.json"
    table = json.loads(table_path.read_text()) if table_path.exists() else {}
    for n, lr in itertools.product(args.pairs, args.lrs):
        task = "" if args.task == "random" else (f"{args.task}{args.lists}-" if args.task == "sequential"
                                                 else f"{args.task}-")
        name = f"{run_dir.name}-{task}n{n}-lr{lr:g}-s{args.steps}"
        out = ROOT / "runs" / "pilot-vocab" / name
        if not (out / "summary.json").exists():
            code = subprocess.run([str(PY), "-m", "clm.finetune", "--run", str(run_dir),
                                   "--config", str(ROOT / "configs" / "stage1.json"),
                                   "--out", str(out), "--pairs", str(n), "--lr", str(lr),
                                   "--steps", str(args.steps), "--task", args.task,
                                   "--lists", str(args.lists),
                                   "--mem-fraction", str(args.mem_fraction)]).returncode
            if code != 0:
                sys.exit(f"{name}: clm.finetune exited {code}")
        s = json.loads((out / "summary.json").read_text())
        secs = [json.loads(l)["seconds"] for l in (out / "log.jsonl").read_text().splitlines()[1:]]
        keys = ((("lists", "pairs_per_list", "chance", "final_mean", "backward_transfer", "acc_matrix"))
                if args.task == "sequential" else
                ("pairs", "chance", "start_acc", "final_acc", "steps_to_50", "steps_to_90"))
        row = {k: s[k] for k in keys}
        row.update(task=args.task, lr=lr, steps=args.steps, sec_per_step=round(sum(secs) / len(secs), 3))
        table[name] = row
        table_path.write_text(json.dumps(table, indent=2) + "\n")
        print(name, json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
