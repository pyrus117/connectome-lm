"""Time one training step per layout and apply the layout rule (PREREG.md section 3).

Random token ids only: no text, no validation file, no loss is kept. For each
layout and arm it finds the largest micro-batch that fits, times full steps
(64 sequences, with accumulation), and projects GPU-hours for the whole matrix.
Rule: use all six layers if the projected total fits --budget-hours, otherwise
layers (2, 3); if neither fits, exit 3.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

from clm.model import ModelConfig, build_model
from clm.train import TrainConfig, param_groups

ROOT = Path(__file__).resolve().parent.parent
LAYOUTS = ((0, 1, 2, 3, 4, 5), (2, 3))
CONN_RUNS, DENSE_RUNS = 12, 3


def time_step(mcfg, arm, micro, timed, device):
    model = build_model(mcfg, arm, 0, ROOT / "data" / "mb_alpn_783.npz").to(device)
    opt = torch.optim.AdamW(param_groups(model, 0.1), lr=1e-4)
    tc = TrainConfig()
    accum = tc.batch // micro
    x = torch.randint(0, mcfg.vocab, (micro, mcfg.ctx + 1), device=device)

    def step():
        for _ in range(accum):
            with torch.autocast(device.type, dtype=torch.bfloat16):
                _, loss = model(x[:, :-1], x[:, 1:])
            (loss / accum).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(timed):
        step()
    torch.cuda.synchronize()
    return (time.time() - t) / timed, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-hours", type=float, required=True)
    ap.add_argument("--timed", type=int, default=5)
    ap.add_argument("--micros", type=int, nargs="+", default=[16, 8, 4, 2])
    # cap PyTorch below the card's limit: on this laptop the internal panel's
    # link failed once while a test filled all 4 GB (2026-10-21)
    ap.add_argument("--mem-fraction", type=float, default=1.0)
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    steps_per_run = TrainConfig().tokens // (TrainConfig().batch * 256)
    report = {"gpu": torch.cuda.get_device_name(), "steps": ModelConfig().steps,
              "micros_tried": args.micros, "mem_fraction": args.mem_fraction, "layouts": {}}
    for layout in LAYOUTS:
        mcfg = ModelConfig(conn_layers=layout)
        row = {}
        for arm in ("real", "dense"):
            for micro in args.micros:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    sec, gb = time_step(mcfg, arm, micro, args.timed, device)
                except torch.OutOfMemoryError:
                    continue
                row[arm] = {"micro": micro, "sec_per_step": sec, "peak_gb": gb,
                            "hours_per_run": sec * steps_per_run / 3600}
                print(json.dumps({"layout": layout, "arm": arm, **row[arm]}), flush=True)
                break
            else:
                row[arm] = {"oom": True}
        if all("hours_per_run" in row[a] for a in row):
            row["total_hours"] = (CONN_RUNS * row["real"]["hours_per_run"]
                                  + DENSE_RUNS * row["dense"]["hours_per_run"])
        report["layouts"][str(layout)] = row
    fits = [l for l in LAYOUTS if report["layouts"][str(l)].get("total_hours", 1e9) <= args.budget_hours]
    report["budget_hours"] = args.budget_hours
    report["chosen_layout"] = list(fits[0]) if fits else None
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "speed.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not fits:
        sys.exit(3)


if __name__ == "__main__":
    main()
