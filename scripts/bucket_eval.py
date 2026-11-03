"""Validation loss by history-count bucket for finished runs (Stage 2 memory arms
and their B0 baseline), from each run's final checkpoint. A position's bucket is
the corpus count of its 3-token (and, separately, 4-token) history: not stored,
1-9, 10-99, 100-999, >=1000 (clm.countread.BUCKET_LABELS; orders 4 and 5 store
no history seen once, so "not stored" includes those).

    .venv/bin/python scripts/bucket_eval.py runs2/b0-M-0 runs2/count-M-0 ...
Writes <run>/bucket.json. The overall loss must match results.json's final_val_loss.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from clm.countread import BUCKET_LABELS, CountReader, bucket_of, reader_for
from clm.data import eval_windows, get_batch, open_tokens
from clm.s2model import B0Config, build_b0
from clm.s2train import S2TrainConfig
from clm.train import ROOT, _autocast


@torch.no_grad()
def evaluate_run(run, device, reader):
    res = json.loads((run / "results.json").read_text())
    mcfg, tcfg = B0Config(**res["model"]), S2TrainConfig(**res["train"])
    model = build_b0(mcfg, res["seed"]).to(device)
    model.load_state_dict(torch.load(run / "ckpt.pt", map_location=device)["model"])
    model.eval()
    mem = (reader_for(ROOT / tcfg.count_dir, tcfg.count_k, ROOT / tcfg.bucket_dir if tcfg.bucket_dir else None)
           if tcfg.memory == "count" else None)
    val = open_tokens(ROOT / tcfg.val_bin)
    starts = eval_windows(len(val), mcfg.ctx, tcfg.eval_tokens)
    cpu = torch.device("cpu")
    acc = {k: np.zeros((len(BUCKET_LABELS), 2)) for k in (3, 4)}
    total, n = 0.0, 0
    for i in range(0, len(starts), tcfg.micro):
        x, y = get_batch(val, starts[i:i + tcfg.micro], mcfg.ctx, cpu)
        kw = {}
        if mem is not None:
            got = mem.lookup(x.numpy(), states=mcfg.count_orders > 0)
            kw["count"] = tuple(torch.from_numpy(a).to(device) for a in got)
        with _autocast(device):
            logits, _ = model(x.to(device), **kw)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.to(device).reshape(-1),
                              reduction="none").view(y.shape).cpu().numpy().astype(np.float64)
        total, n = total + nll.sum(), n + nll.size
        for k, a in acc.items():
            b = bucket_of(reader.history_counts(x.numpy(), k))
            for j in range(len(BUCKET_LABELS)):
                a[j] += (nll[b == j].sum(), (b == j).sum())
    out = {"run": run.name, "arm": res["arm"], "val_loss": total / n, "results_val_loss": res["final_val_loss"],
           "by_history_bucket": {f"history_{k}_tokens": {
               lab: {"share": float(a[j, 1] / a[:, 1].sum()), "loss": float(a[j, 0] / a[j, 1]) if a[j, 1] else None}
               for j, lab in enumerate(BUCKET_LABELS)} for k, a in acc.items()}}
    if abs(out["val_loss"] - res["final_val_loss"]) > 2e-3:
        raise RuntimeError(f"{run}: re-evaluated loss {out['val_loss']:.4f} differs from "
                           f"results.json {res['final_val_loss']:.4f}")
    (run / "bucket.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = CountReader(ROOT / "data/count")
    for arg in sys.argv[1:]:
        out = evaluate_run(Path(arg), device, reader)
        print(arg, round(out["val_loss"], 4), flush=True)


if __name__ == "__main__":
    main()
