"""Stage 2T: test plan predictions P1 and P2 on the 15 finished Stage 1 runs, no training.

P1 (Clark 2024): brain-activity participation ratio, real / shuf, near the
connectivity effective-rank ratio; a ratio near 1 refutes it here.
P2 (Liu 2024): relative change of the brain synapses, |v_T - v_0| / |v_0|,
larger for real than for shuf, er and dense.

Activity: the brain state after each layer, on the first n_tokens validation
tokens (fixed windows, as in evaluation). PR = tr(C)^2 / tr(C^2) of the
token-by-neuron covariance, computed from running sums, so no state is stored.
Usage: .venv/bin/python scripts/stage2t_mechanism.py [--tokens 16384]
Writes results/stage2t.json. Plan: docs/superpowers/plans/2026-11-02-stage2-kenyon-memory.md
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from clm.data import eval_windows, get_batch, open_tokens  # noqa: E402
from clm.finetune import load_pretrained  # noqa: E402
from clm.graphs import ARMS  # noqa: E402
from clm.model import build_model  # noqa: E402
from clm.train import load_config  # noqa: E402

CONFIG = ROOT / "configs" / "stage1.json"


class Moments:
    """Running sums for the covariance of activity vectors (rows = tokens)."""

    def __init__(self, dim, device):
        self.n = 0
        self.s = torch.zeros(dim, dtype=torch.float64, device=device)
        # float32 keeps six 6,293-square sums inside the 0.75 GPU memory cap
        self.ss = torch.zeros(dim, dim, dtype=torch.float32, device=device)

    def add(self, x):
        x = x.float()
        self.n += x.shape[0]
        self.s += x.sum(0).double()
        self.ss += x.T @ x

    def participation_ratio(self):
        mean = self.s / self.n
        cov = self.ss.double() / self.n - torch.outer(mean, mean)
        return float(torch.trace(cov) ** 2 / (cov * cov).sum())


@torch.no_grad()
def activity(model, val, ctx, n_tokens, micro, device):
    brain, n_layer = model.brain, model.cfg.n_layer
    full = [Moments(brain.n, device) for _ in range(n_layer)]
    readout = [Moments(brain.k_out, device) for _ in range(n_layer)]
    starts = eval_windows(len(val), ctx, n_tokens)
    if len(starts) * ctx < n_tokens:
        raise SystemExit(f"validation file holds fewer than {n_tokens} tokens")
    for i in range(0, len(starts), micro):
        x, _ = get_batch(val, starts[i:i + micro], ctx, device)
        h_x = model.tok(x) + model.pos(torch.arange(ctx, device=device))
        h = None
        for li, layer in enumerate(model.layers):
            h_x, h = layer(h_x, h, brain)
            full[li].add(h.T)
            readout[li].add(brain.read(h))
    return [m.participation_ratio() for m in full], [m.participation_ratio() for m in readout]


def weight_change(model, arm, seed, mcfg, tcfg):
    init = build_model(mcfg, arm, seed, ROOT / tcfg.region)
    out = {}
    trained = dict(model.named_parameters())
    for name in ("brain.values", "brain.bias"):
        v0 = dict(init.named_parameters())[name].detach().double()
        vt = trained[name].detach().cpu().double()
        norm0 = float(v0.norm())
        out[name] = float((vt - v0).norm()) / norm0 if norm0 > 0 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--micro", type=int, default=4)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs the GPU (sparse brain forward)")
    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    device = "cuda"
    mcfg, tcfg = load_config(CONFIG)
    val = open_tokens(ROOT / "data" / "fineweb" / "val.bin")
    rows = []
    for arm in ARMS:
        for seed in (0, 1, 2):
            run = ROOT / "runs" / f"{arm}-{seed}"
            model, res = load_pretrained(run, CONFIG, device)
            model.eval()
            pr_full, pr_read = activity(model, val, mcfg.ctx, args.tokens, args.micro, device)
            row = {"arm": arm, "seed": seed, "neurons": model.brain.n, "val_loss": res["final_val_loss"],
                   "pr_full_by_layer": pr_full, "pr_readout_by_layer": pr_read,
                   "rel_change": weight_change(model.cpu(), arm, seed, mcfg, tcfg)}
            rows.append(row)
            print(f"{arm}-{seed}: PR full (last layer) {pr_full[-1]:.1f}, PR readout (last) {pr_read[-1]:.1f}, "
                  f"rel change values {row['rel_change']['brain.values']:.3f}", flush=True)
            del model
            torch.cuda.empty_cache()
    out = ROOT / "results" / "stage2t.json"
    out.write_text(json.dumps({"tokens": args.tokens, "config": str(CONFIG.relative_to(ROOT)),
                               "torch": torch.__version__, "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
