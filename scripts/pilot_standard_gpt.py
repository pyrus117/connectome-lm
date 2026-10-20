"""Reference pilot: a traditional GPT trained exactly like the brain pilots.

Same data order, schedule, token budget and starting weights for every shared
parameter (init_weights keys each tensor by name). The only change is that
each layer has an ordinary FFN instead of a step of the shared brain.

    python scripts/pilot_standard_gpt.py --hidden 1024 --out runs/pilot-standard-1024
"""
import argparse
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import clm.train as T
from clm.model import Attention, init_weights

ROOT = Path(__file__).resolve().parent.parent


class FFNLayer(nn.Module):
    def __init__(self, cfg, hidden):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.w_in = nn.Linear(cfg.d_model, hidden)
        self.w_out = nn.Linear(hidden, cfg.d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.w_out(F.gelu(self.w_in(self.ln2(x))))


class StandardGPT(nn.Module):
    def __init__(self, cfg, hidden):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.ctx, cfg.d_model)
        self.layers = nn.ModuleList(FFNLayer(cfg, hidden) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # train.run records model.brain.fingerprint; this model has no brain
        self.brain = types.SimpleNamespace(fingerprint=f"standard-ffn-{hidden}")

    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for layer in self.layers:
            x = layer(x)
        logits = self.ln_f(x) @ self.tok.weight.t()
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                       targets.reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot_norm_on.json"))
    args = ap.parse_args()
    mcfg, tcfg = T.load_config(args.config)

    def build(cfg, arm, seed, region):
        model = StandardGPT(cfg, args.hidden)
        init_weights(model, seed)
        return model

    T.build_model = build
    print("parameters:", sum(p.numel() for p in build(mcfg, None, 0, None).parameters()), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T.run("er", 0, mcfg, tcfg, args.out, device, config_path=args.config)


if __name__ == "__main__":
    main()
