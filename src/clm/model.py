"""A small pre-norm GPT whose FFN slots can hold a connectome block."""
import hashlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from clm.block import ConnectomeFFN, DenseFFN, connectome_param_count, matched_hidden
from clm.graphs import make_wiring

RUN_ARMS = ("real", "shuf", "er", "real_randio", "dense")
_RESIDUAL = (".attn.proj.weight", ".ffn.proj.weight", ".ffn.w_out.weight")
_NORM_WEIGHT = ("ln1.weight", "ln2.weight", "ln_f.weight")


@dataclass(frozen=True)
class ModelConfig:
    vocab: int = 8192
    ctx: int = 256
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 4
    conn_layers: tuple = (0, 1, 2, 3, 4, 5)
    steps: int = 4
    leak: float = 0.5
    k_io: int = 256


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_head, d // self.n_head).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(b, t, d))


class Layer(nn.Module):
    def __init__(self, cfg, ffn):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn, self.ffn = Attention(cfg), ffn

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg, ffns):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.ctx, cfg.d_model)
        self.layers = nn.ModuleList(Layer(cfg, f) for f in ffns)
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for layer in self.layers:
            x = layer(x)
        logits = self.ln_f(x) @ self.tok.weight.t()          # tied embeddings
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                       targets.reshape(-1))


def _seed_for(seed, name):
    digest = hashlib.sha256(f"{seed}|{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


def init_weights(model, seed):
    """Every parameter gets its own generator keyed by (seed, name), so a
    parameter that exists in two arms starts identical in both, whatever else
    each arm builds."""
    resid_std = 0.02 / math.sqrt(2 * model.cfg.n_layer)
    with torch.no_grad():
        for name, p in model.named_parameters():
            g = torch.Generator().manual_seed(_seed_for(seed, name))
            if name.endswith(".values"):
                ffn = model.get_submodule(name.rsplit(".", 1)[0])
                new = torch.randn(p.shape, generator=g) * ffn.init_std.cpu()
            elif p.dim() == 1:
                new = torch.ones(p.shape) if name.endswith(_NORM_WEIGHT) else torch.zeros(p.shape)
            else:
                std = resid_std if name.endswith(_RESIDUAL) else 0.02
                new = torch.randn(p.shape, generator=g) * std
            p.copy_(new)


def build_model(cfg, arm, seed, region):
    if arm not in RUN_ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {RUN_ARMS}")
    if arm == "dense":
        ref = make_wiring(region, "real", seed, cfg.k_io)
        hidden = matched_hidden(cfg.d_model, connectome_param_count(cfg.d_model, ref))
        slot = lambda: DenseFFN(cfg.d_model, hidden)
    else:
        wiring = make_wiring(region, arm, seed, cfg.k_io)
        slot = lambda: ConnectomeFFN(cfg.d_model, wiring, cfg.steps, cfg.leak)
    ffns = [slot() if i in cfg.conn_layers else DenseFFN(cfg.d_model, 4 * cfg.d_model)
            for i in range(cfg.n_layer)]
    model = GPT(cfg, ffns)
    init_weights(model, seed)
    return model
