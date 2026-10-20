"""A small pre-norm GPT whose layers share one recurrent brain instead of FFNs."""
import hashlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from clm.block import SharedBrain
from clm.graphs import ARMS, make_wiring

RUN_ARMS = ARMS
_RESIDUAL = (".attn.proj.weight", ".w_out.weight")
_NORM_WEIGHT = ("ln1.weight", "ln2.weight", "ln_f.weight")


@dataclass(frozen=True)
class ModelConfig:
    vocab: int = 8192
    ctx: int = 256
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 4
    leak: float = 0.5
    k_io: int = 256
    norm: bool = True


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
    """Attention, then one step of the shared brain in place of an FFN."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.w_in = nn.Linear(cfg.d_model, cfg.k_io, bias=False)
        self.w_out = nn.Linear(cfg.k_io, cfg.d_model, bias=False)

    def forward(self, x, h, brain):
        x = x + self.attn(self.ln1(x))
        with torch.autocast(x.device.type, enabled=False):
            u = self.ln2(x).reshape(-1, x.shape[-1]).float()
            h = brain.step(h, self.w_in(u))
            y = self.w_out(brain.read(h)).reshape(x.shape)
        return x + y, h


class GPT(nn.Module):
    def __init__(self, cfg, brain):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.ctx, cfg.d_model)
        self.brain = brain
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        h = None                          # the brain starts empty for every token
        for layer in self.layers:
            x, h = layer(x, h, self.brain)
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
            if name == "brain.values":
                new = torch.randn(p.shape, generator=g) * model.brain.init_std.cpu()
            elif p.dim() == 1:
                new = torch.ones(p.shape) if name.endswith(_NORM_WEIGHT) else torch.zeros(p.shape)
            else:
                std = resid_std if name.endswith(_RESIDUAL) else 0.02
                new = torch.randn(p.shape, generator=g) * std
            p.copy_(new)


def build_model(cfg, arm, seed, region):
    if arm not in RUN_ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {RUN_ARMS}")
    brain = SharedBrain(make_wiring(region, arm, seed, cfg.k_io), cfg.leak, cfg.norm)
    model = GPT(cfg, brain)
    init_weights(model, seed)
    return model
