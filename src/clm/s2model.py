"""Stage 2 baseline B0: a small modern GPT (speedrun-style recipe).

Pre-norm, RMSNorm without gain, RoPE, QK-norm, ReLU^2 MLP, zero-initialised
output projections and head, no biases, untied head, value embeddings on the
first and last layer, logit softcap. Plan: docs/superpowers/plans/
2026-11-02-stage2-kenyon-memory.md, section 7.1. Stage 1's model.py is untouched.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from clm.model import _seed_for

_ZERO_INIT = (".proj.weight", ".down.weight", "head.weight")


@dataclass(frozen=True)
class B0Config:
    vocab: int = 8192
    ctx: int = 512
    d_model: int = 384
    n_layer: int = 6
    n_head: int = 6
    mlp_mult: int = 4
    softcap: float = 15.0
    rope_base: float = 10000.0


def rms(x):
    return F.rms_norm(x, (x.shape[-1],))


def rotary(ctx, head_dim, base):
    inv = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    ang = torch.outer(torch.arange(ctx, dtype=torch.float32), inv)
    return torch.cos(ang), torch.sin(ang)


def apply_rotary(x, cos, sin):
    """x: (batch, heads, time, head_dim); rotates the two halves of each head."""
    t = x.shape[-2]
    a, b = x.chunk(2, dim=-1)
    c, s = cos[:t].to(x.dtype), sin[:t].to(x.dtype)
    return torch.cat((a * c - b * s, a * s + b * c), dim=-1)


class Block(nn.Module):
    def __init__(self, cfg, value_embed):
        super().__init__()
        d = cfg.d_model
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, cfg.mlp_mult * d, bias=False)
        self.down = nn.Linear(cfg.mlp_mult * d, d, bias=False)
        self.ve_gate = nn.Parameter(torch.zeros(1)) if value_embed else None

    def forward(self, x, cos, sin, ve=None):
        b, t, d = x.shape
        hd = d // self.n_head
        q, k, v = self.qkv(rms(x)).view(b, t, 3, self.n_head, hd).permute(2, 0, 3, 1, 4)
        if self.ve_gate is not None:
            v = v + self.ve_gate * ve.view(b, t, self.n_head, hd).transpose(1, 2)
        q, k = apply_rotary(rms(q), cos, sin), apply_rotary(rms(k), cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        return x + self.down(F.relu(self.up(rms(x))).square())


class B0GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.d_model % cfg.n_head or (cfg.d_model // cfg.n_head) % 2:
            raise ValueError("d_model must split into heads of even size")
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.value_embed = nn.ModuleList(nn.Embedding(cfg.vocab, cfg.d_model) for _ in range(2))
        self.ve_layers = (0, cfg.n_layer - 1)
        self.blocks = nn.ModuleList(Block(cfg, i in self.ve_layers) for i in range(cfg.n_layer))
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        cos, sin = rotary(cfg.ctx, cfg.d_model // cfg.n_head, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, targets=None, extra=None):
        """extra: an optional (batch, time, d_model) input added to the embedding
        (the Stage 2 memory arms use it)."""
        x = self.embed(idx)
        if extra is not None:
            x = x + extra
        x = rms(x)
        ves = {layer: emb(idx) for layer, emb in zip(self.ve_layers, self.value_embed)}
        for i, blk in enumerate(self.blocks):
            x = blk(x, self.cos, self.sin, ves.get(i))
        logits = self.head(rms(x)).float()
        logits = self.cfg.softcap * torch.tanh(logits / self.cfg.softcap)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    def non_embedding_params(self):
        skip = {id(self.embed.weight), *(id(e.weight) for e in self.value_embed)}
        return sum(p.numel() for p in self.parameters() if id(p) not in skip)


def init_b0(model, seed):
    """Name-keyed generators, as in Stage 1: a parameter that exists in two arms
    starts identical in both."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            g = torch.Generator().manual_seed(_seed_for(seed, name))
            if name.endswith(_ZERO_INIT) or name.endswith("ve_gate"):
                new = torch.zeros(p.shape)
            elif "embed" in name:
                new = torch.randn(p.shape, generator=g)
            else:
                bound = 1.0 / math.sqrt(p.shape[1])
                new = (torch.rand(p.shape, generator=g) * 2 - 1) * bound
            p.copy_(new)


def build_b0(cfg, seed):
    model = B0GPT(cfg)
    init_b0(model, seed)
    return model


def param_groups(model, optimizer, lr, wd):
    """Muon takes the 2-D block matrices; AdamW takes embeddings, head and gates.
    With AdamW alone, block matrices get weight decay and the rest none."""
    matrices, rest = [], []
    for name, p in model.named_parameters():
        (matrices if name.startswith("blocks.") and p.dim() == 2 else rest).append(p)
    adam_rest = {"params": rest, "weight_decay": 0.0, "lr": lr}
    if optimizer == "adamw":
        return [torch.optim.AdamW([{"params": matrices, "weight_decay": wd, "lr": lr}, adam_rest],
                                  betas=(0.9, 0.95), fused=False)]
    if optimizer == "muon":
        return [torch.optim.Muon(matrices, lr=lr, weight_decay=wd, adjust_lr_fn="match_rms_adamw"),
                torch.optim.AdamW([adam_rest], betas=(0.9, 0.95))]
    raise ValueError(f"unknown optimizer {optimizer!r}")
