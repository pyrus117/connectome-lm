"""Render the connectome-lm architecture as built to a PNG.

Source of truth: HANDOVER.md, "The design as built" (lines 40-46).
Output: docs/architecture_as_built.png (and .svg beside it).

Run:
    .venv/bin/python scripts/render_architecture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT_PNG = ROOT / "docs" / "architecture_as_built.png"
OUT_SVG = ROOT / "docs" / "architecture_as_built.svg"

INK = "#1b2430"
MUTED = "#5b6770"
LINE = "#8b98a3"
PANEL = "#f4f6f8"
GREY = "#eef1f4"
MIND = "#e8eef6"
MIND_EDGE = "#4a6fa5"
ATTN = "#f5ecdf"
ATTN_EDGE = "#a9743a"
RES = "#e7f0e8"
RES_EDGE = "#4d7a55"
ACCENT = "#a33a3a"


def rbox(ax, x, y, w, h, text, fc, ec, fs=9.5, weight="normal", radius=0.8):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.22,rounding_size={radius}",
            linewidth=1.1, facecolor=fc, edgecolor=ec, zorder=3,
        )
    )
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, weight=weight, zorder=4, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=LINE, lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1, shrinkB=1, zorder=2,
    ))


def note(ax, x, y, text, fs=8.2, color=MUTED, ha="left", va="center"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va,
            style="italic", zorder=5, linespacing=1.4)


def panel(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=1.0, facecolor=PANEL, edgecolor="#d8dee4", zorder=0,
    ))
    ax.text(x + 1.8, y + h - 2.6, title, fontsize=11.5, color=INK,
            weight="bold", ha="left", va="center", zorder=5)


def build():
    fig, ax = plt.subplots(figsize=(16.5, 10.0), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 165)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(2, 97.4, "connectome-lm: the architecture as built", fontsize=17.5,
            color=INK, weight="bold", ha="left", va="center")
    ax.text(2, 94.0,
            "6-layer pre-norm GPT with no FFNs  ·  one shared, recurrently wired "
            "mushroom-body brain  ·  ≈ 5.11 M parameters",
            fontsize=10.0, color=MUTED, ha="left", va="center")

    # ---------------- A. Full model ----------------
    panel(ax, 2, 14, 45, 76, "A. Full model")
    cx, cw = 8.0, 33.0

    rbox(ax, cx, 82.0, cw, 5.0, "token ids  ·  ctx 256", GREY, LINE, fs=8.8)
    arrow(ax, cx + cw / 2, 82.0, cx + cw / 2, 78.9)
    rbox(ax, cx, 73.5, cw, 5.0,
         "token embedding  d 256\nvocab 8192, tied to LM head", GREY, LINE, fs=8.2)
    arrow(ax, cx + cw / 2, 73.5, cx + cw / 2, 69.4)

    for i, dy in enumerate((1.05, 0.6, 0.15)):
        ax.add_patch(FancyBboxPatch(
            (cx + dy + 0.55, 30 + dy + 0.4), cw, 39,
            boxstyle="round,pad=0.22,rounding_size=0.8",
            facecolor="#e2e8ee", edgecolor="#c3ccd4", linewidth=0.9, zorder=1 + i,
        ))
    rbox(ax, cx, 30, cw, 39, "", "#fbfcfd", MIND_EDGE)

    for i in range(6):
        y = 62.2 - i * 5.2
        rbox(ax, cx + 2.4, y, cw - 4.8, 4.2,
             f"Layer {i}:  attn → brain", "#ffffff", "#c8d2db", fs=8.3)
        if i < 5:
            arrow(ax, cx + cw / 2, y, cx + cw / 2, y - 1.0, lw=0.9)

    note(ax, cx + cw / 2, 35.4, "one SharedBrain,\nsix callers", fs=8.2, ha="center")
    note(ax, cx + cw / 2, 32.2, "no FFN anywhere", fs=8.3, ha="center", color=ACCENT)

    arrow(ax, cx + cw / 2, 30, cx + cw / 2, 25.6)
    rbox(ax, cx, 20.6, cw, 5.0, "final LayerNorm", GREY, LINE, fs=8.8)
    arrow(ax, cx + cw / 2, 20.6, cx + cw / 2, 19.1)
    rbox(ax, cx, 14.2, cw, 4.9, "LM head (tied) → logits 8192", GREY, LINE, fs=8.4)

    # ---------------- B. One layer ----------------
    panel(ax, 50, 58, 113, 34, "B. One layer, l ∈ 0…5: attention, then the brain")
    yb = 66.0

    ax.text(52.5, 70.5, "xₗ", fontsize=10.5, color=INK, ha="left", va="center")
    arrow(ax, 54.6, 70.5, 57.4, 70.5)

    rbox(ax, 57.4, yb, 11.6, 9.0, "LayerNorm\npre-norm", GREY, LINE, fs=8.4)
    arrow(ax, 69.0, 70.5, 72.0, 70.5)
    rbox(ax, 72.0, yb - 0.4, 16.0, 9.8, "causal\nself-attention\n4 heads, d 256",
         ATTN, ATTN_EDGE, fs=8.3)
    arrow(ax, 88.0, 70.5, 90.4, 70.5)

    ax.add_patch(plt.Circle((91.9, 70.5), 1.5, facecolor=RES, edgecolor=RES_EDGE,
                            linewidth=1.1, zorder=3))
    ax.text(91.9, 70.5, "+", fontsize=12.5, color=INK, ha="center", va="center", zorder=4)

    # attention residual bypass
    arrow(ax, 55.0, 70.5, 55.0, 80.0, color=RES_EDGE, lw=1.0)
    arrow(ax, 55.0, 80.0, 91.9, 80.0, color=RES_EDGE, lw=1.0)
    arrow(ax, 91.9, 80.0, 91.9, 72.0, color=RES_EDGE, lw=1.0)
    note(ax, 72.0, 81.3, "residual", fs=7.6, ha="center")

    arrow(ax, 93.4, 70.5, 96.0, 70.5)
    rbox(ax, 96.0, yb, 11.6, 9.0, "LayerNorm", GREY, LINE, fs=8.4)
    arrow(ax, 107.6, 70.5, 110.2, 70.5)
    rbox(ax, 110.2, yb, 12.2, 9.0, "w_in\nper layer\n→ 256", MIND, MIND_EDGE, fs=8.0)
    arrow(ax, 122.4, 70.5, 125.2, 70.5)
    rbox(ax, 125.2, 64.3, 14.6, 12.4, "Shared\nBrain\n\nW shared\nby all 6",
         MIND, MIND_EDGE, fs=8.5, weight="bold")
    arrow(ax, 139.8, 70.5, 142.6, 70.5, color=MIND_EDGE)
    rbox(ax, 142.6, yb, 12.2, 9.0, "w_out\nper layer\n← 256", MIND, MIND_EDGE, fs=8.0)
    arrow(ax, 154.8, 70.5, 157.2, 70.5)
    ax.add_patch(plt.Circle((158.4, 70.5), 1.2, facecolor=RES, edgecolor=RES_EDGE,
                            linewidth=1.1, zorder=3))
    ax.text(158.4, 70.5, "+", fontsize=11, color=INK, ha="center", va="center", zorder=4)
    ax.text(160.4, 70.5, "→ l+1", fontsize=8.6, color=INK, ha="left", va="center")

    note(ax, 52.5, 63.4,
         "leak 0.5  ·  GELU  ·  RMS-normalised over all neurons  ·  no learnable gain",
         fs=8.2)
    note(ax, 52.5, 60.7,
         "one brain step per layer per token; state carries across layers, "
         "empty at each new token", fs=8.2)

    # ---------------- C. The brain ----------------
    panel(ax, 50, 18, 113, 37, "C. FlyWire v783 wiring, 760 neurons")
    note(ax, 52.5, 51.2,
         "ALPN + KC + MBON + DAN + MBIN  ·  6,293 source neurons, 570,682 synapses",
         fs=8.4)

    rbox(ax, 54.0, 38.0, 21.0, 8.2, "input neurons\n256 lowest in-degree",
         MIND, MIND_EDGE, fs=8.0)
    rbox(ax, 138.0, 38.0, 21.0, 8.2, "readout neurons\n256 highest in-degree",
         MIND, MIND_EDGE, fs=8.0)

    ax.add_patch(FancyBboxPatch(
        (79.0, 32.0), 55.0, 20.0,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor="#dde7f3", edgecolor=MIND_EDGE, linewidth=1.2, zorder=3,
    ))
    ax.text(106.5, 47.6, "760 neurons  ·  570,682 synapses", fontsize=9.4,
            color=INK, ha="center", va="center", weight="bold", zorder=4)
    ax.text(106.5, 41.6, "one leaky step\nRMS-normalised, GELU",
            fontsize=8.3, color=MUTED, ha="center", va="center", zorder=4,
            linespacing=1.4)
    ax.text(106.5, 35.4, "no learnable gain", fontsize=8.3, color=MUTED,
            ha="center", va="center", zorder=4)

    arrow(ax, 75.0, 42.1, 79.0, 42.1, color=MIND_EDGE)
    arrow(ax, 134.0, 42.1, 138.0, 42.1, color=MIND_EDGE)

    note(ax, 54.0, 29.2,
         "On the real graph the inputs land on 249 ALPN (olfactory); the readouts are "
         "mostly DAN / MBON (reward, output).", fs=8.2)
    note(ax, 54.0, 25.6,
         "dense control arm: fully connected brain, 577,600 parameters vs 576,975 real.",
         fs=8.2)
    note(ax, 54.0, 22.2,
         "Every arm starts from bit-identical shared tensors; one fan-in init rule for the brain.",
         fs=8.2)

    # ---------------- D. Arms ----------------
    panel(ax, 2, 2, 161, 12.5, "D. Arms  ·  15 production runs = 5 arms × 3 seeds")
    arms = [
        ("real", "the FlyWire mushroom body"),
        ("shuf", "degree-preserving shuffle, fresh per seed"),
        ("er", "Erdős–Rényi, same edge count"),
        ("real_randio", "real edges, random input / readout"),
        ("dense", "fully connected brain, 760 neurons"),
    ]
    for i, (name, desc) in enumerate(arms):
        x = 4.0 + i * 31.6
        rbox(ax, x, 5.4, 13.0, 4.4, name, MIND, MIND_EDGE, fs=8.8, weight="bold")
        ax.text(x, 3.4, desc, fontsize=7.5, color=MUTED, ha="left", va="center")

    fig.savefig(OUT_PNG, dpi=200, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT_SVG, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG} ({OUT_PNG.stat().st_size:,} bytes)")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    try:
        build()
    except Exception:
        print("render failed", file=sys.stderr)
        raise
