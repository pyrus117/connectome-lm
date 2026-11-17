"""Render publication figures for the connectome-lm paper (Stages 1 and 2).

Reads exact numbers from `runs/analysis.json` and `results/stage2*.json` and writes
PNG (200 DPI) and SVG figures into `docs/figures/`.

Run:
    .venv/bin/python scripts/render_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "figures"

INK = "#1b2430"
MUTED = "#5b6770"
GRID = "#e2e7ec"
BLUE = "#2b5c8f"
TEAL = "#2a8c82"
GREEN = "#3d7a47"
AMBER = "#c87d20"
RED = "#b03a2e"
PURPLE = "#6c4f8c"
GREY = "#7f8c8d"


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#c5cdd6",
        "axes.linewidth": 0.9,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.85,
    })


def save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{name}.png"
    svg = OUT_DIR / f"{name}.svg"
    fig.savefig(png, dpi=200, facecolor="white", bbox_inches="tight")
    fig.savefig(svg, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png.relative_to(ROOT)} and {svg.relative_to(ROOT)}")


def fig1_stage1_connectome():
    with open(ROOT / "runs" / "analysis.json") as f:
        s1 = json.load(f)
    with open(ROOT / "results" / "stage2t.json") as f:
        s2t = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), dpi=200)
    fig.subplots_adjust(wspace=0.26)

    # Panel A: Stage 1 losses by arm (seeds 0, 1, 2; highlight clean seeds 0 & 2 vs spike seed 1)
    arms = ["dense", "er", "real_randio", "shuf", "real"]
    labels = [
        "dense\n(760 FCN)",
        "er\n(Erdős–Rényi)",
        "real_randio\n(random I/O)",
        "shuf\n(deg-preserving)",
        "real\n(FlyWire MB)",
    ]
    colors = [GREEN, TEAL, AMBER, BLUE, RED]
    x = np.arange(len(arms))

    for i, arm in enumerate(arms):
        vals = [s1["losses"][arm][str(s)] for s in (0, 1, 2)]
        clean_mean = (vals[0] + vals[2]) / 2.0
        ax1.bar(x[i], clean_mean, width=0.56, color=colors[i], alpha=0.82, edgecolor=INK, linewidth=0.8)
        for s_idx, v in enumerate(vals):
            marker = "o" if s_idx in (0, 2) else "^"
            lbl = None
            if i == 0 and s_idx == 0:
                lbl = "Seeds 0, 2 (clean)"
            elif i == 0 and s_idx == 1:
                lbl = "Seed 1 (grad spike in dense/er/randio)"
            ax1.scatter(
                x[i] + (s_idx - 1) * 0.11,
                v,
                color="white" if s_idx in (0, 2) else "#fdebd0",
                edgecolor=INK,
                s=38,
                marker=marker,
                zorder=5,
                label=lbl,
            )
        ax1.text(x[i], clean_mean - 0.009, f"{clean_mean:.3f}", ha="center", va="top",
                 fontsize=8.2, color="white", weight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.4)
    ax1.set_ylim(3.86, 4.055)
    ax1.set_ylabel("Validation cross-entropy (nats/token, lower is better)")
    ax1.set_title("A. Stage 1: Connectome Wiring vs Controls (200M tokens)")
    ax1.grid(axis="y")
    ax1.legend(loc="upper left", frameon=True, fontsize=8.0)
    ax1.annotate(
        "Primary contrast (real − shuf):\n+0.0126, +0.0147, +0.0160 nats\nVerdict: X_WORSE",
        xy=(4, 3.948), xytext=(2.45, 4.015),
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.1},
        fontsize=8.2, weight="bold", color=RED,
        bbox={"boxstyle": "round,pad=0.3", "fc": "#fdf2e9", "ec": RED, "lw": 0.8},
    )

    # Panel B: Stage 2T Participation Ratio across layers 1-6 (readout vs full brain)
    layers = np.arange(1, 7)
    pr_full = {}
    pr_read = {}
    for arm in ("real", "shuf", "dense"):
        rows_arm = [r for r in s2t["rows"] if r["arm"] == arm]
        pr_full[arm] = np.mean([r["pr_full_by_layer"] for r in rows_arm], axis=0)
        pr_read[arm] = np.mean([r["pr_readout_by_layer"] for r in rows_arm], axis=0)

    ax2.plot(layers, pr_read["shuf"], "-o", color=BLUE, lw=2.0, label="shuf readout PR (95.5% 1-hop)")
    ax2.plot(layers, pr_read["real"], "-s", color=RED, lw=2.0, label="real readout PR (19.5% 1-hop)")
    ax2.plot(layers, pr_read["dense"], "--^", color=GREEN, lw=1.5, alpha=0.85, label="dense readout PR")
    ax2.plot(layers, pr_full["shuf"], ":o", color=BLUE, lw=1.4, alpha=0.75, label="shuf full-brain PR")
    ax2.plot(layers, pr_full["real"], ":s", color=RED, lw=1.4, alpha=0.75, label="real full-brain PR")

    ax2.set_xticks(layers)
    ax2.set_xlabel("Transformer layer / recurrent brain step (1 to 6)")
    ax2.set_ylabel("Participation ratio PR = (tr C)² / tr(C²)")
    ax2.set_title("B. Stage 2T: Full-Brain vs Readout Dimension Across Layers")
    ax2.grid(True)
    ax2.legend(loc="upper left", frameon=True, fontsize=7.9)
    ax2.annotate(
        "Readout is 1 step late in real\n(ALPN → KC → MBON = 2 hops;\nlayer 2 PR: 13.7 vs 42.8)",
        xy=(2, 13.7), xytext=(2.75, 11.0),
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.1},
        fontsize=8.1, color=RED, weight="bold",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#fdf2e9", "ec": RED, "lw": 0.8},
    )

    save(fig, "fig1_stage1_connectome")


def fig2_memory_ladder_and_classes():
    with open(ROOT / "results" / "stage2_0.json") as f:
        s20 = json.load(f)
    with open(ROOT / "results" / "stage2a.json") as f:
        s2a = json.load(f)
    with open(ROOT / "results" / "stage2b.json") as f:
        s2b = json.load(f)
    with open(ROOT / "results" / "stage2e.json") as f:
        s2e = json.load(f)
    with open(ROOT / "results" / "stage2g.json") as f:
        s2g = json.load(f)
    with open(ROOT / "results" / "stage2m.json") as f:
        s2m = json.load(f)
    with open(ROOT / "results" / "stage2n.json") as f:
        s2n = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7), dpi=200)
    fig.subplots_adjust(wspace=0.25)

    # Panel A: Scaling ladder (non-embedding params in millions vs val loss)
    n_s, n_m, n_l = 6.816, 13.763, 29.360
    loss_s = s20["per_size"]["S"]["mean"]
    loss_m = s20["per_size"]["M"]["mean"]
    loss_l = s20["per_size"]["L"]["mean"]

    n_grid = np.linspace(6.0, 33.0, 200)
    a_fit = s20["ladder_fit"]["a"]
    b_fit = s20["ladder_fit"]["b"]
    ax1.plot(n_grid, a_fit * (n_grid * 1e6) ** (-b_fit), "--", color=GREY, lw=1.3,
             label="B0 power-law fit ($L = 8.031 N^{-0.0524}$)")
    ax1.plot([n_s, n_m, n_l], [loss_s, loss_m, loss_l], "-o", color=INK, lw=2.0, markersize=7,
             label="B0 ladder (shared rate 8e-3)")

    b0_sep = np.mean(list(s2g["loss"]["b0_reference"].values()))
    count_shared = np.mean(list(s2a["count"].values()))
    count_sep = np.mean(list(s2g["loss"]["count_separate"].values()))
    loop_shared = np.mean([s2m["summary"]["loop_count_shared"][str(s)]["best_loss"] for s in (0, 1, 2)])
    loop_headlow = np.mean([s2n["summary"]["loop_count_headlow"][str(s)]["best_loss"] for s in (0, 1, 2)])

    ax1.scatter([n_m], [b0_sep], color=BLUE, s=55, zorder=5, label=f"B0-M separate rate ({b0_sep:.4f})")
    ax1.scatter([n_m], [count_shared], color=TEAL, s=65, marker="D", zorder=5,
                label=f"COUNT-M shared ({count_shared:.4f}, −0.080 vs B0)")
    ax1.scatter([n_m], [count_sep], color=GREEN, s=75, marker="s", zorder=6,
                label=f"COUNT-M separate ({count_sep:.4f}, −0.122 vs B0)")
    ax1.scatter([n_m], [loop_headlow], color=AMBER, s=75, marker="^", zorder=6,
                label=f"LOOP+COUNT HEADLOW ({loop_headlow:.4f}, Stage 2N)")
    ax1.scatter([n_m], [loop_shared], color=RED, s=85, marker="*", zorder=7,
                label=f"LOOP+COUNT SHARED ({loop_shared:.4f}, Stage 2M)")

    ax1.axhline(loss_l, color=GREY, linestyle=":", lw=1.0)
    ax1.text(20.5, loss_l + 0.005, f"B0-L (29.4M): {loss_l:.3f}", fontsize=8.1, color=MUTED)

    ax1.set_xlabel("Backbone non-embedding parameters (millions)")
    ax1.set_ylabel("Validation cross-entropy (nats/token)")
    ax1.set_title("A. Size Ladder vs Host-RAM COUNT & Looped Models")
    ax1.grid(True)
    ax1.legend(loc="upper right", frameon=True, fontsize=7.7)

    # Panel B: Loss gap vs B0-M across 4-token history count classes
    classes = ["not stored\n(47.3%)", "1–9\n(17.4%)", "10–99\n(19.5%)", "100–999\n(11.5%)", ">=1000\n(4.2%)"]
    keys = ["not stored", "1-9", "10-99", "100-999", ">=1000"]
    b0_c = [s2e["class_losses_mean"]["b0"][k] for k in keys]
    cnt_c = [s2e["class_losses_mean"]["count"][k] - b0_c[i] for i, k in enumerate(keys)]
    ken_c = [s2b["class_losses_mean"]["kenyon"][k] - b0_c[i] for i, k in enumerate(keys)]
    gat_c = [s2e["class_losses_mean"]["gated"][k] - b0_c[i] for i, k in enumerate(keys)]

    xc = np.arange(len(classes))
    w = 0.25
    ax2.bar(xc - w, cnt_c, width=w, color=GREEN, label="Plain COUNT (Stage 2A)")
    ax2.bar(xc, ken_c, width=w, color=AMBER, label="COUNT + Kenyon K31 (Stage 2B)")
    ax2.bar(xc + w, gat_c, width=w, color=BLUE, label="COUNT-gated (Stage 2E)")
    ax2.axhline(0.0, color=INK, lw=0.9)

    ax2.set_xticks(xc)
    ax2.set_xticklabels(classes, fontsize=8.3)
    ax2.set_ylabel("Validation loss difference vs B0-M (nats, negative = better)")
    ax2.set_title("B. Memory Effect by Exact 4-Token History Count Class")
    ax2.grid(axis="y")
    ax2.legend(loc="lower right", frameon=True, fontsize=8.0)

    save(fig, "fig2_memory_ladder_and_classes")


def fig3_cls_consolidation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6), dpi=200)
    fig.subplots_adjust(wspace=0.26)

    # Panel A: Stage 2C / 2C-2 / 2C-3 recall & backward transfer
    arms = [
        "grad_b0\n(fine-tune)",
        "grad_count\n(fine-tune)",
        "write\n(1-shot)",
        "consolidate\n(rolling)",
        "orig_teacher\n(no mem-KL)",
        "orig_teacher\n+ mem-KL",
    ]
    mem_off_or_main = [0.753, 0.512, 0.760, 0.061, 0.763, 0.731]
    mem_on = [None, None, 0.760, 0.040, 0.410, 0.750]
    unseen_5th = [None, None, 0.797, None, 0.177, 0.763]

    x = np.arange(len(arms))
    w = 0.26
    ax1.bar(x - w / 2, mem_off_or_main, width=w, color=BLUE, label="Quiz recall (slow weights / primary)")
    mask_on = [i for i, v in enumerate(mem_on) if v is not None]
    ax1.bar([x[i] + w / 2 for i in mask_on], [mem_on[i] for i in mask_on], width=w, color=TEAL,
            label="Quiz recall with Overlay ON")
    mask_u = [i for i, v in enumerate(unseen_5th) if v is not None]
    ax1.scatter([x[i] + w / 2 for i in mask_u], [unseen_5th[i] for i in mask_u], color=RED, s=55,
                zorder=6, marker="D", label="Held-out 5th list recall (Overlay ON)")

    ax1.axhline(0.710, color=GREEN, linestyle="--", lw=1.0, label="Registered consolidation bar (0.710)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(arms, fontsize=7.8)
    ax1.set_ylim(0, 0.94)
    ax1.set_ylabel("Mean recall across 4 sequential lists (384 pairs/list)")
    ax1.set_title("A. Stage 2C / 2C-2 / 2C-3: Test-Time Writes & Consolidation")
    ax1.grid(axis="y")
    ax1.legend(loc="upper left", frameon=True, fontsize=7.5)

    # Panel B: Context-gated Overlay writes (Stage 2C-3 Part 4b)
    g_names = [
        "Ungated write\n(Stage 2C)",
        "Gated $C(h_3) \\leq 0$\n(81% penalty cut, 100% recall)",
        "Gated $C(h_2) \\leq 9$\n(95.4% penalty cut, ≤0.05 bar)",
        "Gated $C(h_2) \\leq 0$\n(unseen bigram)",
    ]
    recalls = [0.7602, 0.7602, 0.7207, 0.5430]
    penalties = [0.8138, 0.1543, 0.0376, 0.0091]
    cols = [AMBER, GREEN, TEAL, GREY]
    offsets = [(-0.24, -0.045), (0.035, 0.012), (0.035, -0.042), (0.035, 0.005)]

    for i in range(len(g_names)):
        ax2.scatter(penalties[i], recalls[i], color=cols[i], s=95, edgecolor=INK, zorder=5)
        dx, dy = offsets[i]
        ax2.text(penalties[i] + dx, recalls[i] + dy, g_names[i], fontsize=8.0,
                 weight="bold" if i in (1, 2) else "normal")

    ax2.axhline(0.710, color=GREEN, linestyle="--", lw=1.0, label="Min recall bar (0.710)")
    ax2.axvline(0.050, color=RED, linestyle=":", lw=1.1, label="Low-interference LM bar (+0.05 nats)")
    ax2.set_xlim(-0.04, 0.90)
    ax2.set_ylim(0.48, 0.83)
    ax2.set_xlabel("Validation English cross-entropy rise after 4 lists (nats, lower = less interference)")
    ax2.set_ylabel("Mean quiz recall across 4 lists (higher is better)")
    ax2.set_title("B. Stage 2C-3: Context-Gated Overlay Trade-off")
    ax2.grid(True)
    ax2.legend(loc="lower right", frameon=True, fontsize=8.0)

    save(fig, "fig3_cls_consolidation")


def fig4_loop_depth_and_halting():
    with open(ROOT / "results" / "stage2h.json") as f:
        s2h = json.load(f)
    with open(ROOT / "results" / "stage2i_seed0.json") as f:
        s2i0 = json.load(f)
    with open(ROOT / "results" / "stage2m.json") as f:
        s2m = json.load(f)
    with open(ROOT / "results" / "stage2n.json") as f:
        s2n = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7), dpi=200)
    fig.subplots_adjust(wspace=0.25)

    depths = np.arange(1, 7)
    ref_d = np.mean([s2h["val_loss_by_depth"]["reference"][str(s)] for s in (0, 1, 2)], axis=0)
    both_d = np.mean([s2h["val_loss_by_depth"]["both"][str(s)] for s in (0, 1, 2)], axis=0)

    # Read per-depth losses for SHARED and HEADLOW from runs2/*/results.json
    shared_d = []
    headlow_d = []
    for s in (0, 1, 2):
        with open(ROOT / "runs2" / f"loop_count_shared-M-{s}" / "results.json") as f:
            shared_d.append(json.load(f)["val_loss_by_depth"])
        with open(ROOT / "runs2" / f"loop_count_headlow-M-{s}" / "results.json") as f:
            headlow_d.append(json.load(f)["val_loss_by_depth"])
    shared_mean = np.mean(shared_d, axis=0)
    headlow_mean = np.mean(headlow_d, axis=0)

    ax1.plot(depths, ref_d, "-o", color=GREY, lw=1.8, label="Stage 2D/2H reference loop (sampled depth, no gain)")
    ax1.plot(depths, both_d, "-s", color=BLUE, lw=2.0, label="Stage 2H loop_gain + all-depth (P17: −0.0384 nats)")
    ax1.plot(depths, headlow_mean, "-^", color=AMBER, lw=2.0, label="Stage 2N LOOP+COUNT HEADLOW (3.2044)")
    ax1.plot(depths, shared_mean, "-*", color=RED, lw=2.2, markersize=8, label="Stage 2M LOOP+COUNT SHARED (3.2012)")
    ax1.axhline(3.4041, color=INK, linestyle=":", lw=1.1, label="B0-M baseline (3.4041)")
    ax1.axhline(3.2629, color=GREEN, linestyle="--", lw=1.1, label="COUNT-M separate rate (3.2629)")

    ax1.set_xticks(depths)
    ax1.set_xlabel("Recurrent loop passes (depth r = 1..6; total blocks = 4 + 2r)")
    ax1.set_ylabel("Validation cross-entropy (nats/token)")
    ax1.set_title("A. Validation Loss vs Loop Depth (3-Seed Mean)")
    ax1.grid(True)
    ax1.legend(loc="upper right", frameon=True, fontsize=7.6)

    # Panel B: Stage 2I Halting Frontiers on Seed 0
    fx = [pt[1] for pt in s2i0["fixed_depth"]]
    ax2.plot(depths, fx, "-o", color=INK, lw=2.0, label="Fixed depth (1..6)")

    for key, lbl, col, ls in [
        ("change", "State-change exit (free exit, +0.0014)", BLUE, "--"),
        ("entropy", "Entropy exit (−0.0305, X_WORSE)", RED, "-."),
        ("probe_state", "Linear state probe (+0.0034, R²=0.8%)", GREEN, "-"),
        ("oracle", "Oracle executed (+0.0626 ceiling)", PURPLE, ":"),
    ]:
        pts = s2i0["frontiers"][key]
        md = [p["mean_depth"] for p in pts]
        el = [p["executed_loss"] for p in pts]
        ax2.plot(md, el, ls, color=col, lw=1.8, marker=".", label=lbl)

    ax2.axvspan(1.25, 2.70, color="#eef4fa", alpha=0.6, label="Registered P18 evaluation band [1.25, 2.70]")
    ax2.set_xlim(1.0, 6.1)
    ax2.set_ylim(3.265, 3.405)
    ax2.set_xlabel("Mean recurrent passes per token (mean depth)")
    ax2.set_ylabel("Executed validation cross-entropy (nats/token)")
    ax2.set_title("B. Adaptive Halting Frontiers vs Oracle (Seed 0)")
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True, fontsize=7.6)

    save(fig, "fig4_loop_depth_and_halting")


def fig5_stage2mn_2x2_diagnosis():
    with open(ROOT / "results" / "stage2m.json") as f:
        s2m = json.load(f)
    with open(ROOT / "results" / "stage2n.json") as f:
        s2n = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7), dpi=200)
    fig.subplots_adjust(wspace=0.25)

    depths = np.arange(1, 7)
    arms_spec = [
        ("loop_count_shared", "SHARED (embed 8e-3, head 8e-3): Δ = +0.0376", RED, "-o", (0, 1, 2)),
        ("loop_count_headlow", "HEADLOW (embed 64e-3, head 8e-3): Δ = +0.0309", GREEN, "-s", (0, 1, 2)),
        ("loop_count_gainlow", "GAINLOW (embed 64e-3, head 64e-3): Δ = +0.00002", AMBER, "--^", (0, 1, 2)),
        ("loop_count_embedlow", "EMBEDLOW (embed 8e-3, head 64e-3): Δ = +0.00005", PURPLE, ":D", (0,)),
    ]

    for prefix, label, col, style, seeds in arms_spec:
        curves = []
        for s in seeds:
            with open(ROOT / "runs2" / f"{prefix}-M-{s}" / "results.json") as f:
                curves.append(json.load(f)["val_loss_by_depth"])
        mean_c = np.mean(curves, axis=0)
        # Plot relative to depth 1 (in millinats or nats) or absolute loss
        ax1.plot(depths, mean_c, style, color=col, lw=2.0, markersize=6, label=label)

    ax1.set_xticks(depths)
    ax1.set_xlabel("Recurrent loop depth r (passes 1 to 6)")
    ax1.set_ylabel("Validation cross-entropy (nats/token)")
    ax1.set_title("A. Stage 2M/2N 2×2 Factorial: Per-Depth Validation Loss")
    ax1.grid(True)
    ax1.legend(loc="upper right", frameon=True, fontsize=7.7)

    # Panel B: Learned |loop_gain| profile across passes 1..6 and ||head.weight||_2
    sh_g = np.mean([s2m["gain_rows_mean_abs"]["loop_count_shared"][str(s)] for s in (0, 1, 2)], axis=0)
    hl_g = np.mean([s2n["gain_rows_mean_abs"]["loop_count_headlow"][str(s)] for s in (0, 1, 2)], axis=0)
    gl_g = np.mean([s2m["gain_rows_mean_abs"]["loop_count_gainlow"][str(s)] for s in (0, 1, 2)], axis=0)
    el_g = np.array(s2n["gain_rows_mean_abs"]["loop_count_embedlow"]["0"])

    ax2.plot(depths, sh_g, "-o", color=RED, lw=2.0, label="SHARED (||head||₂ = 731, loop alive)")
    ax2.plot(depths, hl_g, "-s", color=GREEN, lw=2.0, label="HEADLOW (||head||₂ = 727, loop alive)")
    ax2.plot(depths, el_g, ":D", color=PURPLE, lw=2.0, label="EMBEDLOW (||head||₂ = 5,202, loop dead)")
    ax2.plot(depths, gl_g, "--^", color=AMBER, lw=2.0, label="GAINLOW (||head||₂ = 5,825, loop dead)")

    ax2.set_xticks(depths)
    ax2.set_xlabel("Recurrent loop pass r (1 to 6)")
    ax2.set_ylabel("Mean absolute learned residual gain |g_r|")
    ax2.set_title("B. Learned Loop Gain Profile & Unembedding Norm Explosion")
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True, fontsize=7.8)
    ax2.annotate(
        "head.weight at 64e-3 (wd=0) grows 8× in L2 norm,\nlocking readout to pass 1 and collapsing passes 2–6",
        xy=(2, 0.015), xytext=(2.4, 0.16),
        arrowprops={"arrowstyle": "->", "color": PURPLE, "lw": 1.1},
        fontsize=8.1, color=INK, weight="bold",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#f5eef8", "ec": PURPLE, "lw": 0.8},
    )

    save(fig, "fig5_stage2mn_2x2_diagnosis")


def main():
    setup_style()
    fig1_stage1_connectome()
    fig2_memory_ladder_and_classes()
    fig3_cls_consolidation()
    fig4_loop_depth_and_halting()
    fig5_stage2mn_2x2_diagnosis()


if __name__ == "__main__":
    main()
