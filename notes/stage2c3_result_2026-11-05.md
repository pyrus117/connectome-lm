# Stage 2C-3 result: memory-on consolidation failed from overlay-channel drift (not double-supply), memory-consistency fixes it, and count-gated overlays cut the English loss penalty by 81–95% (2026-11-05)

Pre-registration `PREREG_stage2c3.md` (commit `9dd70dc`, sha256 `618eb04a6c25b91a0361a7e7b6a4814f742725150e0a81db52870bc53d1fd6ff`, recorded in `results/stage2c3.json`). Code `src/clm/consolidate_fix.py` (`GatedOverlay`, `run_write_gated`, `run_consolidate_variant`), driver `scripts/stage2c3.py`, tests `tests/test_consolidate_fix.py`, mutants M127 to M133 (`scripts/mutation_audit.py`); results `results/stage2c3.json`. Run 2026-11-05 18:34 to 19:11 NZDT on the laptop RTX 3050 (4 GB) with `--mem-fraction 0.75` under `systemd-run --user`. Model: the finished COUNT-M seed 0 (`runs2/count-M-0`, Stage 2A). Task seeds 0, 1, 2; write weight 4.0 (from `results/stage2c_lists.json`). No deviations and no crashes. Before the run the full test suite (291 passed, 1 skipped) and `scripts/mutation_audit.py` (all 133 mutants killed by assertion, C1 and C2 as required) passed.

## Verdict Summary

### Part 4a: Memory-on interference in consolidation

- **H1a (overlay-channel drift in `original_teacher`): SUPPORTED.** On a held-out 128-pair list (`unseen_list`) never trained into the slow weights, `original_teacher`'s memory-on recall drops from **0.797** (`0.797, 0.852, 0.742`) before consolidation to **0.177** (`0.172, 0.211, 0.148`) after phase 4—a loss of **-0.620** (`-0.625, -0.641, -0.594`), more than six times the -0.10 bar.
- **H1b (double-supply interference in `original_teacher`): NOT SUPPORTED (refuted).** `original_teacher`'s recall on the unconsolidated `unseen_list` (0.177) is *lower*, not higher, than its memory-on `final_mean` on the consolidated Lists 1–4 (0.410; needed $\ge +0.10$). The Stage 2C-2 note's guess that "the backbone and the overlay both supply the answer" was wrong: consolidating with `overlay=None` only de-calibrates the backbone's overlay-reading channel, and on Lists 1–4 the consolidated slow weights were partially rescuing that broken channel.
- **Repair 4a (`original_teacher_memory_consistency`, `memory_kl_weight = 1.0`): WORKS (passes all four registered conditions).**
  1. Memory-off consolidation passes all three Stage 2C conditions: `final_mean` **0.731** ($\ge 0.710$), `backward_transfer` **+0.019** ($\ge -0.351$), language-loss rise **+0.031** nats ($\le 0.05$ in every seed).
  2. Memory-on `final_mean` rises from 0.410 (`original_teacher`) to **0.750** (`0.762, 0.724, 0.765`), a gain of **+0.340** (`+0.296, +0.344, +0.380`, clearing +0.10 in every seed and $\ge 0.710$).
  3. Memory-on `backward_transfer` is **-0.032** (`-0.020, -0.053, -0.024`), within $\pm 0.05$ of zero in the mean.
  4. Final unseen-list recall `unseen_with_memory[-1]` is **0.763** (`0.797, 0.789, 0.703`) against **0.797** at the start (change **-0.034**, within 0.05 of `unseen_start`, and **+0.586** above `original_teacher`'s 0.177).
- **Rolling teacher rescue (`rolling_teacher_memory_consistency`): SUPPORTED, does not reach full consolidation.** Adding `memory_kl_weight = 1.0` lifts the rolling teacher's memory-off `final_mean` from 0.061 (Stage 2C) to **0.562** (`0.481, 0.605, 0.602`, gain **+0.502**), clearing +0.10 in every seed, with positive backward transfer (+0.052). It falls short of the 0.710 bar because the rolling teacher's recall still drifts gradually across phases (`0.75 -> 0.66 -> 0.53 -> 0.57` instead of collapsing to `0.018`).

### Part 4b: Context-gated `Overlay` writes

- **`write_gated_k3_c0` (`gate_k = 3, max_hcount = 0`): WORKS.** Suppressing the 1-token overlay write whenever the 3-token history is stored in `order4` ($C(h_3) \ge 2$) preserves **100% of quiz recall** (`final_mean` **0.7602** vs **0.7602** for ungated `write`; **0.0000 difference in every seed**), keeps `backward_transfer` at **0.000**, and cuts the validation language-loss rise after 4 lists from **+0.814** nats (`4.175`) to **+0.154** nats (`3.516`)—an **81.0% reduction** in damage to ordinary English.
- **`write_gated_k2_c9` (`gate_k = 2, max_hcount = 9`): WORKS and passes the LOW-INTERFERENCE bar ($\le 0.05$ nats).** Suppressing the overlay when the 2-token history has count $\ge 10$ keeps `final_mean` at **0.721** (`0.713, 0.710, 0.739`, within 0.05 of ungated `write` in every seed and $\ge 0.710$), `backward_transfer` at **0.000**, and cuts the language-loss rise after 4 lists to **+0.038** nats (`3.399` vs `3.362` memory off)—a **95.4% reduction** in the penalty, under the 0.05-nat bar in every seed.
- **`write_gated_k2_c0` (`gate_k = 2, max_hcount = 0`): FAILS recall condition (1).** Suppressing on any stored 2-token history ($C(h_2) \ge 1$) reduces the language-loss rise to **+0.009** nats (`3.371`, a 98.9% cut), but drops quiz `final_mean` to **0.543** (`0.524, 0.534, 0.571`) because about 29% of random `(ans_{j-1}, cue_j)` bigrams in the packed quiz sequence occurred at least once in the 881M-token corpus.

---

## Detailed Tables

### Part 4a: Consolidation variants (4 lists of 384 pairs, 300 steps/list, lr 3e-4)

Comparison numbers from Stage 2C (`results/stage2c_lists.json`): `write` `final_mean` 0.760 (`0.759, 0.743, 0.779`), registered `consolidate` `final_mean` 0.061 (`0.054, 0.054, 0.074`), `gradient_count` `backward_transfer` -0.651. Starting language loss is 3.362.

| Variant | Memory-off `final_mean` (by seed / mean) | Memory-off `backward_transfer` | Memory-on `final_mean` (by seed / mean) | Memory-on `backward_transfer` | Unseen 5th list recall (`start -> phase 4`) | Language loss rise |
|---|---|---|---|---|---|---|
| `original_teacher` (`memory_kl_weight=0`) | 0.771, 0.777, 0.740 (**0.763**) | +0.028 | 0.466, 0.380, 0.385 (**0.410**) | -0.090 | 0.797 $\to$ **0.177** (`0.172, 0.211, 0.148`) | +0.033 |
| `original_teacher_memory_consistency` (`memory_kl_weight=1`) | 0.743, 0.685, 0.764 (**0.731**) | +0.019 | 0.762, 0.724, 0.765 (**0.750**) | -0.032 | 0.797 $\to$ **0.763** (`0.797, 0.789, 0.703`) | +0.031 |
| `rolling_teacher_memory_consistency` (`memory_kl_weight=1`) | 0.481, 0.605, 0.602 (**0.562**) | +0.052 | 0.559, 0.658, 0.677 (**0.631**) | +0.006 | 0.797 $\to$ **0.510** (`0.414, 0.602, 0.516`) | +0.035 |

Unseen 5th-list recall (`unseen_with_memory`) across phases 1 to 4 (mean over seeds; start is 0.797):

| Variant | Start | After phase 1 | After phase 2 | After phase 3 | After phase 4 |
|---|---|---|---|---|---|
| `original_teacher` | 0.797 | 0.281 | 0.214 | 0.174 | **0.177** |
| `original_teacher_memory_consistency` | 0.797 | 0.651 | 0.724 | 0.760 | **0.763** |
| `rolling_teacher_memory_consistency` | 0.797 | 0.654 | 0.576 | 0.570 | **0.510** |

In `original_teacher`, most of the damage to the overlay-reading channel already happens in phase 1 (`0.797 -> 0.281`), which also explains why Stage 2C's rolling teacher collapsed from phase 2 onward.

### Part 4b: Context-gated `Overlay` writes (4 lists of 384 pairs, write weight 4.0, no training)

Language loss with memory off is 3.362.

| Variant | Gate rule | `final_mean` by seed | Mean `final_mean` | Gap to ungated `write` | `lm_loss_memory_on` after list 4 (by seed / mean) | Language loss rise | Penalty cut | Verdict |
|---|---|---|---|---|---|---|---|---|
| `write` (Stage 2C, ungated) | none | 0.759, 0.743, 0.779 | **0.760** | 0.000 | 4.125, 4.253, 4.147 (**4.175**) | +0.814 | 0% | baseline |
| `write_gated_k3_c0` | $C(h_3) \le 0$ | 0.759, 0.743, 0.779 | **0.760** | **0.000** | 3.516, 3.516, 3.516 (**3.516**) | **+0.154** | **81.0%** | **WORKS** |
| `write_gated_k2_c9` | $C(h_2) \le 9$ | 0.713, 0.710, 0.739 | **0.721** | -0.040 | 3.400, 3.399, 3.399 (**3.399**) | **+0.038** | **95.4%** | **WORKS (low-interference)** |
| `write_gated_k2_c0` | $C(h_2) \le 0$ | 0.524, 0.534, 0.571 | **0.543** | -0.217 | 3.370, 3.372, 3.371 (**3.371**) | **+0.009** | 98.9% | FAILS recall |

Cost: each `write_gated` run took 13 to 15 seconds; each `original_teacher` consolidation run took 191 to 207 seconds; each `memory_consistency` run took 255 to 259 seconds (~29% slower than `original_teacher` because of the second student forward pass on cue batches). Total wall-clock for all 18 runs was 36 minutes.
