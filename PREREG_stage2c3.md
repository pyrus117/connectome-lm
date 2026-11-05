# Stage 2C-3 pre-registration: memory-on consolidation consistency (4a) and context-gated overlay writes (4b)

Dated 2026-11-05. Written and committed before any Stage 2C-3 run; `scripts/stage2c3.py` records this file's sha256 in `results/stage2c3.json`. Follows `PREREG_stage2c.md` (`notes/stage2c_result_2026-11-04.md`) and `PREREG_stage2c2.md` (`notes/stage2c2_result_2026-11-04.md`), and executes flagged follow-ups 4a and 4b from the Stage 2 synthesis audit. Model: the finished COUNT-M seed 0 run (`runs2/count-M-0`, Stage 2A), never retrained during Stage 2C. Code: `src/clm/consolidate_fix.py`, `scripts/stage2c3.py`. The registered `src/clm/fastwrite.py` and `src/clm/fastwrite_arms.py` are untouched. GPU time: about 35 minutes on the laptop RTX 3050.

This is a pre-registered follow-up to the Stage 2C and 2C-2 notes. It does not alter the registered verdicts of Stages 2C or 2C-2.

## 1. Questions

Stages 2C and 2C-2 left two open faults in fast count writes and their consolidation:

### Part 4a: Why does a consolidated backbone read worse with the memory switched on, and how do we fix it?

In Stage 2C-2 (`notes/stage2c2_result_2026-11-04.md`), replacing the rolling teacher with the untouched original model (`original_teacher`) fixed memory-off consolidation (`final_mean` 0.765 across task seeds 0, 1, 2 against 0.760 for `write`). However, when that same consolidated student was quizzed with the memory (`overlay`) still switched on, its `final_mean` dropped to 0.413 (seed 0: 0.465, seed 1: 0.367, seed 2: 0.408). Two causes are possible:

- **H1a (overlay-channel drift).** During `original_teacher` consolidation, the student only runs with `overlay=None`. As its weights move to map `embed[cue] -> answer` directly, they de-calibrate how the network reads `embed[cue] + w * head[answer]` from the overlay, even for brand-new pairs that were never consolidated into the slow weights.
- **H1b (double-supply interference).** For a pair that has already been consolidated into the slow weights, leaving that pair in the overlay supplies the answer twice (slow weights plus `w * head[answer]`), pushing the signal into the high-weight rollover regime seen in Stage 2C's calibration (`w = 4` gave 0.763; `w = 16` fell to 0.643).

To separate H1a from H1b, we quiz the student (with memory on) after each consolidation phase on a **held-out 5th list** (`unseen_list`, 128 pairs drawn from the remaining words of `word_pool` not used in Lists 1–4). Because `unseen_list` is never trained into the slow weights, double-supply cannot happen on it; any loss of recall on `unseen_list` is pure overlay-channel drift.

To repair the fault, we test adding a **memory-on consistency term** (`memory_kl_weight = 1.0`): at each consolidation step, in addition to `cue_kl` with `overlay=None`, the student also minimises `cue_kl(_forward(model, reader, overlay, x, device), t_logits, y)` with weight 1.0. We test this both with the clean teacher (`original_teacher_memory_consistency`) and with the rolling teacher (`rolling_teacher_memory_consistency`, testing whether keeping the student's overlay read calibrated is enough to keep the rolling teacher from collapsing across phases).

### Part 4b: Can a count-table context gate stop 1-token overlay writes from wrecking ordinary English?

In Stage 2C (`notes/stage2c_result_2026-11-04.md`), the `write` arm recalled 0.760 of the 4 lists with 0.000 forgetting, but leaving the 1,536 written cues active raised validation language loss from 3.362 to 4.175 (+0.813 nats in the mean: seed 0 +0.763, seed 1 +0.891, seed 2 +0.785). That happens because `Overlay.slot` fires whenever a cue word appears in ordinary English prose.

In a vocabulary-list quiz (`[eot, cue_0, ans_0, cue_1, ans_1, ...]`), a cue sits either right after `eot` (position 1) or right after a random pair `(cue_{j-1}, ans_{j-1})`, so the multi-token history ending at the cue is almost never stored in the count tables (`reader.history_counts(x, k) <= 0`). In ordinary English prose (`val.bin`), 85.6% of positions have a stored 3-token history ($k = 3$, `order4`, $c_{\min} = 2$) and 99.4% have a stored 2-token history ($k = 2$, `order3`, $c_{\min} = 1$; `notes/count_gate_design_2026-11-04.md`).

`GatedOverlay` suppresses the 1-token overlay slot at any position where `reader.history_counts(x, gate_k) > max_hcount`. We test whether this cuts the language-loss penalty while keeping list recall intact.

## 2. Design

Same model (`runs2/count-M-0`), same 4 lists of 384 pairs (`make_lists(pool, 4, 384, seed)`), same task seeds (0, 1, 2), same write weight (4.0, from `results/stage2c_lists.json`), and same validation windows (`lm_windows` and `anchor_windows`) as Stages 2C and 2C-2. Nothing is tuned. Every figure is a mean over the three task seeds unless stated.

### Part 4a variants (consolidation, 300 steps per list, lr 3e-4, anchor weight 1.0)

Each variant also records `unseen_start` (the untouched model's memory-on recall of the held-out 128-pair `unseen_list`) and `unseen_with_memory` (the student's memory-on recall of `unseen_list` after each phase 1 to 4).

1. **original_teacher.** `teacher_mode="original"`, `freeze_head=False`, `memory_kl_weight=0.0`. Reproduces Stage 2C-2's `original_teacher` bit-for-bit on Lists 1–4 and adds the `unseen_list` diagnostic.
2. **original_teacher_memory_consistency.** `teacher_mode="original"`, `freeze_head=False`, `memory_kl_weight=1.0`. Trains the student to match the clean teacher both with `overlay=None` and with `overlay` active.
3. **rolling_teacher_memory_consistency.** `teacher_mode="rolling"`, `freeze_head=False`, `memory_kl_weight=1.0`. Tests whether `memory_kl_weight=1.0` rescues the rolling teacher without keeping a separate original teacher copy for cue distillation.

### Part 4b variants (gradient-free writes, no training)

1. **write_gated_k3_c0.** `gate_k=3`, `max_hcount=0` (suppresses the overlay when the 3-token history ending at the token is stored in `order4`, i.e. count $\ge 2$).
2. **write_gated_k2_c0.** `gate_k=2`, `max_hcount=0` (suppresses the overlay when the 2-token history ending at the token is stored in `order3`, i.e. count $\ge 1$).
3. **write_gated_k2_c9.** `gate_k=2`, `max_hcount=9` (suppresses the overlay when the 2-token history ending at the token has count $\ge 10$, outside the rare `1-9` bin).

Comparison numbers from `results/stage2c_lists.json` (not rerun): ungated `write` `final_mean` 0.760 (by seed 0.759, 0.743, 0.779), `lm_loss_memory_off` 3.362, `lm_loss_memory_on` after list 4: 4.125, 4.253, 4.147 (mean 4.175, mean rise +0.813 nats); registered `consolidate` `final_mean` 0.061; `gradient_count` `backward_transfer` -0.651.

## 3. Decision rules

All thresholds are fixed before any run.

### Part 4a rules

- **H1a (overlay-channel drift in `original_teacher`) is supported** iff `original_teacher`'s final unseen-list recall `unseen_with_memory[-1]` is at least 0.10 below `unseen_start`, in the mean and in each of the three task seeds.
- **H1b (double-supply interference in `original_teacher`) is supported** iff `original_teacher`'s final unseen-list recall `unseen_with_memory[-1]` exceeds its memory-on `final_mean` on Lists 1–4 (`with_memory.final_mean`, 0.413) by at least 0.10, in the mean and in each seed. (H1a and H1b are not mutually exclusive; both can hold.)
- **Memory-off consolidation works** is scored by the three conditions of `PREREG_stage2c.md` section 6: (a) memory-off `final_mean` $\ge 0.710$, (b) memory-off `backward_transfer` $\ge -0.351$, (c) `lm_loss[-1] - lm_loss_start` $\le 0.05$.
- **Repair 4a (`original_teacher_memory_consistency`) works** iff all four hold:
  1. it passes the three memory-off consolidation conditions above;
  2. its memory-on `final_mean` (`with_memory.final_mean`) is at least 0.710 in the mean and exceeds `original_teacher`'s memory-on `final_mean` (0.413) by at least 0.10 in the mean and in each seed;
  3. its memory-on `backward_transfer` (`with_memory.backward_transfer`) is within $\pm 0.05$ of zero in the mean;
  4. its final unseen-list recall `unseen_with_memory[-1]` is within 0.05 of `unseen_start` in the mean.
- **Rolling teacher rescue (`rolling_teacher_memory_consistency`) is supported** iff its memory-off `final_mean` exceeds the registered consolidate arm's (0.061) by at least 0.10 in the mean and in each seed; it **passes consolidation** iff it also meets the three memory-off consolidation conditions.

### Part 4b rules

- A context-gated write variant **works** iff all three hold:
  1. **Recall preserved:** its `final_mean` is at least the ungated `write` arm's `final_mean` minus 0.05 in the mean ($\ge 0.710$) and in every seed;
  2. **Zero forgetting:** its `backward_transfer` is within $\pm 0.05$ of zero;
  3. **Language-loss penalty cut by at least half:** its final language-loss rise `lm_loss_memory_on[-1] - lm_loss_memory_off` is at most half of the ungated `write` arm's rise in the mean ($\le 0.407$ nats) and in every seed.
- A variant that works is further scored as **low-interference** iff its final language-loss rise after all 4 lists is at most 0.05 nats (the same 0.05-nat language-loss bar applied to consolidation in `PREREG_stage2c.md` section 6).

## 4. Conduct

Runs go under `systemd-run --user` on the laptop RTX 3050 (4 GB) with `--mem-fraction 0.75`. `embeddinggemma.service` is stopped during the GPU run and restarted and checked afterwards. The driver writes `results/stage2c3.json`.

Tests in `tests/test_consolidate_fix.py` and planted-bug mutants M127 to M133 in `scripts/mutation_audit.py` cover `GatedOverlay`, `run_write_gated`, `memory_kl_weight`, and `unseen_list`. Both the test suite and `scripts/mutation_audit.py` must pass before the file is committed and the job is launched. No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash.

## 5. Limits stated in advance

- Single-token words (`word_pool`) and one model checkpoint (`count-M-0`), as in Stages 2C and 2C-2.
- `memory_kl_weight = 1.0` adds one extra student forward pass (`_forward(model, reader, overlay, x, device)`) per consolidation step, costing about 25% more time per step.
- The context gate uses the exact n-gram count tables of the model (`data/count`). In a quiz sequence `[eot, cue_0, ans_0, cue_1, ans_1, ...]`, consecutive pairs are packed without spaces or grammar; if a random `(ans_{j-1}, cue_j)` bigram happened to occur in the 881M-token corpus, `write_gated_k2_c0` will suppress `cue_j` at that position, whereas `write_gated_k3_c0` and `write_gated_k2_c9` are much less likely to suppress it. Measuring all three shows the trade-off between quiz recall and ordinary-text protection.
