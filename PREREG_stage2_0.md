# Stage 2.0 pre-registration: the baseline B0 and its size ladder

Committed before the first Stage 2.0 run. Its sha256 is recorded in every Stage 2.0 `results.json`. Plan: `docs/superpowers/plans/2026-11-02-stage2-kenyon-memory.md` (§7.1, §11).

## 1. Purpose

Stage 2.0 tests no hypothesis. It fixes the baseline that every Stage 2 memory arm must beat, measures seed noise (which sets the Stage 2 SESOI), and builds the three-size ladder used to express memory gains as a compute multiplier.

## 2. Fixed design

- Model: `src/clm/s2model.py` (B0: RMSNorm without gain, RoPE, QK-norm, ReLU² MLP x4, zero-initialised output projections and head, no biases, untied head, value embeddings on the first and last layer, logit softcap 15). Sizes in `configs/stage2_0.json`: S (d 256, 6 layers, 4 heads), M (d 384, 6, 6), L (d 512, 8, 8); context 512; the Stage 1 tokenizer (8,192 tokens).
- Training: `src/clm/s2train.py`; 200,000,000 tokens (12,207 steps of 32 x 512), the Stage 1 data order (`clm.data.batch_starts`, seeded), warmup 200 steps then linear decay to zero, weight decay 0.1 on block matrices, gradient clip 1.0, bf16 autocast, GPU memory cap 0.75, laptop RTX 3050.
- Metric: mean cross-entropy (nats/token) on the first 2,097,152 tokens of `val.bin` at the last step; training wall-clock (sum of step times, evaluation excluded); peak VRAM.

## 3. Pilot (B0-M, seed 0, 50,000,000 tokens with the schedule compressed to that length)

Grid: AdamW at learning rates 1e-3, 2e-3, 4e-3; Muon (torch.optim.Muon, `adjust_lr_fn="match_rms_adamw"`, AdamW for embeddings, head and gates) at 2e-3, 4e-3, 8e-3. **Rule:** the lowest final validation loss wins. If the winner sits on its grid's edge, one further run at half (lower edge) or double (upper edge) that rate is added once, and the rule is applied again. The choice is written to `runs2/pilot/choice.json`; it applies unchanged to every size and seed. Implemented in `scripts/stage2_0_driver.py`.

## 4. Ladder

Sizes M, S, L x seeds 0, 1, 2 with the chosen optimizer and rate, 200M tokens each, in that order.

## 5. What is derived, and how

- **SESOI for Stage 2** = max(0.01, 2 x the standard deviation of the three B0-M final losses), in nats/token. Effects below 0.02 nats get 5 seeds in later stages.
- **Ladder fit:** final loss against non-embedding parameters, a power law fitted on the three size means; used only to convert later loss gains into an equivalent size and compute multiplier.

## 6. Conduct

In the pilot a diverged rate counts as a failed rate. A diverged ladder run (non-finite loss or gradient) is rerun once from scratch with the same seed; a second divergence is reported and the size is fitted without it. No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash. Pilots touch only B0. Interrupted runs resume from their checkpoint; that does not change results (tested: `tests/test_s2.py::test_interrupted_and_resumed_run_ends_bit_identical`).

## 7. Limits stated in advance

The pilot compresses the schedule to 50M tokens, so the chosen rate is a short-horizon proxy. One learning rate serves all sizes. Laptop wall-clock varies with background load; the headline comparisons in later stages run under the same overnight conditions.
