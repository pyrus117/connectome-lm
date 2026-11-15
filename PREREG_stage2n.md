# Stage 2N pre-registration: which 64e-3 AdamW parameter group kills the loop?

Dated 2026-11-16. Committed before any Stage 2N run (the one smoke run in section 6 excepted). Origin: Nate's request on 2026-11-16 ("Go ahead and find the 64e-3 that kills the loop"), following Stage 2M (`notes/stage2m_result_2026-11-16.md`). Stage 2M refuted H1 (`GAINLOW`, putting only `loop_gain` at AdamW 8e-3 left the loop inert, benefit 0.00001 to 0.00004 nats) and supported P23 (`SHARED`, putting all non-block parameters at AdamW 8e-3 revived the loop with COUNT in place, benefit 0.0388, 0.0387, 0.0354 nats, mean best-depth loss 3.2012). That localized the loop killer to the remaining AdamW-trained parameters (`rest` in `s2model.param_groups`): `head.weight` versus the input/value embeddings (`embed.weight`, `value_embed.{0,1}.weight`, `blocks.{0,5}.ve_gate`). Code: `src/clm/s2model.py` (`param_groups`, new `head_mult`), `src/clm/s2train.py` (new `head_mult` setting and `--head-mult` flag), `src/clm/rules2n.py`, `scripts/stage2n_driver.py`, `scripts/stage2n_analyze.py`. Config `configs/stage2n.json`. Tests `tests/test_s2n.py`. Results `results/stage2n.json`.

## 1. Question

Terms are as in `PREREG_stage2m.md`: the *loop's benefit* inside one run is its depth-1 validation loss minus its best-depth validation loss (nats/token, lower loss is better), and a loop is *alive* on an arm if its benefit is at least the SESOI (`0.01` nats) on every one of three seeds (`0, 1, 2`).

**Background and post-hoc observation (read on 2026-11-16 from the Stage 2H, 2L, and 2M seed-0 checkpoints before any Stage 2N run; evidence for the split below, not a test).**

In `LoopGPT` (`src/clm/s2loop.py`), once `loop_gain` is placed in its own AdamW group at 8e-3 (`gain_mult = 1.0`), the remaining AdamW-trained parameters (`rest`, no weight decay) partition into two functional groups:
1. **The untied output head (`head.weight`, shape 8192 x 384):** projects the normalized post-coda state `rms(x)` into logits before the `15.0 * tanh(logits / 15.0)` softcap at every depth readout (`LoopGPT._read`), and also serves as the vector table read by `COUNT` at the input (`B0GPT._input`).
2. **The embeddings (`embed.weight`, `value_embed.0.weight`, `value_embed.1.weight`, and their two scalar gates `blocks.0.ve_gate`, `blocks.5.ve_gate`):** enter at the input and at the first/last attention layers.

Inspecting the final checkpoints on q-pc shows that training `rest` at AdamW 64e-3 (`adam_mult = 8.0`, no weight decay) causes `head.weight`'s L2 norm to grow 8x larger than in the live shared-rate loops:

| Checkpoint | Recipe | Loop status | `head.weight` L2 norm (mean\|w\|) | `embed.weight` L2 norm | `value_embed.1` L2 norm | `blocks.5.ve_gate` |
|---|---|---|---:|---:|---:|---:|
| Stage 2H `loop_gain_all-M-1` | Shared 8e-3 (no COUNT) | **LIVE** (0.0651) | **450.8** (0.198) | 1,970.4 | 2,072.2 | 16.50 |
| Stage 2M `loop_count_shared-M-0` | Shared 8e-3 (+ COUNT) | **LIVE** (0.0388) | **731.3** (0.318) | 1,643.2 | 2,046.7 | 17.23 |
| Stage 2L `loop_gain_all_sep-M-0` | Separate 64e-3 (no COUNT) | **DEAD** (0.00003) | **3,418.3** (1.485) | 6,856.4 | 7,732.6 | 2.82 |
| Stage 2L `loop_gain_all_count_sep-M-0` | Separate 64e-3 (+ COUNT) | **DEAD** (0.00003) | **6,147.7** (2.689) | 3,989.4 | 6,914.0 | 3.03 |
| Stage 2M `loop_count_gainlow-M-0` | `loop_gain` 8e-3, rest 64e-3 (+ COUNT) | **DEAD** (0.00001) | **5,824.5** (2.564) | 3,958.2 | 7,090.9 | 1.82 |

In `loop_count_gainlow-M-0`, the per-depth training loss actually had a +0.0151 nat loop benefit at step 200 (the end of warmup), which then collapsed to +0.0030 by step 400 and +0.0000 by step 6000 as `head.weight` and the embeddings grew at 64e-3.

Stage 2N completes the **2x2 factorial** over (`head.weight` rate in {8e-3, 64e-3}) x (embeddings rate in {8e-3, 64e-3}), with Muon at 8e-3 on block matrices and AdamW at 8e-3 on `loop_gain` in all four cells:

| Cell | `head.weight` AdamW rate | Embeddings (`embed`, `value_embed`, `ve_gate`) AdamW rate | Stage & Status |
|---|---|---|---|
| **SHARED** | 8e-3 | 8e-3 | Stage 2M (done): **LIVE** (benefit 0.0376) |
| **HEADLOW** | **8e-3** | **64e-3** | **Stage 2N arm 1 (P24)** |
| **EMBEDLOW** | **64e-3** | **8e-3** | **Stage 2N arm 2 (P25)** |
| **GAINLOW** | 64e-3 | 64e-3 | Stage 2M (done): **DEAD** (benefit 0.00002) |

**P24.** Is the loop's benefit at least 0.01 nats on every one of three seeds when `head.weight` and `loop_gain` train at AdamW 8e-3 while the embeddings train at 64e-3 (`HEADLOW`)?

**P25.** Is the loop's benefit at least 0.01 nats on every one of three seeds when the embeddings and `loop_gain` train at AdamW 8e-3 while `head.weight` alone trains at 64e-3 (`EMBEDLOW`)?

## 2. Fixed design

- **Recipe for both arms.** Identical to Stage 2M (`configs/stage2n.json`): Size M (13.8M non-embedding parameters), the gained all-depth looped model with the plain COUNT read at the input, 200M tokens, context 512, 12,207 steps of 32 x 512, micro-batch 4, Muon 8e-3 (`wd=0.1`) on 2-D block matrices, warmup 200 steps then linear decay, clip 1.0, bf16, GPU memory cap 0.75, seeds 0, 1, 2.
- **Arms** (new runs on q-pc):
  - **HEADLOW**: `runs2/loop_count_headlow-M-s`. `--gain-mult 1.0 --head-mult 1.0` (with `adam_mult=8.0` from config). `head.weight` and `loop_gain` train at AdamW 8e-3; `embed.weight`, `value_embed.{0,1}.weight`, and `blocks.{0,5}.ve_gate` train at AdamW 64e-3.
  - **EMBEDLOW**: `runs2/loop_count_embedlow-M-s`. `--adam-mult 1.0 --head-mult 8.0`. `embed.weight`, `value_embed.{0,1}.weight`, `blocks.{0,5}.ve_gate`, and `loop_gain` train at AdamW 8e-3; only `head.weight` trains at AdamW 64e-3.
- **Metric.** Validation loss at each depth 1 to 6 on the first 2,097,152 tokens of `data/fineweb/val.bin`, at the last step (`clm.rules2h.summary`).

## 3. Decision rule

Rules in `clm.rules2n` (`benefit`, `alive`, `keep_arm`, `status`), SESOI = 0.01 nats.

- **P24** is supported if and only if `HEADLOW` runs all three seeds and its benefit is at least 0.01 on every seed.
- **P25** is supported if and only if `EMBEDLOW` runs all three seeds and its benefit is at least 0.01 on every seed.
- **Per-arm futility stop after seed 0.** Seed 0 of both `HEADLOW` and `EMBEDLOW` always runs (~13 hours). Because `alive` requires all three seeds to reach 0.01, any arm whose seed-0 benefit is below 0.01 is already falsified on seed 0 (`keep_arm(s0) == False`); that arm does not run seeds 1 and 2 and is reported `NOT_SUPPORTED_FUTILITY`. Any arm whose seed-0 benefit is at least 0.01 continues to seeds 1 and 2 (`keep_arm(s0) == True`). The driver prints only `FUTILITY <arm> continue` or `FUTILITY <arm> stop`.
- **2x2 Reading table.**
  - *P24 supported, P25 not supported:* **`head.weight` at 64e-3 is both necessary and sufficient to kill the loop.** Whenever `head.weight` is at 8e-3 (`SHARED`, `HEADLOW`), the loop is alive; whenever `head.weight` is at 64e-3 (`EMBEDLOW`, `GAINLOW`), the loop is dead.
  - *P24 not supported, P25 supported:* **The embeddings (`embed`, `value_embed`, `ve_gate`) at 64e-3 are both necessary and sufficient to kill the loop.**
  - *Both P24 and P25 supported:* **Interaction only.** Either group alone at 64e-3 leaves the loop alive; the loop dies only when both `head.weight` and the embeddings are driven at 64e-3 together (`GAINLOW`).
  - *Neither P24 nor P25 supported:* **Either group alone at 64e-3 is sufficient to kill the loop.** Both `head.weight` and the embeddings must be at 8e-3 (`SHARED`) for the loop to live.
- **Described, no decision.** For each arm that completes all three seeds, Stage 1 verdict rule (`clm.analyze.verdict`, SESOI 0.01, paired by seed) on best-depth loss against Stage 2M `SHARED` and Stage 2M `GAINLOW` (and `HEADLOW` vs `EMBEDLOW` if both complete three seeds), plus mean marginal gain by pass, mean absolute `loop_gain` per pass, `head.weight` L2 norm in each final checkpoint, and cost.

## 4. Stated expectation, before any run

`head.weight` sits directly before the `15.0 * tanh(logits / 15.0)` softcap at every depth readout (`LoopGPT._read`) and also supplies the COUNT vectors at the input; its L2 norm grew 8x larger at 64e-3 (`5,824` vs `731`). We therefore expect `head.weight` at 64e-3 to be the primary culprit: **P24 (`HEADLOW`) is expected to hold and P25 (`EMBEDLOW`) is expected not to hold** (stopping after seed 0). However, `embed` and `value_embed` also grew 2.4x–3.5x at 64e-3, so either the reverse or the "both must be at 8e-3" outcome is possible. The rule decides.

## 5. What each outcome does

- Identifies which 64e-3 AdamW parameter group (or both) kills the loop, answering Nate's diagnostic question.
- If `HEADLOW` (or `EMBEDLOW`) is alive and beats `SHARED` (`3.2012`) on best-depth loss by at least 0.01 on all three seeds (`X_BETTER`), it becomes a candidate loop+COUNT recipe alongside `SHARED`; otherwise `SHARED` remains the best loop+COUNT recipe.
- Every outcome is written up in `notes/stage2n_result_<date>.md`.

## 6. Procedure

All training and analysis on q-pc, detached, GPU memory cap 0.75, driver `scripts/stage2n_driver.py` (resumable). Order: seed 0 `HEADLOW`, seed 0 `EMBEDLOW`, per-arm futility check, then seeds 1 and 2 for each arm that continues, then `scripts/stage2n_analyze.py`.

Before the first run:
- The test suite and `scripts/mutation_audit.py` pass on the committed code (mutants M215 to M222 included).
- The branch is synced to q-pc and the q-pc tree is clean.
- A 50-step smoke check of both `HEADLOW` and `EMBEDLOW` on q-pc (`runs2/smoke_2n_*`, not read by the analysis) verifies that both train, that their optimizer parameter groups have the exact expected learning rates (`head.weight`, `loop_gain`, `embed`/`value_embed`/`ve_gate`, and `blocks`), and that their step-0 loss matches Stage 2M (`9.0109`). Validation loss is not read.

Time: ~6.5 hours per run on q-pc. Two seed-0 runs take ~13 hours; if one arm continues, 4 runs take ~26 hours; if both continue, 6 runs take ~39 hours. No money.

## 7. Conduct

No validation loss of a registered arm is read before the analysis (except by `stage2n_driver.py`'s mechanical seed-0 futility check). Status checks read only `driver_stage2n.log` and training fields (`step`, `loss`, `train_loss_by_depth`, `seconds`). A diverged run is rerun once from scratch; a second divergence is reported and gives no verdict. No change to code, config or data after this file is committed except crash fixes logged in `notes/deviations.md`. Planted-bug mutants M215 to M222 guard `head_mult` routing, learning rate application, zero weight decay, CLI/config wiring, and `rules2n` futility/status rules.

## 8. Limits stated in advance

1. **Embeddings and `ve_gate` are grouped together.** The 5 non-head, non-gain AdamW parameters (`embed.weight`, `value_embed.{0,1}.weight`, `blocks.{0,5}.ve_gate`) move together as one group; if they kill the loop, this stage does not split token embeddings from value embeddings.
2. **Per-arm futility stop.** An arm that fails on seed 0 stops at one seed (`NOT_SUPPORTED_FUTILITY`), which is sufficient to falsify its 3-seed `alive` claim (`all >= 0.01`).
3. **Equal tokens, not equal compute (~2.4x block passes per step); three seeds; one size (M); one corpus.**

TL:DR. Stage 2M showed that lowering `loop_gain`'s AdamW rate to 8e-3 did not revive the loop (`GAINLOW` dead), whereas lowering all AdamW parameters to 8e-3 did (`SHARED` alive, 3.2012 nats). Stage 2N completes the 2x2 factorial by splitting the remaining AdamW parameters into `head.weight` versus the embeddings (`embed`, `value_embed`, `ve_gate`), with `loop_gain` at 8e-3 in both: `HEADLOW` (`head.weight` at 8e-3, embeddings at 64e-3, P24) and `EMBEDLOW` (embeddings at 8e-3, `head.weight` at 64e-3, P25). Seed 0 of both runs first (~13 h); only arms with seed-0 benefit >= 0.01 run seeds 1 and 2 (~26 h if one continues, ~39 h if both).
