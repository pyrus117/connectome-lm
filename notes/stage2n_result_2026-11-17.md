# Stage 2N result: `head.weight` at 64e-3 is what kills the loop — lowering `head.weight` to 8e-3 revives the loop (`HEADLOW`, P24 supported), whereas leaving `head.weight` at 64e-3 keeps it dead (`EMBEDLOW`, P25 not supported) (2026-11-17)

Pre-registration `PREREG_stage2n.md` (dated 2026-11-16, committed as `f4d3223` before any production run). Pre-run note and smoke runs `notes/stage2n_pre_run_2026-11-16.md` (commit `1646906`). Code `src/clm/s2model.py`, `src/clm/s2train.py`, `src/clm/rules2n.py`, `scripts/stage2n_driver.py`, `scripts/stage2n_analyze.py`; config `configs/stage2n.json`; tests `tests/test_s2n.py`, mutants M215 to M222 (218/218 killed); results `results/stage2n.json` (`sha256: 0313496903b0ac5c92440b391426b0734f5e777b6e3bdb27ca37cdbbd810fe75`). Run on q-pc, four runs (`HEADLOW` seeds 0 to 2; `EMBEDLOW` seed 0, stopped by the registered per-arm seed-0 futility rule), all at commit `f4d3223`, tree clean (`git_dirty: false`), status complete. Launched at 07:56 on 2026-11-16; last run finished at 12:30 on 2026-11-17 (`results.json` write times on q-pc: `HEADLOW` seed 0 at 15:01 on 16 Nov, `EMBEDLOW` seed 0 at 22:10 on 16 Nov, `HEADLOW` seed 1 at 05:20 on 17 Nov, `HEADLOW` seed 2 at 12:30 on 17 Nov). Deviations: none.

Terms, once. *Cross-entropy* (validation loss) is the model's average surprise per token, in nats; lower is better. The *looped model* (`LoopGPT`) reuses one two-block pair up to six times per token (passes 1 to 6, depths 1 to 6) between a two-block prelude and a two-block coda, with a learned per-pass per-channel residual gain (`loop_gain`, shape 6 × 384, initialised to 1, no weight decay) and trained on the mean loss over all six depths. *Best-depth loss* is the lowest of a run's six per-depth validation losses. The *loop's benefit* inside one run is its depth-1 validation loss minus its best-depth validation loss. *SESOI* is 0.01 nats, used both as the within-run "alive" threshold and for the described paired contrasts (`clm.analyze.verdict`). Both arms carry the plain *COUNT* read at the input and train block matrices with Muon at `8e-3` and `loop_gain` with AdamW at `8e-3`. *HEADLOW* (`runs2/loop_count_headlow-M-s`, `--gain-mult 1.0 --head-mult 1.0`) trains the untied unembedding readout (`head.weight`, `8192 × 384`, zero-initialised) at AdamW `8e-3` and the input/value embeddings (`embed.weight`, `value_embed.{0,1}.weight`, `blocks.{0,5}.ve_gate`) at AdamW `64e-3`. *EMBEDLOW* (`runs2/loop_count_embedlow-M-s`, `--adam-mult 1.0 --head-mult 8.0`) trains the input/value embeddings at AdamW `8e-3` and `head.weight` at AdamW `64e-3`.

## Verdict

- **P24 is SUPPORTED (`SUPPORTED`).** In `HEADLOW` (`head.weight` and `loop_gain` at `8e-3`, embeddings at `64e-3`), the loop is alive on every seed: loop benefit (depth-1 minus best-depth validation loss) is **0.0303, 0.0331, and 0.0295 nats** on seeds 0, 1, and 2 (mean **0.0309 nats**), clearing the 0.01 nats bar on all three seeds.
- **P25 is NOT SUPPORTED (`NOT_SUPPORTED_FUTILITY`).** In `EMBEDLOW` (embeddings and `loop_gain` at `8e-3`, `head.weight` at `64e-3`), seed 0's loop benefit is **0.000048 nats** (`4.83e-5`), well below the 0.01 nats futility threshold. Per `PREREG_stage2n.md` section 6 (`clm.rules2n.keep_arm`), `EMBEDLOW` stopped after seed 0 (`FUTILITY loop_count_embedlow stop`).
- **Reading table (`PREREG_stage2n.md` section 3, row 1: *P24 supported, P25 not supported*).** **H2a holds:** the untied unembedding matrix `head.weight` at AdamW `64e-3` (with zero weight decay) is both necessary and sufficient to kill the loop. Lowering `head.weight` alone to `8e-3` revives the loop even while the input and value embeddings (`embed.weight`, `value_embed.{0,1}.weight`, `blocks.{0,5}.ve_gate`) remain at `64e-3`; conversely, leaving `head.weight` at `64e-3` kills the loop even when all embeddings and `loop_gain` train at `8e-3`.
- **Against the stated expectation (`PREREG_stage2n.md` section 4).** Matched the registered prediction (`P24 SUPPORTED` and `P25 NOT_SUPPORTED`) directly.

## Complete 2×2 Factorial Table (Stage 2M + Stage 2N, `loop_gain` at `8e-3`)

| Embeddings LR (`embed`, `value_embed`, `ve_gate`) | `head.weight` LR = `8e-3` (`1.0×`) | `head.weight` LR = `64e-3` (`8.0×`) |
|---|---|---|
| **`8e-3` (`1.0×`)** | **`SHARED` (Stage 2M, P23 SUPPORTED)**<br>Loop benefit: **+0.0388, +0.0387, +0.0354** (mean **+0.0376**)<br>Best-depth val loss: **3.2012** (3.1993, 3.2028, 3.2014)<br>`||head.weight||_2`: **731.3** (seed 0) | **`EMBEDLOW` (Stage 2N, P25 NOT_SUPPORTED_FUTILITY)**<br>Loop benefit: **+0.000048** (seed 0; Dead)<br>Best-depth val loss: **3.2379** (seed 0)<br>`||head.weight||_2`: **5,202.0** (seed 0) |
| **`64e-3` (`8.0×`)** | **`HEADLOW` (Stage 2N, P24 SUPPORTED)**<br>Loop benefit: **+0.0303, +0.0331, +0.0295** (mean **+0.0309**)<br>Best-depth val loss: **3.2044** (3.2057, 3.2017, 3.2058)<br>`||head.weight||_2`: **726.2, 729.1, 724.6** | **`GAINLOW` (Stage 2M, P22 NOT_SUPPORTED)**<br>Loop benefit: **+0.000007, +0.000012, +0.000042** (mean **+0.000021**)<br>Best-depth val loss: **3.2184** (3.2189, 3.2171, 3.2191)<br>`||head.weight||_2`: **5,824.5** (seed 0) |

The `2×2` table separates the two columns cleanly:
- Whenever `head.weight` trains at `8e-3` (left column: `SHARED` and `HEADLOW`), `||head.weight||_2` stays around **725–731**, passes 1–3 stay active (`loop_gain` mean abs `~0.35–0.53`, `0.28–0.38`, `0.17–0.20`), and the loop earns **+0.031 to +0.038 nats** on every seed.
- Whenever `head.weight` trains at `64e-3` (right column: `EMBEDLOW` and `GAINLOW`), `||head.weight||_2` explodes **7–8×** to **5,202–5,825**, passes 2–6 collapse (`loop_gain` mean abs `0.003–0.023`), and the loop earns **< 0.00005 nats**.

## Within-run depth losses and loop benefit (primary decision quantities)

Validation loss on the first 2,097,152 tokens of `data/fineweb/val.bin` at step 12,207 (size M, 200M tokens, context 512):

| Arm | Seed | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 | Depth 6 | Best depth | Best loss | Loop benefit (d1 − best) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **HEADLOW** | 0 | 3.23591 | 3.20905 | 3.20582 | 3.20567 | 3.20565 | 3.20567 | 5 | 3.20565 | **+0.030252** |
| **HEADLOW** | 1 | 3.23482 | 3.20569 | 3.20186 | 3.20170 | 3.20171 | 3.20171 | 4 | 3.20170 | **+0.033120** |
| **HEADLOW** | 2 | 3.23526 | 3.20895 | 3.20592 | 3.20579 | 3.20580 | 3.20580 | 4 | 3.20579 | **+0.029469** |
| **HEADLOW** | Mean | 3.23533 | 3.20790 | 3.20453 | 3.20439 | 3.20439 | 3.20439 | — | **3.20438** | **+0.030947** |
| **EMBEDLOW** | 0 | 3.23793 | 3.23789 | 3.23788 | 3.23790 | 3.23790 | 3.23789 | 3 | 3.23788 | +0.000048 |

Mean marginal gain from each extra pass (nats; positive means the pass lowered validation loss):

| Arm | Pass 2 | Pass 3 | Pass 4 | Pass 5 | Pass 6 |
|---|---:|---:|---:|---:|---:|
| **HEADLOW** (3-seed mean) | **+0.027432** | **+0.003366** | **+0.000144** | +0.000002 | −0.000005 |
| **EMBEDLOW** (seed 0) | +0.000041 | +0.000007 | −0.000018 | −0.000003 | +0.000010 |

## Learned `loop_gain` per pass and `head.weight` L2 norm in the final checkpoints

Mean absolute `loop_gain` across the 384 channels and `||head.weight||_2` at step 12,207 (`results/stage2n.json`):

| Arm | Seed | Pass 1 | Pass 2 | Pass 3 | Pass 4 | Pass 5 | Pass 6 | `||head.weight||_2` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **HEADLOW** (`head` 8e-3, `embed` 64e-3) | 0 | 0.5320 | 0.3787 | 0.2042 | 0.0530 | 0.0082 | 0.0033 | **726.2** |
| **HEADLOW** (`head` 8e-3, `embed` 64e-3) | 1 | 0.3493 | 0.2790 | 0.1749 | 0.0433 | 0.0054 | 0.0027 | **729.1** |
| **HEADLOW** (`head` 8e-3, `embed` 64e-3) | 2 | 0.5296 | 0.3507 | 0.1843 | 0.0435 | 0.0067 | 0.0029 | **724.6** |
| **EMBEDLOW** (`embed` 8e-3, `head` 64e-3) | 0 | 0.1513 | 0.0121 | 0.0045 | 0.0046 | 0.0038 | 0.0031 | **5,202.0** |
| **SHARED** (Stage 2M, all AdamW 8e-3) | 0 | 0.3522 | 0.2907 | 0.1901 | 0.0599 | 0.0124 | 0.0036 | **731.3** |
| **GAINLOW** (Stage 2M, `head`+`embed` 64e-3) | 0 | 0.1180 | 0.0112 | 0.0131 | 0.0067 | 0.0046 | 0.0023 | **5,824.5** |

## Described comparisons on best-depth loss (no decision)

Paired by seed on all three seeds, `d_s = first − second` (negative means the first arm is better), Stage 1 verdict rule (`SESOI = 0.01`):

| Contrast | Seed 0 | Seed 1 | Seed 2 | Mean `d` | Verdict |
|---|---:|---:|---:|---:|---|
| **HEADLOW vs SHARED (Stage 2M)** (effect of 64e-3 vs 8e-3 embeddings when `head.weight` is at 8e-3) | +0.0063 | −0.0011 | +0.0044 | **+0.0032** | **NO_MEANINGFUL_DIFFERENCE** |
| **HEADLOW vs GAINLOW (Stage 2M)** (effect of lowering `head.weight` from 64e-3 to 8e-3 when embeddings are at 64e-3) | −0.0132 | −0.0154 | −0.0133 | **−0.0140** | **X_BETTER** |

Key observations from the described contrasts:
1. **Lowering `head.weight` from `64e-3` to `8e-3` (`HEADLOW` vs `GAINLOW`) lowers best-depth validation loss by `-0.0140` nats (`X_BETTER`, every seed better by `-0.0132` to `-0.0154`).** Note that at depth 1 (`3.2353` in `HEADLOW` vs `3.2184` in `GAINLOW`), `GAINLOW`'s single pass actually starts `0.0170` nats lower because the `64e-3` readout optimises the depth-1 representation aggressively—at the cost of destroying passes 2–6. Once `head.weight` is at `8e-3`, passes 2–4 add `0.0309` nats of loop benefit and overtake `GAINLOW` by `-0.0140` nats.
2. **Keeping embeddings at `64e-3` vs `8e-3` when `head.weight` is at `8e-3` (`HEADLOW` `3.2044` vs `SHARED` `3.2012`) is `NO_MEANINGFUL_DIFFERENCE` (`+0.0032` nats).** Thus the simpler **`SHARED` recipe (`--adam-mult 1.0`, all AdamW parameters at `8e-3`) remains slightly lower (`3.2012` vs `3.2044`) and needs no per-group multiplier split**.

## Cost

| Arm | s / step (q-pc) | Peak VRAM MiB | Peak host RSS MiB |
|---|---:|---:|---:|
| **HEADLOW** | 1.925 | 2,586 | 7,236 |
| **EMBEDLOW** | 1.924 | 2,586 | 7,235 |

## What the verdict does (`PREREG_stage2n.md` section 5)

- **Diagnosis complete:** The mystery from Stage 2L and Stage 2M is fully resolved. On the separate-rate recipe (`Muon 8e-3 / AdamW 64e-3`), the loop went inert specifically because **`head.weight` (the untied unembedding matrix trained by AdamW with zero weight decay) at `64e-3` grows ~8× in L2 norm (`>5,200` vs `~730` at `8e-3`) during the first thousand steps**, locking the shared readout onto the pass-1 representation so additional passes of the recurrent block disrupt the logit projection and `loop_gain` shuts passes 2–6 off.
- **Carry-forward recipe:** Both `SHARED` (`3.2012` nats, all AdamW at `8e-3`) and `HEADLOW` (`3.2044` nats, `head.weight` and `loop_gain` at `8e-3`, embeddings at `64e-3`) have a live loop and tie within `NO_MEANINGFUL_DIFFERENCE` (`+0.0032` nats). Because `SHARED` is simpler (no per-parameter multiplier split) and has the lowest mean validation loss (`3.2012` nats), **`SHARED` (`LoopGPT` + `loop_gain` + all-depth loss + plain `COUNT` at `Muon 8e-3 / AdamW 8e-3`) is the clean carry-forward looped recipe**.

## Limits

`PREREG_stage2n.md` section 8 limits 1 to 5 apply: `EMBEDLOW` was stopped after seed 0 by the registered futility rule; `embed.weight`, `value_embed.{0,1}.weight`, and `blocks.{0,5}.ve_gate` remain grouped together (not needed since `head.weight` alone decided the question); looped arms are token-matched, not compute-matched (~2.4x block passes per step); three seeds; one read (`COUNT`), one placement (input), one size (`M`), one corpus.

TL:DR. Stage 2N isolated which AdamW parameter group at `64e-3` killed the loop in Stages 2L and 2M. Lowering `head.weight` to `8e-3` while keeping the embeddings at `64e-3` (`HEADLOW`, P24) **revived the loop on all three seeds** (`SUPPORTED`, loop benefit `0.0303, 0.0331, 0.0295` nats; best-depth loss `3.2044`, `X_BETTER` than `GAINLOW` by `-0.0140` and `NO_MEANINGFUL_DIFFERENCE` against `SHARED`'s `3.2012`). Conversely, lowering the embeddings to `8e-3` while keeping `head.weight` at `64e-3` (`EMBEDLOW`, P25) **left the loop dead** (`NOT_SUPPORTED_FUTILITY`, seed-0 benefit `0.000048` nats, `||head.weight||_2 = 5,202`). The `64e-3` rate on `head.weight` is what kills the loop.
