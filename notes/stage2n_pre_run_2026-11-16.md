# Stage 2N before the runs: post-hoc evidence and the smoke runs (2026-11-16)

Pre-registration `PREREG_stage2n.md`, committed as `f4d3223`. Neither measurement below is a test of a pre-registered claim.

## Post-hoc reading of Stage 2M, Stage 2L and Stage 2H seed-0 checkpoints (`q-pc`)

In `LoopGPT`, once `loop_gain` is placed at `8e-3` (`--gain-mult 1.0`), the remaining AdamW parameters (`rest`, zero weight decay) split into two functional families:
1. **Unembedding / readout (`head.weight`, shape `8192 × 384`, untied from `embed.weight`, zero-initialised)** — projects `norm(x)` to logits at every loop pass.
2. **Input & value embeddings (`embed.weight` `8192 × 384`, `value_embed.{0,1}.weight` `8192 × 384`, `blocks.{0,5}.ve_gate` `3 × 16`)** — supply token representations into the residual stream and into attention keys/values at blocks 0 and 5 on every loop pass.

L2 norms (`||W||_2`) and max absolute entries (`max |W|`) in the final checkpoint (step 12,207):

| Parameter | Stage 2H `loop_gain_all-M-1` (AdamW `8e-3`, loop alive) | Stage 2M `loop_count_shared-M-0` (AdamW `8e-3`, loop alive) | Stage 2L `loop_gain_all_sep-M-0` (AdamW `64e-3`, loop dead) | Stage 2M `loop_count_gainlow-M-0` (`loop_gain` `8e-3`, rest `64e-3`, loop dead) | Ratio (`gainlow` / `shared`) |
|---|---:|---:|---:|---:|---:|
| `embed.weight` | 485.8 (max 1.84) | 380.4 (max 1.98) | 924.8 (max 4.65) | 895.0 (max 4.61) | **2.35×** |
| `head.weight` | 450.8 (max 1.88) | 731.3 (max 2.13) | 3,418.3 (max 9.10) | 5,824.5 (max 11.42) | **7.96×** |
| `value_embed.0.weight` | 538.9 (max 1.97) | 328.9 (max 2.11) | 711.6 (max 4.30) | 456.8 (max 4.21) | **1.39×** |
| `value_embed.1.weight` | 280.1 (max 2.21) | 323.5 (max 2.19) | 608.0 (max 5.02) | 1,116.8 (max 5.47) | **3.45×** |
| `blocks.0.ve_gate` | 6.66 (max 2.09) | 5.21 (max 1.66) | 19.63 (max 5.56) | 21.84 (max 6.96) | **4.19×** |
| `blocks.5.ve_gate` | 4.86 (max 1.48) | 7.80 (max 2.58) | 22.86 (max 5.95) | 32.16 (max 7.90) | **4.12×** |

Training loss by depth across steps in `loop_count_gainlow-M-0` (`train_loss_by_depth` in `log.jsonl`):
- Step 200 (end of warmup): depth 1 = `4.8190`, best deeper pass = `4.8039` (**loop benefit `+0.0151` nats**).
- Step 400: depth 1 = `4.3469`, best deeper pass = `4.3439` (**`+0.0030` nats**).
- Step 1,000: depth 1 = `3.9511`, best deeper pass = `3.9498` (**`+0.0013` nats**).
- Step 6,000 to 12,206: depth 1 and all deeper passes agree to within `0.0001` nats (`0.0000` benefit).

The loop starts learning during warmup in `GAINLOW`, then collapses between steps 200 and 1,000 as the `64e-3` AdamW parameters reach peak learning rate and grow to large norms.

## Pre-registered 50-step smoke runs (seed 0, `q-pc`, commit `f4d3223`)

Both smoke runs used `--config configs/stage2n.json --size M --seed 0 --tokens 819200` (50 steps) and were deleted before `stage2n_driver.py` launched. No validation loss field was read.

1. **`HEADLOW` (`runs2/smoke_2n_headlow`, `--gain-mult 1.0 --head-mult 1.0`):**
   - 50 steps in 95.8 s (`1.916 s/step`).
   - Optimizer groups in `ckpt.pt`: Muon group 0 (`24` block matrices, `lr=0.002` at step 50); AdamW group 0 (`5` embedding params, `lr=0.016`, `lr_mult=8.0`, `wd=0.0`), group 1 (`1` param `loop_gain`, `lr=0.002`, `lr_mult=1.0`, `wd=0.0`), group 2 (`1` param `head.weight`, `lr=0.002`, `lr_mult=1.0`, `wd=0.0`).
   - Step 0 `train_loss_by_depth`: `[9.0109, 9.0109, 9.0109, 9.0109, 9.0109, 9.0109]`.
   - Step 49 `train_loss_by_depth`: `[6.4096, 6.4128, 6.4196, 6.4268, 6.4336, 6.4398]` (mean `6.4237`).

2. **`EMBEDLOW` (`runs2/smoke_2n_embedlow`, `--adam-mult 1.0 --head-mult 8.0`):**
   - 50 steps in 97.49 s (`1.950 s/step`).
   - Optimizer groups in `ckpt.pt`: Muon group 0 (`24` block matrices, `lr=0.002` at step 50); AdamW group 0 (`6` params: embeddings + `loop_gain`, `lr=0.002`, `lr_mult=1.0`, `wd=0.0`), group 1 (`1` param `head.weight`, `lr=0.016`, `lr_mult=8.0`, `wd=0.0`).
   - Step 0 `train_loss_by_depth`: `[9.0109, 9.0109, 9.0109, 9.0109, 9.0109, 9.0109]`.
   - Step 49 `train_loss_by_depth`: `[6.1988, 6.1870, 6.1873, 6.1945, 6.2059, 6.2204]` (mean `6.1990`).
