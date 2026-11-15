# Stage 2M result: lowering the loop gain's AdamW rate does not revive the loop (P22 not supported), but the shared-rate loop with COUNT is alive and reaches 3.201 nats (P23 supported) (2026-11-16)

Pre-registration `PREREG_stage2m.md` (dated 2026-11-14, committed as `feb582e` before any production run). Pre-run note and smoke run `notes/stage2m_pre_run_2026-11-14.md` (commit `19f36e6`). Code `src/clm/s2model.py`, `src/clm/s2train.py`, `src/clm/rules2m.py`, `scripts/stage2m_driver.py`, `scripts/stage2m_analyze.py`; config `configs/stage2m.json`; tests `tests/test_s2m.py`, mutants M206 to M214 (210/210 killed); results `results/stage2m.json`. Run on q-pc, six runs (two arms, seeds 0 to 2), all at commit `feb582e`, tree clean (`git_dirty: false`), status complete. Launched at about 08:35 on 2026-11-14; last run finished at 03:29 on 2026-11-16 (`results.json` write times on q-pc: GAINLOW seeds 0, 1, 2 at 15:33 on 14 Nov, 05:54 on 15 Nov, 20:18 on 15 Nov; SHARED seeds 0, 1, 2 at 22:43 on 14 Nov, 13:03 on 15 Nov, 03:29 on 16 Nov). Deviations: one conduct slip logged in `notes/deviations.md` (commit `39248b3`), see Events.

Terms, once. *Cross-entropy* (validation loss) is the model's average surprise per token, in nats; lower is better. The *looped model* reuses one two-block pair up to six times per token (passes 1 to 6, depths 1 to 6) between a two-block prelude and a two-block coda, with a learned per-pass per-channel residual gain (`loop_gain`, shape 6 x 384, initialised to 1, no weight decay) and trained on the mean loss over all six depths. *Best-depth loss* is the lowest of a run's six per-depth validation losses. The *loop's benefit* inside one run is its depth-1 validation loss minus its best-depth validation loss. *SESOI* is 0.01 nats, used both as the within-run "alive" threshold and for the described paired contrasts (`clm.analyze.verdict`). Both arms carry the plain *COUNT* read at the input. *GAINLOW* (`runs2/loop_count_gainlow-M-s`) uses Stage 2L's separate-rate recipe (Muon 8e-3 on block matrices, AdamW 64e-3 on embeddings and head) except that `loop_gain` alone trains at AdamW 8e-3 (`--gain-mult 1.0`). *SHARED* (`runs2/loop_count_shared-M-s`) uses Stage 2H's shared-rate recipe (Muon 8e-3 and AdamW 8e-3 on all non-block parameters, `--adam-mult 1.0`) plus the plain COUNT read.

## Verdict

- **P22 is NOT SUPPORTED (`NOT_SUPPORTED`).** In GAINLOW, training `loop_gain` at AdamW 8e-3 while keeping embeddings and head at 64e-3 did not revive the loop. Loop benefit (depth-1 minus best-depth validation loss) is 0.000007, 0.000012, and 0.000042 nats on seeds 0, 1, and 2—all well below the 0.01 nats bar. All six depths within each run agree to within 0.00004 nats.
- **P23 is SUPPORTED (`SUPPORTED`).** In SHARED (Muon 8e-3, AdamW 8e-3, with the plain COUNT read), the loop is alive on every seed: loop benefit is **0.0388, 0.0387, and 0.0354 nats** (mean **0.0376 nats**), clearing the 0.01 nats bar on all three seeds.
- **Reading table (`PREREG_stage2m.md` section 3, row 3: *P22 not supported, P23 supported*).** Lowering the AdamW rate on `loop_gain` alone is not enough to explain why the loop went inert on the separate-rate recipe (H1 refuted). Another AdamW-trained parameter at 64e-3 (embeddings, value embeddings, or untied head; H2) is what renders the extra loop passes inert, whereas at the shared 8e-3 rate the loop works alongside the COUNT read and adds 0.038 nats of depth benefit on top of depth 1.
- **Against the stated expectation (`PREREG_stage2m.md` section 4).** P22 was expected to hold under H1 and did not. P23 was uncertain because the COUNT read might have absorbed what the loop adds; instead P23 held cleanly on every seed, with about 62% of Stage 2H's loop benefit (0.0376 vs 0.0615 nats) surviving alongside COUNT.
- **Futility check.** Printed `FUTILITY continue` after seed 0 (SHARED seed 0 benefit was 0.0388 >= 0.01).

## Within-run depth losses and loop benefit (primary decision quantities)

Validation loss on the first 2,097,152 tokens of `data/fineweb/val.bin` at step 12,207 (size M, 200M tokens, context 512):

| Arm | Seed | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 | Depth 6 | Best depth | Best loss | Loop benefit (d1 − best) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **GAINLOW** | 0 | 3.21891 | 3.21890 | 3.21890 | 3.21891 | 3.21892 | 3.21890 | 3 | 3.21890 | +0.000007 |
| **GAINLOW** | 1 | 3.21710 | 3.21709 | 3.21709 | 3.21709 | 3.21709 | 3.21710 | 4 | 3.21709 | +0.000012 |
| **GAINLOW** | 2 | 3.21911 | 3.21907 | 3.21908 | 3.21908 | 3.21909 | 3.21908 | 2 | 3.21907 | +0.000042 |
| **GAINLOW** | Mean | 3.21837 | 3.21835 | 3.21836 | 3.21836 | 3.21837 | 3.21836 | — | **3.21835** | **+0.000021** |
| **SHARED** | 0 | 3.23815 | 3.20496 | 3.19973 | 3.19934 | 3.19933 | 3.19932 | 6 | 3.19932 | **+0.038826** |
| **SHARED** | 1 | 3.24151 | 3.20816 | 3.20312 | 3.20284 | 3.20283 | 3.20282 | 6 | 3.20282 | **+0.038692** |
| **SHARED** | 2 | 3.23680 | 3.20577 | 3.20164 | 3.20142 | 3.20140 | 3.20140 | 6 | 3.20140 | **+0.035398** |
| **SHARED** | Mean | 3.23882 | 3.20630 | 3.20150 | 3.20120 | 3.20119 | 3.20118 | 6 | **3.20118** | **+0.037639** |

Reference loop benefits (`results/stage2m.json`, `reference_benefit`):
- **Stage 2L `LOOP+COUNT`** (separate rates, `loop_gain` at 64e-3): 0.000033, 0.000031, 0.000001 (mean 0.000022).
- **Stage 2H `both`** (shared rate 8e-3, no COUNT read): 0.060729, 0.065063, 0.058689 (mean 0.061493).

Mean marginal gain from each extra pass (nats; positive means the pass lowered validation loss):

| Arm | Pass 2 | Pass 3 | Pass 4 | Pass 5 | Pass 6 |
|---|---:|---:|---:|---:|---:|
| **GAINLOW** | +0.000020 | −0.000004 | −0.000004 | −0.000006 | +0.000007 |
| **SHARED** | **+0.032521** | **+0.004801** | **+0.000300** | +0.000011 | +0.000006 |

In SHARED, as in Stage 2H, pass 2 provides the bulk of the loop benefit (+0.0325 nats), pass 3 adds +0.0048 nats, pass 4 adds +0.0003 nats, and passes 5 and 6 are switched off by the learned gain without hurting loss (best depth is 6 on every seed).

## Learned `loop_gain` per pass in the final checkpoints

Mean absolute `loop_gain` across the 384 channels at step 12,207 (`gain_rows_mean_abs` in `results/stage2m.json`):

| Arm | Seed | Pass 1 | Pass 2 | Pass 3 | Pass 4 | Pass 5 | Pass 6 |
|---|---|---:|---:|---:|---:|---:|---:|
| **GAINLOW** (`gain` 8e-3, rest 64e-3) | 0 | 0.1180 | 0.0112 | 0.0131 | 0.0067 | 0.0046 | 0.0023 |
| **GAINLOW** (`gain` 8e-3, rest 64e-3) | 1 | 0.1060 | 0.0174 | 0.0065 | 0.0046 | 0.0036 | 0.0025 |
| **GAINLOW** (`gain` 8e-3, rest 64e-3) | 2 | 0.1256 | 0.0230 | 0.0057 | 0.0062 | 0.0031 | 0.0025 |
| **SHARED** (all AdamW 8e-3) | 0 | 0.3522 | 0.2907 | 0.1901 | 0.0599 | 0.0124 | 0.0036 |
| **SHARED** (all AdamW 8e-3) | 1 | 0.3842 | 0.3001 | 0.1981 | 0.0565 | 0.0105 | 0.0031 |
| **SHARED** (all AdamW 8e-3) | 2 | 0.4457 | 0.3197 | 0.1922 | 0.0504 | 0.0085 | 0.0028 |
| **Stage 2L `LOOP+COUNT`** (all AdamW 64e-3) | 0 | 0.3175 | 0.0549 | 0.0156 | 0.0131 | 0.0122 | 0.0115 |
| **Stage 2L `LOOP+COUNT`** (all AdamW 64e-3) | 1 | 0.3091 | 0.0955 | 0.0162 | 0.0119 | 0.0111 | 0.0107 |
| **Stage 2L `LOOP+COUNT`** (all AdamW 64e-3) | 2 | 0.3208 | 0.0639 | 0.0161 | 0.0122 | 0.0113 | 0.0109 |

Even with `loop_gain` trained at 8e-3, GAINLOW still drove the gains of passes 2 to 6 down to 0.002–0.023 (even smaller than Stage 2L's), whereas SHARED preserved almost the exact pass-gain profile of Stage 2H's `both` arm (~0.39, 0.30, 0.19, 0.056, 0.010, 0.003). That confirms `loop_gain` was driven toward zero on passes 2–6 not because its own learning rate was too high, but because the rest of the network (embeddings / value embeddings / untied head at 64e-3) made passes 2–6 unhelpful or harmful, so the gain learned to shut them off even at 8e-3.

## Described comparisons on best-depth loss (no decision)

Paired by seed, `d_s = first − second` (negative means the first arm is better), Stage 1 verdict rule (`SESOI = 0.01`):

| Contrast | Seed 0 | Seed 1 | Seed 2 | Mean `d` | Verdict |
|---|---:|---:|---:|---:|---|
| **SHARED vs Stage 2H `both`** (effect of adding COUNT to a live shared-rate loop) | −0.1333 | −0.1338 | −0.1303 | **−0.1325** | **X_BETTER** |
| **SHARED vs plain COUNT (shared rate, Stage 2A)** (effect of adding the live loop to shared-rate COUNT) | −0.1228 | −0.1178 | −0.1286 | **−0.1230** | **X_BETTER** |
| **SHARED vs GAINLOW** | −0.0196 | −0.0143 | −0.0177 | **−0.0172** | **X_BETTER** |
| **GAINLOW vs Stage 2L `LOOP+COUNT`** | −0.0085 | −0.0116 | −0.0061 | **−0.0087** | **NO_MEANINGFUL_DIFFERENCE** |
| **GAINLOW vs plain COUNT (separate rates, Stage 2K)** | −0.0165 | −0.0715 | −0.0778 | **−0.0553** | **X_BETTER** |
| **GAINLOW vs Stage 2H `both`** | −0.1137 | −0.1195 | −0.1126 | **−0.1153** | **X_BETTER** |

Key observations from the described contrasts:
1. **SHARED (`3.2012` mean) is the lowest validation loss achieved at Size M (200M tokens) in the project.** It beats Stage 2L's `LOOP+COUNT` (`3.2271`), `GAINLOW` (`3.2184`, by `-0.0172` nats, `X_BETTER`), plain `COUNT` on separate rates (`3.2629` on laptop in Stage 2G, `3.2737` on q-pc in Stage 2K), and Stage 2H's loop (`3.3336`, by `-0.1325` nats, `X_BETTER`).
2. **Furthermore, SHARED has tight seed-to-seed stability** (seeds `3.1993`, `3.2028`, `3.2014`, range `0.0035` nats), avoiding the large seed spread (`SD ~0.03–0.05`) that afflicted plain `COUNT` at the 64e-3 AdamW rate in Stages 2K and 2K2.
3. **Why SHARED (`adam_mult=1.0`, 8e-3) beats plain COUNT (`adam_mult=1.0`, `3.3242` in Stage 2A) by `-0.1230` nats:** Only `0.0376` nats of that `-0.1230` comes from passes 2 to 6 inside SHARED (SHARED depth 1 is already `3.2388`, which is `-0.0854` nats below Stage 2A's `3.3242`). Note that Stage 2A trained with `lr=0.032` coupled across Muon and AdamW (`micro=8`), whereas SHARED trained with `lr=0.008` on both Muon and AdamW (`micro=4`, all-depth loss). So the `-0.1230` contrast combines the loop's `0.0376`-nat multi-pass benefit with the Muon `8e-3` vs `32e-3` backbone difference and all-depth training.

## Cost

| Arm | s / step (q-pc) | Lookup s / step | Peak VRAM MiB | Peak host RSS MiB |
|---|---:|---:|---:|---:|
| **GAINLOW** | 1.937 | 0.117 | 2,586 | 7,239 |
| **SHARED** | 1.932 | 0.117 | 2,586 | 7,237 |

Both arms match Stage 2L's `LOOP+COUNT` (`1.921 s/step`, `2,586 MiB` VRAM, `7,239 MiB` RSS). As noted in `PREREG_stage2m.md` limit 3, the all-depth looped arms do about 2.4 times the block passes per step of a plain 6-layer model (`~4.9x` wall-clock vs `0.397 s/step` for plain `COUNT` at `micro=8`), though at inference depth 3 (`3.2015` nats, within `0.0003` of depth 6) uses 10 block evaluations vs 6 for `B0-M`.

## Smoke run pointer (`PREREG_stage2m.md` section 6)

Recorded before the production runs in `notes/stage2m_pre_run_2026-11-14.md` (`runs2/smoke_2m`, 400 steps of GAINLOW seed 0 on q-pc, validation loss not read): step-0 loss matched Stage 2L (`9.0109`), optimizer groups verified, step-399 training losses by depth were `4.2124, 4.2094, 4.2076, 4.2064, 4.2060, 4.2064`, and mean absolute loop gains at step 400 were `0.605, 0.669, 0.641, 0.621, 0.604, 0.590`. On the full 12,207-step schedule (where LR stays near peak much longer after the 200-step warmup), GAINLOW's passes 2 to 6 still collapsed later in training.

## Events and deviations

1. **Pre-futility validation peek on GAINLOW seed 0 (`notes/deviations.md`, 2026-11-14, commit `39248b3`).** While checking progress at 20:24 NZ on 2026-11-14, the last line of `runs2/loop_count_gainlow-M-0/log.jsonl` was printed, exposing `val_loss_by_depth` (`~3.2189` at all six depths) before the driver's futility check ran. Nothing in the code, config, driver, or rules was changed; all later status checks read training fields only.
2. **Missing Stage 2A reference files on q-pc at analysis time.** After all six runs completed cleanly on q-pc (`git_dirty: false`), `stage2m_driver.py` invoked `scripts/stage2m_analyze.py`, which exited because `runs2/count-M-{0,1,2}/results.json` (trained on the Linux laptop in Stage 2A) had not been copied to q-pc. Copying those three reference `results.json` files into `q-pc`'s git-ignored `runs2/` directory and re-running `scripts/stage2m_analyze.py` on q-pc produced `results/stage2m.json` (`sha256: 74b51f77b730620428a9b2587fb2d5f1bd6ee2cf3e33c3fabde8cf2bd5db2294`) with zero code edits.

## What the verdict does (`PREREG_stage2m.md` section 5)

- **P22 not supported, P23 supported:**
  - The loop + COUNT combination works out of the box on the **shared-rate recipe (`SHARED`: Muon 8e-3, AdamW 8e-3)**, earning **0.0376 nats** from looping (`P23`) and reaching **3.2012 nats** best-depth validation loss (**3.2015 nats at depth 3**), beating every previous Size-M recipe (`Stage 2L LOOP+COUNT` 3.2271, `GAINLOW` 3.2184, `Stage 2G COUNT` 3.2629, `Stage 2H loop` 3.3336).
  - If we want to know *which* specific AdamW parameter at 64e-3 (`embed`, `value_embed`, or `head.weight`) killed the loop on the separate-rate recipe (H2), a further split stage would be needed—though since `SHARED` (8e-3) already beats `GAINLOW` (64e-3) on best-depth loss by `-0.0172` nats (`X_BETTER`), `SHARED` is already the stronger LOOP+COUNT recipe.
  - Next steps remain with Nate: (1) whether to carry `SHARED` (or plain `COUNT`, or both) into Gate 3 (Size-L check / over-training check), (2) whether to split the remaining AdamW parameters (`head.weight` vs `embed`), (3) the paused RL exit question, and (4) the ETHOS side test.

## Limits

`PREREG_stage2m.md` section 8 limits 1 to 8 apply: GAINLOW moved only `loop_gain`'s rate and cannot distinguish between `embed`, `value_embed`, and `head.weight` (H2); the gradient-starvation mechanism was not directly instrumented; looped arms are token-matched, not compute-matched (~2.4x block passes per step); three seeds; `SHARED` vs Stage 2H `both` crosses laptop (seed 0) and q-pc (seeds 1–2); one read (`COUNT`), one placement (input), one size (`M`), one corpus.

TL:DR. Stage 2M tested why Stage 2L's loop was inert on the separate-rate recipe. Moving `loop_gain` alone down to AdamW 8e-3 (`GAINLOW`, P22) did **not** revive the loop (`NOT_SUPPORTED`, benefit 0.00001 to 0.00004 nats), refuting H1 and implicating the 64e-3 AdamW rate on the other non-block parameters (embeddings/head, H2). Running the loop + COUNT on Stage 2H's shared 8e-3 rate (`SHARED`, P23) **did** revive the loop (`SUPPORTED`, benefit 0.0388, 0.0387, 0.0354 nats) and reached **3.2012 nats** mean best-depth validation loss (3.2015 at depth 3), beating `GAINLOW` (-0.0172, `X_BETTER`), Stage 2H's loop without COUNT (-0.1325, `X_BETTER`), and plain COUNT.
