# Stage 2H result: the per-pass gain plus the all-depth loss beats Stage 2D's looped model by 0.038 nats, P17 is supported (2026-11-09)

Pre-registration `PREREG_stage2h.md` (dated 2026-11-06, commit `eabc1c3`). Code `src/clm/s2loop.py`, `src/clm/s2model.py`, `src/clm/s2train.py`, `src/clm/rules2h.py`, `scripts/stage2h_driver.py`, `scripts/stage2h_analyze.py`; config `configs/stage2h.json`; runs `runs2/loop_gain_all-M-{0,1,2}`, `runs2/loop-M-{0,1,2}`, `runs2/loop_gain-M-0`, `runs2/loop_all-M-0`; results `results/stage2h.json` (committed with this note). Deviations logged for Stage 2H: two, dated 2026-11-08 (the Windows port) and 2026-11-09 (the analysis loader fix), both in `notes/deviations.md`.

## Verdict: P17 is supported (X_BETTER). The futility stop did not trigger.

**P17 (does the looped model with both changes reach a best-depth validation loss at least 0.01 nats lower than Stage 2D's looped model, on three seeds?): yes.** Pairs are matched by seed. d is the best-depth loss of `both` minus the best-depth loss of `reference`, so a negative d means `both` is better. SESOI (the smallest effect that matters, registered as 0.01 nats) is the bar.

| Seed | d (nats) |
|---|---|
| 0 | -0.0378 |
| 1 | -0.0475 |
| 2 | -0.0298 |
| **Mean** | **-0.0384** |

X_BETTER needs a mean d of -0.01 or lower and every seed below zero. Both hold, and the mean is almost four times the SESOI. The same rule, read as description only, also gives X_BETTER at depth 1 (mean d -0.0297) and at the deepest depth, 6 (mean d -0.0415).

The futility stop (PREREG section 5) would have cancelled seeds 1 and 2 if seed 0 had not been better. Seed 0 was better, so all runs went ahead.

## What was run

Four arms, all size M (13.8M non-embedding parameters, 200M tokens, 12,207 steps of 32 x 512 tokens, the Stage 2D recipe). "Seed" is the number that fixes the random starting weights and data order. "Depth" is how many times the shared two-block loop runs, from 1 to 6. "Pass" is one run through the loop.

- **`reference`** is Stage 2D's looped model: no gain, one random depth per micro-batch, micro-batch 8. Seed 0 is Stage 2D's finished run `runs2/loop-M-0`, reused. Seeds 1 and 2 are new.
- **`both`** is the new model. It has a per-pass residual gain (`loop_gain`): a table of 6 x 384 numbers, one per pass and channel, all starting at 1. On pass r the stream x becomes x + g_r * (loop(x) - x), so the gain scales how big a step that pass may take. It also trains on the mean loss over all six depths in every step (`--loop-loss all`), where `reference` trains on one random depth. Micro-batch 4 (registered for the all-depth arms). Seeds 0, 1, 2.
- **`gain`** (ablation, seed 0 only) is the gain alone. An ablation removes one change to see what the other does by itself.
- **`all`** (ablation, seed 0 only) is the all-depth loss alone.

All losses are validation loss in nats per token (lower is better; a nat is the unit the loss is measured in) at the last step, on the first 2,097,152 tokens of `data/fineweb/val.bin`. "Best-depth loss" is the lowest of the six per-depth losses of one trained model.

Machines. Seed 0 of `both` and `reference` ran on the Linux laptop (RTX 3050, 4 GB of GPU memory). Every other run ran on q-pc, the Windows work laptop (RTX 4060, 8 GB). See Limits for what that means.

## Best-depth losses

B0-M's three Stage 2.0 losses average 3.4041. B0-M is the plain six-block model with no loop, the reference point for every stage.

| Run | Seed 0 | Seed 1 | Seed 2 | Mean | Below B0-M mean (3.4041) |
|---|---|---|---|---|---|
| `reference` | 3.3705 (depth 4) | 3.3841 (depth 4) | 3.3615 (depth 4) | 3.3720 | about 0.032 |
| `both` | 3.3326 (depth 6) | 3.3366 (depth 5) | 3.3317 (depth 5) | 3.3336 | about 0.0705 |

The reference's best depth is 4 on every seed, as in Stage 2D. The best depth of `both` moves between 5 and 6, but see the next section: its losses at depths 4, 5 and 6 are almost the same, so which one is lowest carries little meaning.

`both` is lower than B0-M by about twice as much as the reference is. That is not a like-for-like compute comparison: a looped model at depth 4 runs 12 blocks per token against B0's 6.

## Ablations (seed 0, one run each)

| Seed 0 | Best-depth loss | Difference from reference seed 0 (3.3705) |
|---|---|---|
| `gain` alone | 3.3468 (depth 5) | -0.0237 |
| `all` alone | 3.3452 (depth 4) | -0.0252 |
| `both` | 3.3326 (depth 6) | -0.0378 |

Each change alone helps by about 0.024 to 0.025. The two single effects add up to -0.0489, which is larger than the combined -0.0378. So the changes overlap: part of what each one fixes, the other fixes too. B0-M's seed standard deviation (the spread between seeds) is about 0.0027, and these are single runs, so read the ablations as pointers only. They do not say which change is better.

The PREREG expected `all` to lower the depth-1 loss towards B0-M's 3.4041, and `gain` to make the marginal gains of passes 4 to 5 and 5 to 6 non-negative. Both held, one of them only just. `all` brought the depth-1 loss from the reference's 3.4221 (seed 0) down to 3.4034, which is 0.0007 below B0-M's mean of 3.4041. `gain` made the passes 4 to 5 gain +0.000013, but the 5 to 6 gain is -0.000023: not strictly non-negative, yet about zero, and far from the reference's -0.0010 and -0.0022. Read the second expectation as met in practice and missed to the letter.

## Marginal gains

The marginal gain of one extra pass is the loss at depth r minus the loss at depth r + 1, so a positive number means the extra pass helped. Mean over the three seeds.

| Extra pass | 1 to 2 | 2 to 3 | 3 to 4 | 4 to 5 | 5 to 6 |
|---|---|---|---|---|---|
| `reference` | 0.0431 | 0.0083 | 0.0014 | -0.0010 | -0.0022 |
| `both` | 0.0534 | 0.0076 | 0.00046 | 0.000016 | -0.0000037 |

The reference repeats what Stage 2D saw: past depth 4 an extra pass hurts. In `both` the gains shrink to nothing instead. After depth 4 the change per pass is 0.000016 and then -0.0000037, which is zero for practical purposes. Depth 4 and depth 6 of `both` differ by under 0.0005. The audit's first complaint (late passes overshoot because nothing damps them) is fixed in the sense that more passes no longer cost anything measurable.

The first extra pass is worth more in `both` (0.0534 against 0.0431). Every later pass is worth about the same or less (0.0076 against 0.0083, then 0.00046 against 0.0014). Why was not tested.

## Learned gains

The gain starts at 1 on every pass and channel. This table is its mean over the 384 channels after training, for `both`.

| Pass | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 1 | 0.385 | 0.436 | 0.461 |
| 2 | 0.313 | 0.311 | 0.318 |
| 3 | 0.199 | 0.203 | 0.186 |
| 4 | 0.061 | 0.063 | 0.049 |
| 5 | 0.012 | 0.014 | 0.009 |
| 6 | 0.0018 | 0.0032 | 0.0013 |

All three seeds learned the same shape. The gain drops from 1 to about 0.4 on pass 1, falls slowly to pass 3, then shrinks fast: from pass 3 onward each pass has a gain roughly 3 to 7 times smaller than the one before it. By pass 5 the loop writes almost nothing into the stream. Some individual channels stay large: the largest gains in passes 1 to 3 are about 2 to 2.9, well above the starting value of 1.

## Cost

Seconds per training step and peak GPU memory. Step times are not comparable across machines, and these runs were split across two. Each figure is the mean over that arm's runs. `gain` and `all` ran only on q-pc. `reference` and `both` mix machines: their seed 0 ran on the Linux laptop and seeds 1 and 2 on q-pc, so those two figures are blends.

| Arm | Seconds per step | Peak GPU memory |
|---|---|---|
| `reference` | 0.508 | about 2.6 GB |
| `gain` | 0.436 | about 2.6 GB |
| `both` | 1.848 | about 2.6 GB |
| `all` | 1.833 | about 2.6 GB |

The all-depth loss is what costs time. Each step runs the loop six times and reads out after every pass, about 26 block passes against the reference's average of 11, about 2.4 times (registered limit 2). The gain alone adds no visible cost. So the win is not a compute-matched win: `both` sees the same tokens as `reference` but does about 2.4 times the work per step.

## Reading beyond the rule

These points are read from the tables and were not registered as tests.

- **The learned gain is a soft stopping point.** The model was never told when to stop looping. It learned to shrink its own step to almost zero by pass 5, near the four passes that Stage 2D's reference found best. This was not registered, and it is a description of three trained models, not a tested claim about why.
- **Extra passes are now free to the loss.** The flat tail of the marginal gains means the question "how many passes?" matters much less for `both`. That bears on later halting work, because a halting rule has least to win when every depth beyond 4 gives the same loss.
- **The effect is far above the seed spread, but the reference seeds are themselves spread out.** The three reference best-depth losses span 0.0226 (3.3615 to 3.3841), against 0.0049 for `both` (3.3317 to 3.3366). B0-M's seed SD of about 0.0027 comes from a different model, so it only gives a rough guide. Every pair still favours `both` by about 0.03 or more (smallest 0.0298). Why the reference varies more was not tested.
- **The two changes overlap.** Gain alone and all-depth loss alone each give about 0.024 to 0.025 on seed 0, and together 0.0378. They are not independent fixes of two separate problems.

## What this unlocks

Per PREREG section 7, the gain plus the all-depth loss become the LoopGPT recipe for later looped tests (the audit's 5B halting probe, and COUNT plus LoopGPT), each with its own pre-registration. Before any halting or RL decision, the Stage 2D oracle analysis (`scripts/stage2d_analyze.py`) must be repeated on `loop_gain_all-M-0`. The Stage 2D RL gate (OPEN) was measured on the old model. That repeat is the next step and will run on q-pc. Its result will be a separate note. RL stays paused (Nate, 2026-11-03), and no RL was run in this stage. No outcome of Stage 2H changes the memory track.

## Limits

These are PREREG section 9 restated, plus what happened during the stage.

- **Equal tokens, not equal compute.** `both` does about 2.4 times the block passes per step (registered limit 2). A win here does not mean `both` is better per unit of compute. No compute-matched comparison was registered or run.
- **Two machines.** Seed 0 of `both` and `reference` ran on the Linux laptop, everything else on q-pc. Floating-point results are not bit-identical across the two. The effect (about 0.03) is about ten times the seed SD (about 0.0027), so it is very unlikely to be a machine artefact, but the cross-machine difference was not measured. See `notes/deviations.md` entries of 2026-11-08 (the Windows port) and 2026-11-09 (the analysis loader defaults Stage 2D seed 0's unrecorded `adam_mult` to 1.0). The first changed no model, loss, optimiser, config or rule. The second changed only the analysis loader, before it had produced any number.
- **Micro-batch 4 for the all-depth arms against micro-batch 8 for the reference** (limit 4). It changes only rounding in bf16 accumulation, not the data or the loss.
- **One shared Muon and AdamW rate**, as in Stage 2D. The separate rates of Stages 2F and 2G were not used, so these results compare with Stage 2D, not with 2G.
- **One size, one corpus** (about 14.5 tokens per non-embedding parameter).
- **The best depth is the lowest of six readouts of one model.** A mild selection effect, small here because depths 4 to 6 of `both` differ by under 0.0005.
- **Ablations are one seed each** and cannot resolve 0.01 nats.
- **Three seeds resolve about 0.01 nats and no finer.** The mean d of -0.0384 is well past that, but seeds 3 and 4 were not run.
- **The learned-gain pattern is unregistered.** It is described from three runs.
- **One gain parameterisation, equal depth weights, no step-embedding variant** (limit 8). A different way of damping late passes was not tested.

TL:DR. Stage 2D's looped model got a per-pass gain and a loss averaged over all six depths, and on three matched seeds its best-depth validation loss fell by 0.0384 nats on average (-0.0378, -0.0475, -0.0298), nearly four times the 0.01 bar, so P17 is supported (X_BETTER, also at depth 1 and depth 6). The new model is about 0.0705 below B0-M and the reference about 0.032. Extra passes now cost nothing measurable after depth 4, and the model learned on its own to shrink its late-pass writes to near zero. Each change alone gives about 0.024 to 0.025 on one seed and the two overlap. The win is not compute-matched (about 2.4 times the block passes per step), seed 0 ran on a different machine from seeds 1 and 2, and the ablations are single runs. Next is the Stage 2D oracle analysis on `loop_gain_all-M-0` (to run on q-pc, separate note) before any halting or RL decision; RL stays paused.
