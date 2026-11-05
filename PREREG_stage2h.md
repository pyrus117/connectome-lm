# Stage 2H pre-registration: a per-pass gain and the loss on every depth in the looped model (follow-up 5A)

Dated 2026-11-06. Committed before any Stage 2H run. Its sha256 is recorded in each run's `results.json`. Plan: `docs/superpowers/plans/2026-11-02-stage2-kenyon-memory.md`. Code: `src/clm/s2loop.py`, `src/clm/s2model.py`, `src/clm/s2train.py`, `src/clm/rules2h.py`, `scripts/stage2h_driver.py`, `scripts/stage2h_analyze.py`. Config `configs/stage2h.json`.

**Origin.** The Stage 2 audit note (`neural-vault/research-logs/2026-11-04 connectome-lm Stage 2 synthesis, local minima and follow-up avenues.md`, Local Minimum 5) says Stage 2D's `LoopGPT` (`src/clm/s2loop.py`) runs its two shared loop blocks with no pass index, and trains each micro-batch at one uniformly sampled depth. It makes two claims.

1. Without knowing which pass it is on, the loop cannot damp its step on late passes, so passes 5 and 6 overshoot. In Stage 2D their marginal gains were -0.0010 and -0.0022 nats.
2. Sampling one depth per micro-batch gives depth 1 only a sixth of the updates. Depth 1 (3.4221) ended 0.018 nats worse than B0-M (3.4041) at identical six-block compute.

The audit's flagged follow-up 5A reads: "Add a lightweight per-pass step embedding or per-pass residual gain (6 x d_model = 2,304 params) and train on the multi-depth loss across all passes r=1...6 per forward pass."

Stage 2D's result is in `notes/stage2d_result_2026-11-04.md`, `results/stage2d.json` and the run `runs2/loop-M-0`. Its validation loss at depths 1 to 6 was 3.4221, 3.3798, 3.3718, 3.3705, 3.3715, 3.3737. The best is depth 4, 0.0336 below B0-M's mean of 3.4041, at twice the blocks. B0-M's three Stage 2.0 losses are 3.405540, 3.405689 and 3.400939. Their standard deviation is 0.0027, computed from those three numbers. Stage 2D's file says 0.0054, which does not match them, so 0.0027 is the figure used here.

## 1. Question

Cross-entropy (validation loss) is the model's average surprise per token, in nats; lower is better.

**P17 (the only decision).** Does the looped model with both changes (a per-pass residual gain and the mean loss over all six depths) reach a best-depth validation loss at least 0.01 nats lower than Stage 2D's looped model, on three seeds?

"Best-depth loss" is the lowest of the six per-depth validation losses of one trained model at its last step.

Described, no decision: the depth-1 loss (symptom 2), the deepest-depth loss and the marginal gain of each extra pass (symptom 1), and the learned gains.

**Stated expectation, before any run.** The audit expects P17 to hold. Against that, training every depth spends capacity on the shallow readouts, and the plan (section 8 of the Stage 2 plan) did not expect looping to help at about 14.5 tokens per non-embedding parameter. Either outcome is written up.

## 2. Fixed design

- **Model.** `LoopGPT` as in Stage 2D: B0-M's six blocks as 2 prelude blocks, a 2-block loop shared across passes, and 2 coda blocks. d_model 384, 6 heads, context 512, vocab 8,192, untied head, 13,762,562 non-embedding parameters. Depth r runs the loop r times; maximum depth is 6.
- **Training settings are Stage 2D's, unchanged.** Muon lr 8e-3 on the block matrices, with AdamW at the same shared rate (`adam_mult` 1.0) on the rest. Weight decay 0.1 on block matrices only. Warmup 200 steps then linear decay to zero. Gradient clip 1.0, bf16, GPU memory cap 0.75. 200,000,000 tokens = 12,207 steps of 32 x 512. Same data order and name-keyed initialisation per seed as B0-M. The separate Muon and AdamW rates of Stage 2F and 2G are not used, so that every arm stays comparable with Stage 2D's run (limit 3).
- **Change 1, the gain (`loop_gain`, `--loop-gain`).** A parameter of shape 6 x 384 = 2,304 numbers, one per pass and channel, initialised to 1. Pass r replaces the residual stream x by x + g_r * (loop(x) - x), where g_r is that pass's row, applied per channel. At initialisation this is exactly the ungated model, so a seed starts from the same weights as Stage 2D (tested). It is trained by AdamW with no weight decay, because it is a 2-D parameter outside the blocks and the existing split already puts it there. Why a gain and not a step embedding: the audit's mechanism is that late passes cannot damp their step, and a gain on the change is that damping directly. The step-embedding variant the audit also named is not tested (limit 8).
- **Change 2, the all-depth loss (`--loop-loss all`).** Every micro-batch runs the loop six times and reads out after every pass through the coda and head. The training loss is the plain mean of the six cross-entropies, with equal weights. Why: every depth then gets an update on every step, instead of a sixth of them. Stage 2D instead drew one depth per micro-batch (`--loop-loss sampled`, the default, unchanged).
- **Micro-batch 4 for the all-depth arms.** Six readouts at micro-batch 8 do not fit the laptop GPU's 0.75 memory cap, so the all-depth arms use micro-batch 4, with 8 accumulation steps for the same 32 windows per step. Measured on the laptop (RTX 3050, random tokens, gain on, 2026-11-06):

  | Arm and micro-batch | Seconds per step | Peak VRAM |
  |---|---|---|
  | all-depth, micro 8 | out of memory | out of memory |
  | all-depth, micro 4 | 1.765 | 2,582 MiB |
  | all-depth, micro 2 | 1.856 | 1,466 MiB |
  | sampled depth, micro 8 | 0.605 | 2,637 MiB |

  Stage 2D's actual run took 8,060 s (0.66 s per step on average). Micro 4 changes only rounding in bf16 accumulation, not the data or the loss. It is still a difference between arms, and is listed as limit 4.
- **Metric.** Validation loss at each depth 1 to 6 on the first 2,097,152 tokens of `data/fineweb/val.bin`, at the last step. The training script already records it as `val_loss_by_depth`.

## 3. Arms

All arms are size M and write under `runs2/`. Each arm is chosen by command-line switches on the one config `configs/stage2h.json`.

- **`reference`.** Stage 2D's looped model: no gain, sampled depth, micro 8. Seed 0 is Stage 2D's finished run `runs2/loop-M-0` (commit 744cdce, clean tree, pre-registration `PREREG_stage2d.md`). It is reused, not rerun. Seeds 1 and 2 are new: `runs2/loop-M-1` and `runs2/loop-M-2`. They use the same code path as seed 0. The code for a model without the gain is unchanged, and the changed lines are tested to leave it so.
- **`both`.** Gain plus all-depth loss, micro 4. Seeds 0, 1, 2: `runs2/loop_gain_all-M-0`, `runs2/loop_gain_all-M-1`, `runs2/loop_gain_all-M-2`.
- **`gain`** (ablation, seed 0 only, micro 8, so it draws the same sampled depths as the reference): `runs2/loop_gain-M-0`.
- **`all`** (ablation, seed 0 only, micro 4): `runs2/loop_all-M-0`.

## 4. Procedure

One resumable driver, `scripts/stage2h_driver.py`, runs under `systemd-run --user` with the GPU memory cap 0.75. It does this because the harness kills long background jobs and the laptop GPU needs headroom. `embeddinggemma` is stopped by hand before the driver starts and restarted by hand after it ends, then checked that it answers.

Order:

1. `both` seed 0 (`reference` seed 0 already exists).
2. The futility check (section 5).
3. Unless the check stops the stage, `both` and `reference` seeds 1 and 2.
4. `gain` seed 0 and `all` seed 0.
5. The analysis, once.

While runs are in progress only the step, seconds and training loss are read. A validation loss of a `both` or `reference` run is never read, except by the mechanical futility check.

Time on the laptop: about 6.2 h per all-depth run (12,207 steps x 1.765 s plus evaluation) and about 2.2 h per sampled-depth run. If the futility check continues, the total is about 18.6 h for `both` x3, 4.4 h for `reference` seeds 1 and 2, 2.2 h for `gain` and 6.2 h for `all`, about 31 h. If it stops, `both` seed 0 plus the two ablations come to about 14.6 h. No money.

Every run records `git_dirty`. The analysis rejects a dirty run, a run whose pre-registration hash is not this file's (Stage 2D's seed 0 is checked against `PREREG_stage2d.md`), a run with the wrong arm or settings, and a run with fewer than six depths.

## 5. Decision rule

Pairs are matched by seed. d_s = best-depth loss of `both` minus best-depth loss of `reference`. The verdict is the Stage 1 rule, `clm.analyze.verdict`, with a SESOI (smallest effect that matters) of 0.01 nats.

- **X_BETTER:** mean d <= -0.01 and every d_s < 0.
- **X_WORSE:** mean d >= +0.01 and every d_s > 0.
- **NO_MEANINGFUL_DIFFERENCE:** |mean d| < 0.01 and every |d_s| < 0.02.
- **INCONCLUSIVE:** anything else.
- **P17** is supported if and only if the verdict is X_BETTER.

Seeds 3 and 4 are not added automatically. An INCONCLUSIVE goes to Nate. The same rule is also applied, as description only, to the depth-1 loss and the deepest-depth loss of `both` against `reference`.

**Futility stop (registered now).** After `both` seed 0 and `reference` seed 0 exist, take (best-depth loss of `both` seed 0) minus (best-depth loss of `reference` seed 0). If it is zero or more, X_BETTER is impossible, because it needs every seed below zero. Then seeds 1 and 2 of `both` and `reference` are not run, and P17 is reported as NOT_SUPPORTED_FUTILITY with the seed-0 difference. This check is the only validation read of a primary arm before the analysis. The driver does it and prints only "FUTILITY continue" or "FUTILITY stop", with no numbers. The two ablation runs still go ahead either way. Reason: it saves about 17 hours when the answer is already settled, and it cannot change a verdict.

## 6. Ablations

Seed 0 only, described, no decision. The expectations are stated now.

- `all` should lower the depth-1 loss towards B0-M's 3.4041 (symptom 2).
- `gain` should make the marginal gains of passes 4 to 5 and 5 to 6 non-negative (symptom 1).

One seed cannot resolve 0.01 nats, so these are pointers only.

Also reported: the learned gains per pass (mean, standard deviation, minimum, maximum and distance from 1 of each row), seconds per step and peak VRAM per arm, and the B0-M mean for orientation.

## 7. What a verdict does

- **P17 supported:** the gain plus the all-depth loss become the LoopGPT recipe for later looped tests (the audit's 5B halting probe, and COUNT plus LoopGPT), each with its own pre-registration. Before any halting or RL decision, the Stage 2D oracle analysis (`scripts/stage2d_analyze.py`) must be repeated on `loop_gain_all-M-0`. The 2D RL gate (OPEN, paused by Nate on 2026-11-03) was measured on the old model. No RL is run in this stage.
- **P17 not supported** (X_WORSE, NO_MEANINGFUL_DIFFERENCE, INCONCLUSIVE or futility stop): written up, and Stage 2D's recipe stays. The ablations are the pointer to which part helps or hurts.
- No outcome changes the memory track.

Every outcome is written up in `notes/stage2h_result_<date>.md`.

## 8. Conduct

A diverged run (non-finite loss or gradient) is rerun once from scratch. A second divergence is reported and gives no verdict. No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash. This stops the design drifting toward a result. After any change to `src/`, the test suite and `scripts/mutation_audit.py` must pass.

The tests also include a resumed run that ends bit-identical, for the gain, for the all-depth loss, and for both together.

Planted-bug mutants guard this stage (M134 to M151). A test that cannot catch its mutant does not count.

- M134: gain applied to the output instead of the change.
- M135: every pass uses pass 1's gain.
- M136: gain initialised to zero instead of one.
- M137: the all-depth loss returns only the deepest depth.
- M138: a depth read out before its pass runs.
- M139: the gain missing from the per-token exit path.
- M140: the gain missing from the all-depths readout.
- M141: the training loss is the sum of the depth losses instead of the mean.
- M142: the all-depth switch ignored.
- M143: the gain switch not reaching the model.
- M144: the two switches accepted outside the looped arch.
- M145 and M146: the command-line switches `--loop-gain` and `--loop-loss`, and `--micro`, not reaching the config.
- M147: best depth chosen as the highest loss.
- M148: marginal gains with the wrong sign.
- M149: P17 read on the deepest depth instead of the best depth.
- M150: the contrast computed as reference minus new.
- M151: the futility stop at "greater than zero" instead of "zero or more".

Existing mutants M53 and M54 target the same bugs as before. They were re-pointed at the lines that now call the per-pass function.

## 9. Limits stated in advance

1. **One size and one corpus.** 13.8M non-embedding parameters, about 14.5 tokens per parameter.
2. **Equal tokens, not equal compute.** An all-depth step runs about 26 block passes (2 prelude, 12 loop, 12 coda) against the reference's average of 11. That is about 2.4 times. A win is not a compute-matched win, and a loss is not a surprise.
3. **The shared Muon and AdamW rate of Stage 2D**, not the Stage 2F/2G separate rates.
4. **Micro 4 for the all-depth arms against micro 8 for the reference.**
5. **The best depth is the lowest of six readouts of one model.** That is a mild selection effect. It is small because neighbouring depths differ by about 0.001 to 0.003.
6. **The reference's seed 0 comes from an earlier commit** and was trained while other jobs shared the laptop. Step time rose partway, with no effect on losses.
7. **The ablations are one seed each.**
8. **Equal weights over depths, one gain parameterisation, and no step-embedding variant.** Gradient clipping at 1.0 may bind differently for the mean-of-six loss.
9. **Three seeds resolve about 0.01 nats and no finer.**

TL:DR. We test whether two changes to Stage 2D's looped model, a per-pass residual gain (2,304 numbers, starting at 1) and training on the mean loss of all six depths at every step, lower the best-depth validation loss by at least 0.01 nats (P17, audit follow-up 5A). Three seeds of the new model (`both`) are matched against Stage 2D's looped model (`reference`, seed 0 reused, seeds 1 and 2 new). P17 holds only on X_BETTER. If seed 0 of `both` is not better than seed 0 of `reference`, the stage stops early and reports NOT_SUPPORTED_FUTILITY. Two one-seed ablations point at which change helps. The new arm costs about 2.4 times the compute per step, so a win is not compute-matched. About 31 h on the laptop, or about 14.6 h if the futility stop fires, and no money.
