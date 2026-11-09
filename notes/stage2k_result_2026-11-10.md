# Stage 2K result: on separate learning rates the gated COUNT read is worse than plain COUNT, and P19 is not supported (2026-11-10)

Pre-registration `PREREG_stage2k.md` (dated 2026-11-09, commit `5454f95`). Code `src/clm/rules2k.py`, `src/clm/s2model.py` (`count_scale`), `src/clm/s2train.py` (`--count-scale`), `scripts/stage2k_driver.py`, `scripts/stage2k_analyze.py`; config `configs/stage2k.json`; tests `tests/test_s2k.py`, mutants M173 to M185 (the test suite passed in full and `scripts/mutation_audit.py` killed 181 of 181 mutants before the runs); results `results/stage2k.json` (committed with this note). Run on q-pc (RTX 4060 laptop GPU, 8 GB), GPU memory cap 0.75, nine runs, started 23:01 on 2026-11-09 NZ time and all finished in the morning of 2026-11-10. Each run took about 75 minutes. Deviations logged for Stage 2K: none.

Terms, once. A *gated read* is the Stage 2E version of the COUNT memory read: a learned weight for each n-gram order, each source and each count bin, plus a learned vector for an order that has no stored history. A *scalar control* (the "scaled" arm) is plain COUNT with one learned number that multiplies the whole memory read, with no other weights. It asks whether the gate only turned the memory volume up. *Muon* trains the 2-D block matrices and *AdamW* trains the embeddings, the head, and every gate or scalar. *Separate rates* means AdamW runs at a multiple of the Muon rate (Stage 2F): Muon at 8e-3 and AdamW at 64e-3 here. *Cross-entropy* (validation loss) is the model's average surprise per token, in nats; lower is better. *SESOI* is the smallest difference worth caring about, 0.01 nats. *Paired by seed* means each run is compared with the run of the same seed in the other arm. *X_WORSE* and *X_BETTER* are the registered verdicts for "the first arm is clearly worse (or better) by at least the SESOI, on every seed".

## Verdict

- **P19 not supported.** Gated against COUNT is X_WORSE: mean d = +0.0461, worse on every seed (+0.0587, +0.0479, +0.0318). P19 needed X_BETTER.
- **Reading: NOT_HOLDING.** The scale-only control is worse still. Scaled against COUNT is X_WORSE: mean d = +0.0770 (+0.1063, +0.0685, +0.0561).
- **Gated against scaled is X_BETTER** (mean d = -0.0308; -0.0476, -0.0206, -0.0243). By the registered rule this does not change the reading. The reading only reaches STRUCTURE if the gate first beats COUNT (PREREG section 3), and it did not. What the result shows is that the gated arm is less bad than the scaled arm. It separates two arms that both lost to plain COUNT. It does not rescue the gate, and it does not show that the gate's structure helps.
- **Five-seed rule: not triggered.** It applies only if `gated_vs_count` or `gated_vs_scaled` is INCONCLUSIVE. Neither was. Seeds 3 and 4 were not run.

Final validation loss (nats per token, first 2,097,152 tokens of `data/fineweb/val.bin`, last step). All three arms were trained on q-pc on the same code:

| Seed | COUNT | COUNT-gated | COUNT-scaled |
|---|---|---|---|
| 0 | 3.2354 | 3.2942 | 3.3418 |
| 1 | 3.2886 | 3.3365 | 3.3571 |
| 2 | 3.2969 | 3.3287 | 3.3531 |

Paired gap d_s (first arm minus second; positive means the first arm is worse), `clm.analyze.verdict`, SESOI 0.01:

| Comparison | seed 0 | seed 1 | seed 2 | mean d | Verdict |
|---|---|---|---|---|---|
| gated minus COUNT | +0.0587 | +0.0479 | +0.0318 | +0.0461 | X_WORSE |
| gated minus scaled | -0.0476 | -0.0206 | -0.0243 | -0.0308 | X_BETTER |
| scaled minus COUNT | +0.1063 | +0.0685 | +0.0561 | +0.0770 | X_WORSE |

The order is the same on every seed: COUNT is best, gated is next, scaled is last. The smallest gap that decides P19 is +0.0318 (gated against COUNT, seed 2), three times the SESOI.

## By history class

Classes are set by the stored count of the exact 4-token history, the five classes of Stage 2E. Mean over three seeds. The B0 column is B0-M at the shared rate (`class_losses_mean.b0_shared_rate`), kept for reference. B0 was trained on a different recipe from the three COUNT arms, so the B0 column is not a like-for-like comparison.

| Class | B0 (shared rate) | COUNT | gated | scaled |
|---|---|---|---|---|
| not stored | 3.3395 | 3.3111 | 3.3663 | 3.4009 |
| 1-9 | 3.3823 | 3.1157 | 3.1591 | 3.1896 |
| 10-99 | 3.4589 | 3.1917 | 3.2243 | 3.2464 |
| 100-999 | 3.4903 | 3.3392 | 3.3675 | 3.3956 |
| >=1000 | 3.5050 | 3.4789 | 3.5249 | 3.5472 |

Gap between arms (positive means the first arm is worse):

| Class | gated minus COUNT | gated minus scaled | scaled minus COUNT |
|---|---|---|---|
| not stored | +0.0552 | -0.0346 | +0.0898 |
| 1-9 | +0.0434 | -0.0305 | +0.0740 |
| 10-99 | +0.0325 | -0.0221 | +0.0547 |
| 100-999 | +0.0283 | -0.0282 | +0.0565 |
| >=1000 | +0.0460 | -0.0222 | +0.0682 |

- The gated arm is worse than COUNT in every class, by 0.0283 to 0.0552. The largest gap is in "not stored" (+0.0552), the class the absent-history vectors were meant to help.
- The scaled arm is worse than COUNT in every class, by 0.0547 to 0.0898, and worse than gated in every class.
- The two classes the Stage 2E design aimed at ("not stored" and ">=1000") are the two where the gated arm loses most to COUNT (+0.0552 and +0.0460).

PREREG section 2 requires the overall loss from `scripts/bucket_eval.py` to match `results.json` within 0.002. All nine runs match: the largest difference between the re-evaluated and the recorded loss is 8.0e-9 (read from each run's `bucket.json` on q-pc; it is not stored in `results/stage2k.json`).

## Gate table and learned scalar

Mean of three seeds. Orders 2 to 5 are rows 0 to 3. Columns are the exact count bins 1-9, 10-99, 100-999 and >=1000. The bucket columns stay at 1.0 in every row because no bucket tables were used in this stage (the Kenyon arms were not run), so they are left out. Every gate starts at 1.0.

| Order | 1-9 | 10-99 | 100-999 | >=1000 |
|---|---|---|---|---|
| 2 | 1.91 | 7.68 | 8.64 | 11.37 |
| 3 | 4.67 | 12.13 | 13.05 | 12.75 |
| 4 | 6.66 | 15.63 | 17.21 | 16.03 |
| 5 | 9.29 | 25.06 | 33.88 | 33.62 |

Learned absent-vector norms by order (2 to 5): 72.3, 47.2, 14.7, 25.8.

The gates rise with the history length and with the count bin, flattening above the 10-99 bin at orders 3 and 4, the same shape as Stage 2E. They are smaller than Stage 2E's top values (order 5 reached 90.5 there against 33.9 here), though that compares across machines and recipes.

The scale-only control's one learned scalar ended at 9.17 (seed 0), 8.88 (seed 1) and 8.83 (seed 2), mean 8.96. It started at 1.0. So the memory read was turned up about nine times, and the arm that did this was the worst of the three.

## Cost

| | COUNT | gated | scaled |
|---|---|---|---|
| s per step (q-pc) | 0.341 | 0.351 | 0.343 |
| peak VRAM MiB | 1,609 | 1,611 | 1,609 |
| peak host RSS MiB | 7,222 | 7,429 | 7,261 |

The gated read is about 3% slower per step than COUNT and the scalar about 0.6%. Peak VRAM is about 1.6 GB for all three. The 200-step smoke test, run before the real runs, measured about 0.33 s per step on q-pc. The PREREG section 6 estimate (1.4 to 1.6 hours a run, from laptop speeds) was too high. The real runs took about 75 minutes each.

## Cross-machine check (described, no decision; PREREG section 3)

Stage 2F trained plain COUNT on the Linux laptop. Stage 2K retrained it on q-pc with the same recipe, seeds, data order and initialisation. The difference by seed is the first direct measure of what changing machine does to this model:

| Seed | COUNT on laptop (Stage 2F) | COUNT on q-pc | q-pc minus laptop |
|---|---|---|---|
| 0 | 3.2362 | 3.2354 | -0.0008 |
| 1 | 3.2842 | 3.2886 | +0.0044 |
| 2 | 3.2681 | 3.2969 | +0.0288 |

Three points from these numbers, and no more.

- **Seed 2 differs by 0.0288 (about 0.029).** That is nearly three times the 0.01 effect this stage was built to resolve. Seeds 0 and 1 differ by 0.0008 and 0.0044, well under it. With three pairs, one large difference does not say whether the cause is the machine, the code version, or luck in a training run. This check cannot tell them apart.
- **The spread of plain COUNT across seeds is large.** The sample standard deviation (the usual spread measure, dividing by n - 1) of the three q-pc runs (3.2354, 3.2886, 3.2969) is about 0.0334. For the three laptop runs (3.2362, 3.2842, 3.2681) it is about 0.0244. Stage 2I's note gives B0-M's seed standard deviation on the shared recipe as about 0.0027. So the seed-to-seed spread of COUNT at the separate-rate recipe is roughly 9 to 12 times B0-M's. It is large on the laptop as well as on q-pc, so it is not only a q-pc effect. Three runs give a rough standard deviation at best.
- **At this recipe, a gap of less than about 0.03 between single runs is not trustworthy.** That reads from the two points above. Any future claim of a small effect (around 0.01) at the 64e-3 AdamW rate needs more seeds than three.

This does not threaten the verdict. Gated against COUNT and scaled against COUNT are worse by +0.0318 to +0.1063 on every seed, all one-signed, and the three arms were paired on one machine, one code version and one tree, so the machine difference does not enter the contrasts. Gated against scaled is smaller on seed 1 (-0.0206), under 0.03, but it is also one-signed on all three seeds and does not decide anything here. For scale, the Stage 2G gate-1 margin (COUNT-M against B0-M on separate rates, mean -0.1224, smallest seed gap -0.1009) is far above this spread. Nothing more is claimed from that.

## Reading beyond the rule (interpretation, untested)

These points are not registered tests.

- Both the gate table and the scalar trained with AdamW at 64e-3 and no weight decay. That is eight times the gate rate Stage 2E used. PREREG section 8 limit 2 flagged this before the runs: "A gate may move faster than is good for it." The learned scalar landed near 9 and did worse than leaving the read alone. The exact gates are large (order 5 up to about 34).
- One plausible reading is that these fast, untuned parameters overshoot. Another is that separate rates already give plain COUNT the memory scale the gate supplied in Stage 2E, because plain COUNT can scale its read by moving the head rows quickly (PREREG section 4). This stage cannot separate the two. Neither was tested.

Options that were not run, listed plainly:

- Gate and scalar at the lower Muon rate (8e-3), or with a fixed scale.
- A per-order scalar (four parameters), which would sit between the one-number control and the 1,568-parameter gate.
- Seeds 3 and 4 for all three arms.

Whether to run any of them is Nate's decision. No outcome of this stage changes gate 1 (Stage 2G).

## What the verdict does (PREREG section 5)

NOT_HOLDING: COUNT stays ungated at the separate-rate recipe, and the gate line closes at this size. The gate table and the scalar are not carried forward. Nothing here changes gate 1.

## Limits

These are PREREG section 8 restated, plus what happened during the stage.

- **The scalar control is one global scalar.** The gated arm bundles three things the scalar lacks: a weight per order, a weight per count bin and source, and the absent vectors. A gated win over the scalar would say the bundle helps, not which part. A per-order scalar would narrow it and was not run.
- **The gate and scalar rates are not tuned.** They train at 64e-3 with AdamW and no weight decay, eight times Stage 2E's gate rate.
- **One machine for the new runs.** The three arms are paired on q-pc. Anything compared with Stage 2E's gap of 0.0111 compares across machines, recipes and code versions.
- **One model size and corpus.** 13.8M non-embedding parameters, FineWeb-Edu. Three seeds resolve about 0.01 nats and no finer, and the cross-machine check above suggests that COUNT at this recipe is noisier than that.
- **The scalar adds one parameter** to the non-embedding count (the gated arm adds 1,568), which is negligible against 13.8M.
- **The cross-machine check is descriptive.** It has three pairs and is not a test.
- **The class evaluation uses 4-token history counts only**, as in Stage 2E.
- **Runs.** All nine runs finished clean: `git_dirty` false and commit `5454f95` on every run. No run diverged and none was restarted.
- **Smoke test.** Before the runs, a 200-step smoke test of each arm showed identical first-step losses (9.0109 at step 0 and 8.9999 at step 1) across the three arms. That confirms they start bit-identical. The smoke test's validation loss was not read.

## Files

`PREREG_stage2k.md` (commit `5454f95`), `src/clm/rules2k.py`, `src/clm/s2model.py` (`count_scale`), `src/clm/s2train.py` (`--count-scale`), `scripts/stage2k_driver.py`, `scripts/stage2k_analyze.py`, `configs/stage2k.json`, `tests/test_s2k.py`, mutants M173 to M185, `results/stage2k.json` (committed with this note). Run directories: `runs2/count_decoupled_q-M-s`, `runs2/count_gated_sep-M-s`, `runs2/count_scaled_sep-M-s` for seeds 0, 1 and 2.

TL:DR. We repeated Stage 2E's gated COUNT read on the separate-rate recipe (Muon 8e-3, AdamW 64e-3), with a one-scalar control, on three seeds on q-pc. The gate did not hold. Gated is worse than plain COUNT by +0.0461 on average (+0.0587, +0.0479, +0.0318, X_WORSE), and the scalar control is worse still (+0.0770, X_WORSE), so P19 is not supported and the registered reading is NOT_HOLDING. Gated beats scaled by 0.0308 (X_BETTER), but that only ranks two arms that both lost to COUNT, and the rule gives it no weight because the gate never beat COUNT. The gated arm is worse than COUNT in every history class, most in "not stored" (+0.0552). The five-seed rule did not fire, and seeds 3 and 4 were not run. Per PREREG section 5, COUNT stays ungated and the gate line closes at this size; gate 1 is unchanged. The cross-machine check found plain COUNT on q-pc differs from Stage 2F's laptop runs by -0.0008, +0.0044 and +0.0288, and plain COUNT's seed spread (sample SD about 0.033 on q-pc, 0.024 on the laptop) is about ten times B0-M's, so single-run gaps under about 0.03 are not trustworthy at this recipe. That does not threaten this verdict, whose gaps are much larger. The suggestion that fast, untuned gate rates overshoot, or that separate rates already give COUNT the scale the gate supplied, is untested. Whether to try slower rates, a per-order scalar or more seeds is Nate's call.
