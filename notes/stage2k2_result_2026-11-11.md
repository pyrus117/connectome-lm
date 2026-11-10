# Stage 2K2 result: on a slower rate no COUNT read beats plain COUNT across five seeds, and P20 is not supported (2026-11-11)

Pre-registration `PREREG_stage2k2.md` (dated 2026-11-10, commit `67b7473`). Code `src/clm/rules2k2.py`, `src/clm/s2model.py` (`param_groups(..., gate_mult)`, per-order `count_scale`), `src/clm/s2train.py` (`--gate-mult`, `--count-scale-orders`), `scripts/stage2k2_driver.py`, `scripts/stage2k2_analyze.py`; config `configs/stage2k.json` (unchanged); tests `tests/test_s2k2.py`, mutants M186 to M198 (the test suite passed in full and `scripts/mutation_audit.py` killed 194 of 194 mutants before the runs); results `results/stage2k2.json` (already committed). Run on q-pc (RTX 4060 laptop GPU, 8 GB), GPU memory cap 0.75, 17 runs: 15 new candidate runs (three arms, five seeds) and 2 new plain COUNT runs (seeds 3 and 4). Batch: the first new run (`count_gated_low`, seed 0) finished at 14:05 NZ on 2026-11-10, about 73 minutes after it started, and the last (`count_oscaled_low`, seed 4) finished at 09:39 on 2026-11-11 (file times of the `results.json` files on q-pc). Each run took about 75 minutes. Deviations logged for Stage 2K2: none.

Terms, once. *Cross-entropy* (validation loss) is the model's average surprise per token, in nats; lower is better. A *gated read* is the Stage 2E version of the COUNT memory read: a learned weight for each n-gram order, each source and each count bin, plus a learned vector for an order that has no stored history (1,568 extra numbers). A *scalar* is one learned number that multiplies the whole memory read. A *per-order scalar* is four learned numbers, one for each n-gram order from 2 to 5, so it sits between the single scalar and the gate. *Muon* trains the 2-D block matrices and *AdamW* trains the embeddings and the head. *Separate rates* means AdamW runs at a multiple of the Muon rate: Muon at 8e-3, AdamW at 64e-3. In Stage 2K the gate and scalar trained at that 64e-3 (the *high rate*). In this stage they train in their own AdamW group at 8e-3, the Muon rate (the *low rate*, `gate_mult` 1.0). *Paired by seed* means each run is compared with the run of the same seed in plain COUNT; the gap is d_s, the candidate's loss minus COUNT's, so positive means the candidate is worse. *SESOI* is the smallest difference worth caring about, 0.01 nats. The registered verdicts (`clm.analyze.verdict`, the Stage 1 rule): *X_BETTER* means the mean gap is at or below -0.01 and every seed gap is below zero. *X_WORSE* means the mean gap is at or above +0.01 and every seed gap is above zero. *NO_MEANINGFUL_DIFFERENCE* means the mean gap is under 0.01 in size and every seed gap is under 0.02 in size. *INCONCLUSIVE* is everything else. The *sample standard deviation* (SD) is the usual spread measure, dividing by n - 1.

## Verdict

- **P20 not supported.** P20 needed at least one candidate to be X_BETTER than plain COUNT on five seeds. None was.
- **All three candidates are INCONCLUSIVE against COUNT** (paired by seed, SESOI 0.01). Positive means the candidate is worse than COUNT.
  - gated_low: mean d = +0.0147 (+0.0135, +0.0074, +0.0208, -0.0217, +0.0535).
  - scaled_low: mean d = +0.0261 (+0.0281, +0.0189, +0.0298, -0.0103, +0.0642).
  - oscaled_low: mean d = +0.0169 (+0.0192, +0.0136, -0.0001, -0.0118, +0.0637).
- **INCONCLUSIVE here is not a tie.** It means the seeds disagree and the mean difference is not clearly past the bar. The rule asks every seed to point the same way. Seed 3 points the other way for all three candidates, and so does seed 2 for oscaled_low (by 0.0001). The mean gaps are positive, which leans toward the candidates being worse, but the large seed 4 gaps (+0.0535 to +0.0642) carry much of that mean, and seed 3 reverses it. The registered rule gives neither "worse" nor "no difference".
- **Carry-forward: plain COUNT.** `clm.rules2k2.carry_forward` picks the lowest-loss candidate among those that are X_BETTER, and plain COUNT if none is. None is.
- **Against the stated expectation (PREREG section 4).** The expectation was that the candidates would land near COUNT, INCONCLUSIVE or NO_MEANINGFUL_DIFFERENCE, with the gated read and the per-order scalar the likeliest to show a small gain. The verdicts match that. No candidate showed a gain. By mean gap the order from nearest to COUNT is gated_low (+0.0147), oscaled_low (+0.0169), scaled_low (+0.0261).
- **Among the candidates (described, no decision).** gated_low against scaled_low is X_BETTER. gated_low against oscaled_low is INCONCLUSIVE. scaled_low against oscaled_low is INCONCLUSIVE. Per-seed values are in the contrast table below. Since no candidate beat COUNT, the X_BETTER here ranks two reads that did not beat COUNT and does not change the carry-forward.

Seed by seed, the candidates against COUNT:

- **Seeds 0, 1 and 4:** all three candidates are worse than COUNT.
- **Seed 2:** gated_low (+0.0208) and scaled_low (+0.0298) are worse. oscaled_low is level with COUNT (-0.0001).
- **Seed 3:** all three candidates are better than COUNT (-0.0217, -0.0103, -0.0118).

Seed 3 is the exception because plain COUNT's seed 3 loss (3.3721) is the highest of its five runs. The other four are 3.2354, 3.2886, 3.2969 and 3.2633. This is not an error. It is the most extreme COUNT run. Seed 3 is also the highest-loss seed for each candidate (3.3503, 3.3618, 3.3602), but by much less than for COUNT. Seed 4 goes the other way: COUNT's seed 4 loss (3.2633) is in the lower half of its five, while the candidates' seed 4 losses are not low (3.3168, 3.3275, 3.3270), so seed 4 has the largest gap for every candidate.

The sample SD of the five plain-COUNT losses is 0.0512. Stage 2K saw about 0.033 on the first three of these runs (seeds 0 to 2). Two more seeds raised it. That is the noise level a 0.01 gap has to be read against, and it is why a gap this size is hard to read.

Final validation loss (nats per token, first 2,097,152 tokens of `data/fineweb/val.bin`, last step). COUNT seeds 0 to 2 are Stage 2K's runs, not rerun. Everything else was trained on q-pc.

| Seed | COUNT | gated_low | scaled_low | oscaled_low |
|---|---|---|---|---|
| 0 | 3.2354 | 3.2489 | 3.2635 | 3.2546 |
| 1 | 3.2886 | 3.2960 | 3.3076 | 3.3023 |
| 2 | 3.2969 | 3.3177 | 3.3267 | 3.2968 |
| 3 | 3.3721 | 3.3503 | 3.3618 | 3.3602 |
| 4 | 3.2633 | 3.3168 | 3.3275 | 3.3270 |
| Mean | 3.2913 | 3.3060 | 3.3174 | 3.3082 |

Paired gap d_s (first arm minus second; positive means the first arm is worse), `clm.analyze.verdict`, SESOI 0.01. The three rows against COUNT are in `results/stage2k2.json`. The three candidate-against-candidate rows are subtracted here from the per-seed losses above; their means match the difference of the stored mean gaps.

| Comparison | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean d | Verdict |
|---|---|---|---|---|---|---|---|
| gated_low minus COUNT | +0.0135 | +0.0074 | +0.0208 | -0.0217 | +0.0535 | +0.0147 | INCONCLUSIVE |
| scaled_low minus COUNT | +0.0281 | +0.0189 | +0.0298 | -0.0103 | +0.0642 | +0.0261 | INCONCLUSIVE |
| oscaled_low minus COUNT | +0.0192 | +0.0136 | -0.0001 | -0.0118 | +0.0637 | +0.0169 | INCONCLUSIVE |
| gated_low minus scaled_low | -0.0146 | -0.0116 | -0.0090 | -0.0114 | -0.0107 | -0.0115 | X_BETTER |
| gated_low minus oscaled_low | -0.0057 | -0.0062 | +0.0209 | -0.0099 | -0.0102 | -0.0022 | INCONCLUSIVE |
| scaled_low minus oscaled_low | +0.0089 | +0.0053 | +0.0299 | +0.0015 | +0.0005 | +0.0092 | INCONCLUSIVE |

Two points on the candidate-against-candidate rows. gated_low against scaled_low is X_BETTER because the mean (-0.0115) is past -0.01 and all five seeds are negative, though one seed (seed 2, -0.0090) is under 0.01 in size. The two INCONCLUSIVE rows against oscaled_low are both set by seed 2, where oscaled_low matches COUNT and the other two reads are worse by 0.0209 and 0.0299.

The paired gaps spread less than the raw losses. The sample SD of the five per-seed gaps against COUNT is about 0.027 for gated_low, 0.027 for scaled_low and 0.029 for oscaled_low (computed here from the per-seed gaps, not stored in the results file). The raw COUNT spread is 0.0512. Pairing helped, but not enough to make a 0.01 gap reliable on five seeds.

## By history class

Classes are set by the stored count of the exact 4-token history, the five classes of Stage 2E. Values are the mean over five seeds (`class_losses_mean`).

| Class | COUNT | gated_low | scaled_low | oscaled_low |
|---|---|---|---|---|
| not stored | 3.3302 | 3.3504 | 3.3642 | 3.3539 |
| 1-9 | 3.1323 | 3.1469 | 3.1598 | 3.1548 |
| 10-99 | 3.2066 | 3.2129 | 3.2189 | 3.2098 |
| 100-999 | 3.3545 | 3.3565 | 3.3677 | 3.3586 |
| >=1000 | 3.4956 | 3.5101 | 3.5144 | 3.5072 |

Gap against COUNT (`class_gaps_vs_count`; positive means the candidate is worse):

| Class | gated_low | scaled_low | oscaled_low |
|---|---|---|---|
| not stored | +0.0202 | +0.0340 | +0.0236 |
| 1-9 | +0.0146 | +0.0275 | +0.0225 |
| 10-99 | +0.0062 | +0.0123 | +0.0032 |
| 100-999 | +0.0020 | +0.0133 | +0.0041 |
| >=1000 | +0.0145 | +0.0188 | +0.0116 |

- Each candidate is worse than COUNT in every class, by 0.0020 to 0.0340.
- The largest gap is in "not stored" for all three candidates (+0.0202, +0.0340, +0.0236). That is the class the absent-history vectors in the gated read were meant to help.
- The smallest gaps are in the middle classes: gated_low in 100-999 (+0.0020), scaled_low in 10-99 (+0.0123), oscaled_low in 10-99 (+0.0032).
- This table averages over all five seeds, including seed 3, where the candidates beat COUNT overall. The class gaps are still all positive.

PREREG section 2 requires the overall loss from `scripts/bucket_eval.py` to match `results.json` within 0.002 for each run. All 20 runs that enter the tables match: the largest difference is 1.04e-8 (read from each run's `bucket.json` on q-pc; it is not stored in `results/stage2k2.json`).

## Gate table, absent norms and learned scalars

Mean of five seeds of gated_low. Orders 2 to 5 are the rows. Columns are the exact count bins 1-9, 10-99, 100-999 and >=1000. The bucket columns stay at 1.0 in every row because no bucket tables were used in this stage, so they are left out. Every gate starts at 1.0.

| Order | 1-9 | 10-99 | 100-999 | >=1000 |
|---|---|---|---|---|
| 2 | 1.12 | 2.01 | 3.90 | 7.65 |
| 3 | 3.93 | 9.89 | 10.46 | 9.91 |
| 4 | 5.38 | 12.06 | 13.00 | 11.67 |
| 5 | 7.27 | 18.26 | 23.58 | 22.62 |

Learned absent-vector norms by order (2 to 5): 8.8, 34.4, 14.6, 19.6.

The gates rise with the history length and, in the first three bins, with the count bin, the same shape as Stage 2E and Stage 2K. Orders 3, 4 and 5 dip a little in the top bin; order 2 keeps rising. Order 5 peaks at 23.58, against about 34 in Stage 2K at the high rate and 90.5 in Stage 2E. Those compare across rates, seed counts and machines, so they are only for scale. The absent norms differ from Stage 2K's (72.3, 47.2, 14.7, 25.8) most at order 2. This note does not explain that.

Learned scalars at the end of training (every scalar starts at 1.0).

scaled_low, the single scalar, by seed:

| Seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| scalar | 10.56 | 9.27 | 8.84 | 8.01 | 8.70 |

oscaled_low, one scalar per order, by seed (orders 2, 3, 4, 5):

| Seed | order 2 | order 3 | order 4 | order 5 |
|---|---|---|---|---|
| 0 | 7.37 | 9.72 | 9.97 | 12.25 |
| 1 | 7.24 | 8.65 | 8.74 | 10.69 |
| 2 | 7.72 | 8.91 | 9.05 | 11.11 |
| 3 | 7.56 | 7.65 | 7.42 | 8.65 |
| 4 | 8.45 | 8.25 | 8.00 | 9.45 |

Every scalar, single or per order, ended between 7.2 and 12.3. The order 5 scalar is the largest in every seed.

## Low rate against high rate

Stage 2K trained the gated and scaled arms at the high rate on seeds 0 to 2. The same seeds are compared here. Negative means the slower rate is better. The oscaled arm has no high-rate counterpart.

| Arm | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| gated, high rate (Stage 2K) | 3.2942 | 3.3365 | 3.3287 | |
| gated, low rate | 3.2489 | 3.2960 | 3.3177 | |
| low minus high | -0.0452 | -0.0405 | -0.0110 | -0.0323 |
| scaled, high rate (Stage 2K) | 3.3418 | 3.3571 | 3.3531 | |
| scaled, low rate | 3.2635 | 3.3076 | 3.3267 | |
| low minus high | -0.0783 | -0.0496 | -0.0263 | -0.0514 |

- The slower rate (8e-3 for the gate parameters) helped in every one of the six pairs, by 0.011 to 0.078. That supports Stage 2K's guess that 64e-3 overshot.
- On the same three seeds the gap to COUNT shrank. For gated it went from a mean of +0.0461 (Stage 2K) to +0.0139. For scaled it went from +0.0770 to +0.0256. The slower-rate arms still do not beat plain COUNT.
- The learned single scalar stayed near 9 at the slow rate too (8.0 to 10.6 in this stage, 8.83 to 9.17 in Stage 2K). So the scalar's size is not only a rate effect. This note makes no claim about why.

## Cost

| | COUNT | gated_low | scaled_low | oscaled_low |
|---|---|---|---|---|
| s per step (q-pc) | 0.341 | 0.350 | 0.343 | 0.343 |
| peak VRAM MiB | 1,609 | 1,611 | 1,609 | 1,609 |
| peak host RSS MiB | 7,222 | 7,427 | 7,220 | 7,219 |

The gated read is about 2.6% slower per step than COUNT, the single scalar about 0.5% and the per-order scalar about 0.6%. Peak VRAM is about 1.6 GB for all four. The 17 runs took about 75 minutes each, about 21 hours of training in total, in line with the PREREG section 6 estimate.

## Reading beyond the rule (interpretation, untested)

These points are not registered tests.

- On seeds 0 to 2 the slower rate closed about two thirds of the gap to COUNT, but did not close it. One reading is that these reads have nothing left to add once plain COUNT is on separate rates (PREREG section 4 raised this). Another is that 8e-3 is still not the right rate for them. This stage cannot separate the two.
- Whether the single scalar's size near 9 is something the model needs, or something it drifts to, was not tested.

Options that were not run, listed plainly:

- A still lower rate for the gate parameters.
- A fixed scale instead of a learned one.
- More seeds.
- A different read.

No option is recommended here. Whether to run any of them is Nate's decision.

## What the verdict does (PREREG section 5)

Plain COUNT carries forward. The gate line closes at this size and rate. Stage 2L (COUNT plus the gained looped model) uses plain COUNT. Nothing here changes gate 1 (Stage 2G).

## Limits

These are PREREG section 8 restated, plus what happened during the stage.

- **One slower rate, not tuned.** 8e-3 was chosen as the Muon rate. A different rate could do better or worse, and this stage cannot say.
- **Best of three.** Three candidate reads were tried against COUNT and P20 asked whether any wins. No correction for three tries was applied. It did not matter here, since none won.
- **Seed spread.** Plain COUNT's seed SD is 0.0512 on five seeds. A rule that needs every seed on the same side resolves 0.01 only if the paired spread is much smaller than that. The paired spread is about 0.027 to 0.029, which is not small enough. An INCONCLUSIVE verdict was a likely outcome, as PREREG section 8 limit 3 said.
- **COUNT seeds 0 to 2 are from Stage 2K**, trained at an earlier commit than the new runs. The COUNT path through `param_groups` is unchanged when no gate rate is set (tested), but those runs were not rerun.
- **One machine, one size, one corpus.** q-pc for all runs; 13.8M non-embedding parameters; FineWeb-Edu.
- **Parameter counts.** The scalars add 1 or 4 parameters and the gate 1,568, negligible against 13.8M.
- **The class evaluation uses 4-token history counts only**, as in Stage 2E.
- **Runs.** All 17 runs finished clean on q-pc with `git_dirty` false. COUNT seeds 0 to 2 were trained at commit `5454f95`; all new runs at `67b7473`. This was read from the `git_commit` and `git_dirty` fields of all 20 `results.json` files on q-pc: COUNT seeds 0 to 2 show `5454f95`, every other run shows `67b7473`, and `git_dirty` is false in all 20. No deviations were logged.
- **Bucket check.** All 20 re-evaluations match `results.json`; the largest difference is 1.04e-8, against a limit of 0.002.

## Files

`PREREG_stage2k2.md` (commit `67b7473`), `src/clm/rules2k2.py`, `src/clm/s2model.py`, `src/clm/s2train.py`, `scripts/stage2k2_driver.py`, `scripts/stage2k2_analyze.py`, `tests/test_s2k2.py`, mutants M186 to M198 (suite 100% and mutation audit 194/194 before the runs), `results/stage2k2.json` (already committed). Run directories: `runs2/count_decoupled_q-M-s` (seeds 0 to 2 from Stage 2K, seeds 3 and 4 new), `runs2/count_gated_low-M-s`, `runs2/count_scaled_low-M-s`, `runs2/count_oscaled_low-M-s`, seeds 0 to 4 each.

TL:DR. We reran the gated COUNT read, a one-number scalar and a four-number per-order scalar with their parameters on the slower 8e-3 rate, five seeds each, against plain COUNT on q-pc. P20 is not supported. All three candidates are INCONCLUSIVE against COUNT (mean gaps +0.0147, +0.0261 and +0.0169, positive meaning worse), which means the seeds disagree, not that the reads tie. Seeds 0, 1 and 4 have every candidate worse than COUNT, seed 2 has two worse and one level, and seed 3 has all three better, because plain COUNT's seed 3 loss (3.3721) is the most extreme of its five runs. COUNT's seed SD is 0.0512, which is why a 0.01 gap is hard to read. Each candidate is worse than COUNT in every history class, most in "not stored". The slower rate helped in all six pairs against Stage 2K's high-rate arms (by 0.011 to 0.078) but did not bring any read past COUNT, and the learned single scalar stayed near 9. By the registered rule plain COUNT carries forward, the gate line closes at this size and rate, Stage 2L uses plain COUNT, and gate 1 is unchanged. A still lower rate, a fixed scale, more seeds or a different read were not tried, and none is recommended.