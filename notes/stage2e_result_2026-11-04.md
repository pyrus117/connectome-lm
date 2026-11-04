# Stage 2E result: the gated COUNT read beats plain COUNT by a hair over the bar, and Kenyon buckets still do not help (2026-11-04)

Pre-registration `PREREG_stage2e.md` (branch `count-gate`, merged into `stage2`, commit c35f661). Code `src/clm/countread.py`, `src/clm/s2model.py` (`B0GPT._gate`), `src/clm/s2train.py` (`--count-gate`), `src/clm/rules2e.py`, `scripts/stage2e_driver.py`, `scripts/stage2e_analyze.py`, `scripts/bucket_eval.py`; config `configs/stage2e.json`; results `results/stage2e.json`. Run on the laptop RTX 3050 (4 GB), GPU memory cap 0.75, from 2026-11-04 10:53 to 20:35 NZDT, six runs. Every run records commit c35f661 and `git_dirty` false. One run was resumed after a forced restart (below and `notes/deviations.md`).

Terms, once. A *history* is the last few tokens of the text so far. The *gated read* gives each n-gram order a learned weight for each (source, count bin), where source is exact history or bucket fallback, plus a learned vector for an order with no stored history. *Cross-entropy* (validation loss) is the model's average surprise per token, in nats; lower is better. *SESOI* is the smallest difference worth caring about, 0.01 nats. *Paired by seed* means each run is compared with the run of the same seed in the other arm.

## Verdict

- **P10 supported.** Gated against COUNT is X_BETTER: mean d = -0.0111, every seed negative. It clears the -0.01 bar by 0.0011.
- **P10a not supported.** The "not stored" gain is -0.0083, short of -0.01 by 0.0017. The no-harm clause holds (no stored class worse by 0.01).
- **P10b not supported.** The ">=1000" gain is -0.0084, short of -0.01 by 0.0016.
- **P11 not supported.** Kenyon-gated minus gated is +0.0037 in "not stored" (needed <= -0.01) and +0.0035 overall (NO_MEANINGFUL_DIFFERENCE).
- **P11s not tested.** The random-gated arm runs only if Kenyon-gated beats gated or P11 holds; neither did (`random_arm: not needed`).
- Five-seed rule: not triggered. Gated against COUNT is X_BETTER, and Kenyon against gated is NO_MEANINGFUL_DIFFERENCE, so neither is INCONCLUSIVE. Seeds 3 and 4 not run.

Final validation loss (nats/token, first 2,097,152 tokens of `data/fineweb/val.bin`, last step):

| Seed | COUNT (Stage 2A runs, not rerun) | COUNT-gated | Kenyon-gated |
|---|---|---|---|
| 0 | 3.3221 | 3.3145 | 3.3175 |
| 1 | 3.3206 | 3.3075 | 3.3138 |
| 2 | 3.3300 | 3.3173 | 3.3186 |

Paired gap d_s (first arm minus second; negative means the first arm is better), `clm.analyze.verdict`, SESOI 0.01:

| Comparison | seed 0 | seed 1 | seed 2 | mean d | Verdict |
|---|---|---|---|---|---|
| gated minus COUNT | -0.0075 | -0.0131 | -0.0127 | -0.0111 | X_BETTER |
| Kenyon-gated minus gated | +0.0030 | +0.0063 | +0.0013 | +0.0035 | NO_MEANINGFUL_DIFFERENCE |
| Kenyon-gated minus COUNT | -0.0046 | -0.0068 | -0.0114 | -0.0076 | NO_MEANINGFUL_DIFFERENCE |

The P10 margin is thin: seed 0 alone is -0.0075, under the bar, and the mean clears -0.01 by about one tenth of the SESOI. The registration called this outcome likely to land near the SESOI, and it did.

`scripts/bucket_eval.py` overall loss matches `results.json` in all six runs (largest difference 1.1e-7, limit 0.002).

## By history class

Classes are set by the stored count of the exact 4-token history. Mean over three seeds, with B0 from `runs2/b0-M-s/bucket.json`:

| Class | B0 | COUNT | gated | Kenyon-gated |
|---|---|---|---|---|
| not stored | 3.3395 | 3.3672 | 3.3589 | 3.3626 |
| 1-9 | 3.3823 | 3.1600 | 3.1515 | 3.1563 |
| 10-99 | 3.4589 | 3.2346 | 3.2185 | 3.2211 |
| 100-999 | 3.4903 | 3.3846 | 3.3638 | 3.3661 |
| >=1000 | 3.5050 | 3.5324 | 3.5240 | 3.5269 |

Gap to COUNT (negative means better than COUNT):

| Class | gated minus COUNT | Kenyon-gated minus COUNT | Kenyon-gated minus gated |
|---|---|---|---|
| not stored | -0.0083 | -0.0046 | +0.0037 |
| 1-9 | -0.0085 | -0.0037 | +0.0048 |
| 10-99 | -0.0161 | -0.0135 | +0.0026 |
| 100-999 | -0.0208 | -0.0185 | +0.0024 |
| >=1000 | -0.0084 | -0.0055 | +0.0030 |

- The gain is in every class, largest in the stored middle classes (10-99, 100-999), not in the two classes the design aimed at. Those two classes lose against B0 by 0.0195 ("not stored") and 0.0190 (">=1000") with the gate, against 0.0278 and 0.0274 for COUNT. The gate removed about 30% of each penalty; Section 4 of the registration allowed for a full removal worth about 0.014 overall.
- Kenyon-gated is worse than gated in every class, by 0.0024 to 0.0048, so the buckets cost a little even with the gate free to turn them down.

## Gate tables (mean of three seeds; order 2 to 5 is rows 0 to 3; columns are exact bins 1-9, 10-99, 100-999, >=1000, then bucket bins in the same order)

Gated: exact gates grew far above their start of 1.0: order 2 at 1.3, 4.5, 11.0, 13.7; order 3 at 7.9, 22.2, 24.3, 24.4; order 4 at 13.3, 34.6, 37.7, 35.8; order 5 at 22.2, 68.6, 90.5, 80.4. Gates rise with the history length and with the count bin, flattening above the 10-99 bin at orders 3 and 4. Absent-vector norms by order: 36.2, 30.0, 7.9, 17.7.

Kenyon-gated: exact gates are close to the gated arm's (order 5: 20.6, 63.3, 83.9, 75.4). Bucket gates, orders 3 to 5: 1.05 to 5.05, and relative to the mean exact gate the bucket gates sit at 0.06 to 0.29, so buckets are read at about a tenth to a quarter of exact weight, near Nate's expectation of 0.1. Order 2 has no bucket slots (those gates stay at 1.0 and are left out). Absent-vector norms: 36.9, 43.0, 7.1, 14.5.

The size of the exact gates (13 to 90, against 1.0 at the start) means the memory is being turned up against the token embedding by a large factor. That is the scale freedom of limit 1 in the registration, and the result cannot separate it from structure.

## Cost

| | COUNT | gated | Kenyon-gated |
|---|---|---|---|
| mean s/step | 0.404 | 0.464 | 0.446 |
| mean lookup s | 0.046 | 0.052 | 0.107 |
| peak VRAM MiB | 1,593 | 1,683 | 1,594 |
| peak host RSS MiB | 7,406 | 7,918 | 9,694 |

The gated read is about 15% slower per step than COUNT.

## Restart

The laptop was restarted by Nate at 18:28 while `count_gated-M-2` was at about step 9,257. It resumed from its step-9000 checkpoint. Details, and what was done with the log, are in `notes/deviations.md` and `runs2/count_gated-M-2/RESTART.txt`. Not bit-identical to an uninterrupted run; the effect is at the level of ordinary run-to-run noise, well below 0.01.

## Reading (interpretation, not a measured result)

- The gate helps, but only just over the bar, and for a reason other than the one registered. The absent vector did not rescue the "not stored" class, and the ">=1000" penalty remains. The largest gains sit in classes with stored counts.
- Two explanations, untested and not separable from these runs: the gain is the scale freedom of limit 1 (a larger overall memory signal), or it is a real count-dependent weighting. The scale-only control in section 6 of the registration separates them.
- Bucket fills still do not pay at this size, with or without a gate. The buckets add nothing the exact slots and the gate do not already give.

## What the verdict does (PREREG section 6)

- P10 X_BETTER: the gated read becomes the COUNT read of later stages, and the next test is the scale-only control (one global scalar gate, no per-order or per-bin split, no absent vector). Whether to open gate 3 (a size-L check) with the gated read goes to Nate with these numbers.
- P11 not supported: the bucket line is closed at this model size. Multi-probe Kenyon reading (Avenue B) is not built.
