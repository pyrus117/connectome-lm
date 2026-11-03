# Stage 2B result: Kenyon buckets make no meaningful difference to the network, and P4 is not supported (2026-11-04)

Pre-registration `PREREG_stage2b.md` (branch `stage2b0`). Code `src/clm/countread.py`, `src/clm/kenyon.py`, `src/clm/s2train.py`, `scripts/build_bucket_tables.py`, `scripts/stage2b_driver.py`, `scripts/stage2b_analyze.py`, `scripts/bucket_eval.py`; config `configs/stage2b.json`; results `results/stage2b.json`. Run on the laptop RTX 3050 (4 GB), GPU memory cap 0.75, from 2026-11-03 23:51 to 2026-11-04 08:21 (8 h 30 min). This stage was **run despite a failed gate**: gate 2 failed in Stage 2B-0 (`notes/stage2b0_result_2026-11-03.md`), and Nate asked for the runs anyway. The amendment is in `PREREG_stage2b.md` and `notes/deviations.md`. Nothing else in the registration changed.

Terms, once. A *history* is the last few tokens of the text so far. A *bucket* is a group of histories that share one table row; the Kenyon hash (modelled on the fly's Kenyon cells) groups histories that look alike, and the random-bucket control groups them by chance. *Cross-entropy* (validation loss) is the model's average surprise per token, in nats; lower is better. *SESOI* is the smallest difference worth caring about, here 0.01 nats. *Paired by seed* means each Kenyon run is compared with the COUNT run that has the same seed, so noise the two share cancels out.

## Verdict: NO_MEANINGFUL_DIFFERENCE on all three comparisons. P4 is NOT supported.

The question (P4): if the empty n-gram slots in COUNT are filled with Kenyon bucket counts, does validation loss fall by at least 0.01 nats against COUNT, and does the gain sit on histories with no exact counts and beat a plain hash that pools unrelated histories?

No. The Kenyon arm finished slightly worse than COUNT, not better, and by less than the SESOI. The random-bucket arm did the same.

Final validation loss (nats/token, first 2,097,152 tokens of `data/fineweb/val.bin`, last step):

| Seed | COUNT (Stage 2A runs, not rerun) | COUNT + Kenyon | COUNT + random buckets |
|---|---|---|---|
| 0 | 3.3221 | 3.3282 | 3.3284 |
| 1 | 3.3206 | 3.3253 | 3.3240 |
| 2 | 3.3300 | 3.3330 | 3.3322 |

Paired gap d_s (first arm minus second; positive means the first arm is worse), computed by `clm.analyze.verdict` with SESOI 0.01:

| Comparison | seed 0 | seed 1 | seed 2 | mean d | Verdict |
|---|---|---|---|---|---|
| Kenyon minus COUNT | +0.0062 | +0.0047 | +0.0030 | +0.0046 | NO_MEANINGFUL_DIFFERENCE |
| random minus COUNT | +0.0064 | +0.0034 | +0.0022 | +0.0040 | NO_MEANINGFUL_DIFFERENCE |
| Kenyon minus random | -0.0002 | +0.0013 | +0.0008 | +0.0007 | NO_MEANINGFUL_DIFFERENCE |

The rule (`PREREG_stage2b.md` section 5): NO_MEANINGFUL_DIFFERENCE needs |mean d| < 0.01 and every |d_s| < 0.02. All three comparisons meet both parts. The five-seed extension applies only to an INCONCLUSIVE result, so it does not apply here and seeds 3 and 4 are not run.

## P4, by history class

The rule for P4: in each of "not stored" and "1-9", (Kenyon minus COUNT) <= -0.01 and (Kenyon minus random-bucket) <= -0.01; and in each of "100-999" and ">=1000", |Kenyon minus COUNT| < 0.01. Classes are set by the stored count of the exact 4-token history. Mean loss over three seeds:

| Class | COUNT | Kenyon | random buckets |
|---|---|---|---|
| not stored | 3.3672 | 3.3757 | 3.3745 |
| 1-9 | 3.1600 | 3.1610 | 3.1601 |
| 10-99 | 3.2346 | 3.2368 | 3.2365 |
| 100-999 | 3.3846 | 3.3860 | 3.3865 |
| >=1000 | 3.5324 | 3.5336 | 3.5337 |

Gap to COUNT (positive means worse than COUNT):

| Class | Kenyon minus COUNT | random minus COUNT |
|---|---|---|
| not stored | +0.0085 | +0.0073 |
| 1-9 | +0.0010 | +0.0001 |
| 10-99 | +0.0022 | +0.0019 |
| 100-999 | +0.0014 | +0.0019 |
| >=1000 | +0.0012 | +0.0013 |

The first part of P4 fails. In "not stored", the one class where the idea predicts a gain, both bucket arms are worse than COUNT, and by about the same amount. Kenyon is 0.0013 worse than random buckets there (3.3757 against 3.3745), not 0.01 better. In "1-9" Kenyon is 0.0010 worse than COUNT. The second part holds: in "100-999" and ">=1000" the gaps are 0.0014 and 0.0012, well inside 0.01. So the loss moved the wrong way where a gain was predicted, and barely moved elsewhere.

Kenyon gains nothing over random buckets. The offline similarity effect from 2B-0 (random buckets gained 0.0001, Kenyon 0.0116) left no mark in the network.

`scripts/bucket_eval.py` must match `results.json` overall within 0.002 (section 2), which checks that the class evaluation reads each model the way training did. It does: across all nine runs (and the no-leave-one-out diagnostic) the largest difference is 2.7e-7 nats (Kenyon seed 1).

## Reading (interpretation, not a measured result)

The offline n-gram gain of 0.0116 nats in 2B-0 (the gate 2 near-miss) did not carry into the network. Two possible reasons, neither tested:

- The network already gets most of what pooled counts give from its exact slots, so the buckets add little.
- In COUNT, an empty slot is a signal the model has learned to read as "no information here". A filled bucket slot replaces that signal with counts pooled from other histories, and the loss of the clean signal may cost more than the pooled counts give.

The cause is not established. Nothing in these runs separates the two.

## What the verdict does (PREREG section 6)

Kenyon is not X_BETTER against COUNT, so the plan's gate 3 (a size-L check and table scaling) is not triggered by Kenyon. Stages 2C and 2D do not depend on this result. Whether to open gate 3 on COUNT alone is a separate decision for Nate.

## Cost

| | COUNT | COUNT + Kenyon | COUNT + random buckets |
|---|---|---|---|
| seconds per step | 0.4044 | 0.3993 | 0.3984 |
| mean lookup seconds per step | 0.0456 | 0.0886 | 0.0311 |
| peak VRAM (MiB) | 1593 | 1593 | 1593 |
| peak host RSS (MiB) | 7,406 | 9,376 | 8,838 |

Kenyon to COUNT time ratio 0.987. The bucket lookups cost nothing in step time because a prefetch (reading the next batch's table rows while the GPU works on the current one) hides them. Before the runs a 400-step smoke run of the random-bucket arm gave a step of 0.402 s and a median lookup of 0.085 s, hidden by the prefetch; the first ~10 steps are cold-cache reads of up to 10 s. Kenyon adds about 2 GiB of host memory (peak RSS 9,376 MiB against 7,406 MiB).

The Kenyon bucket tables (`data/count_kenyon`) took 1 h 2 min to build: 1.14M, 2.83M and 3.58M buckets at 2-, 3- and 4-token histories. The random-bucket tables (`data/count_randbucket`) were built earlier on 2026-11-03, during 2B-0.

## Conduct

- Six runs, each 200M tokens at size M, Muon lr 0.032: Kenyon arm `runs2/count_kenyon-M-{0,1,2}`, random-bucket arm `runs2/count_random-M-{0,1,2}`. The COUNT baseline is the Stage 2A runs, not rerun.
- No divergence. No change to code or config after `PREREG_stage2b.md`, so there is no new entry in `notes/deviations.md` for this stage beyond the amendment above.
- Before the runs, the test suite (all pass, 1 skipped) and `scripts/mutation_audit.py` (M1 to M89 all killed; the two required control cases behaved as designed) were rerun on the identical code. The first audit launch lost its output, so it was rerun with logs while training started.

## Limits

From `PREREG_stage2b.md` section 8, plus one carried over from 2B-0:

- The hash settings were chosen at a 100M-token universe and one projection seed. The tables here were built at 881M tokens.
- The network may use the buckets differently from the offline mixture. Here a bucket fills an empty slot of COUNT's average; offline, the buckets were mixed by weights fit on held-out text.
- COUNT's learning rate sits at the top of its pilot grid (the pilot picked 0.032, the extension of 0.016), so COUNT may be under-tuned. All three arms share the rate, so the comparison stays fair, but the absolute losses may not be the best COUNT can do.
- One model size (13.8M non-embedding parameters), one corpus.
- Three seeds resolve a difference of about 0.01 nats and no finer.
- The Kenyon setting is K31 (31 inputs per cell), selected in place of the registered winners2 because winners2's tables cannot be built on this machine (`notes/deviations.md`, 2B-0). A table that can be held is a pooled one, and the offline gain for it was smaller than for winners2.

TL:DR. Run despite a failed gate, Stage 2B found no meaningful difference: Kenyon was +0.0046 nats worse than COUNT, random buckets +0.0040 worse, and Kenyon beat random buckets by nothing (+0.0007), all under the 0.01 SESOI. In "not stored" histories, where a gain was predicted, both bucket arms were worse than COUNT by about the same amount, so P4 is not supported. Kenyon does not trigger gate 3; gate 3 on COUNT alone is Nate's decision.
