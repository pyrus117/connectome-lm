# Stage 2B pre-registration: the Kenyon backoff inside the network

Dated 2026-11-03. Committed before any Stage 2B run. Plan: `docs/superpowers/plans/2026-11-02-stage2-kenyon-memory.md` (§7.3, §9 P4, §11). This stage runs only if Stage 2B-0's gate 2 passed (`PREREG_stage2b0.md` §4) and Stage 2A's gate 1 passed (`PREREG_stage2a.md` §5). It is registered now so that nothing about it can drift toward a result.

Code: `src/clm/countread.py` (bucket fallback), `src/clm/kenyon.py`, `src/clm/s2train.py`, `scripts/build_bucket_tables.py`, `scripts/stage2b_driver.py`, `scripts/stage2b_analyze.py`, `scripts/bucket_eval.py`. Config `configs/stage2b.json`. Branch `stage2b0`.

## 1. Question

**P4.** Put the Kenyon backoff into the COUNT memory. Does COUNT + Kenyon beat COUNT? And is the gain where the idea says it should be, on histories with no exact counts, and absent for a plain hash that pools unrelated histories?

In Stage 2A, COUNT was compared with B0. Here the baseline is COUNT itself. A history is the last few tokens of the text so far. Cross-entropy is the model's average surprise per token, in nats.

## 2. Fixed design

- **Arms.** All three are COUNT-M with the Stage 2A read and recipe: Muon at lr 0.032 (chosen by Stage 2A's pilot), weight decay 0.1, warmup 200 steps then linear decay to zero, clip 1.0, bf16, GPU memory cap 0.75, context 512, 200M tokens (12,207 steps of 32 x 512). Same data order and same name-keyed initialisation per seed, so every tensor starts identical across arms and a gap comes from the memory, not from a lucky start. Leave-one-out reads while training, as in Stage 2A. Seeds 0, 1, 2.
  - (a) **COUNT**: the Stage 2A runs `runs2/count-M-0`, `-1`, `-2`. Not rerun, so the baseline cannot be nudged after the Kenyon results are known.
  - (b) **COUNT + Kenyon**: `runs2/count_kenyon-M-s`.
  - (c) **COUNT + random-bucket control**: `runs2/count_random-M-s`.
- **The Kenyon backoff in the read.** Orders 3, 4 and 5 (histories of 2, 3 and 4 tokens) have bucket tables. At each of those orders, if the exact history is not stored for this position, the order's slot holds the first K = 8 stored continuations of the history's bucket instead of staying empty. Each score is C(bucket, w) / C(bucket), divided by J = 4, the same form as an exact row. Exact histories keep their own counts.
  - "Not stored" means never seen, under the minimum history count (c_min 2 at orders 4 and 5), or dropped by leave-one-out.
  - A history containing end-of-text gets no bucket.
  - The 1-token order is exact only.
  - Why this form: it changes one thing from Stage 2A. An empty slot becomes a filled one, and nothing else moves.
- **Leave-one-out for buckets.** During training the position's own occurrence is subtracted from the bucket's continuation count and total. The bucket row is dropped if its total reaches 0. Validation uses no correction. Why: the count corpus contains the training stream, so a bucket's counts include the position's own next token, which validation text never gets. This was tested against a brute-force reference.
- **The hash (the fly's Kenyon cells).** The one Stage 2B-0's sweep selected by its pre-registered rule: PPMI token codes (64-d), dense random projection into two sub-codebooks of 2,048 cells, 2 winners per sub-codebook, homeostatic thresholds on. Two winners had the lowest rare-history held-out loss in the sweep. The hash is rebuilt deterministically from Stage 2B-0's final-run codes and calibration sample (`scripts/build_bucket_tables.py`) and saved with the tables.
- **Random-bucket control.** It hashes the token ids of the history (splitmix64) into 4,194,304 buckets, so it pools unrelated histories. It shows what pooling alone does.
- **Bucket tables.** Built over the whole 881,048,239-token count corpus: all histories of 2, 3 and 4 tokens, including those seen once. Top 16 continuations per bucket with the bucket's total. Bucket rows are kept at any count.
- **Metric.** As in Stage 2A: mean cross-entropy on the first 2,097,152 tokens of `data/fineweb/val.bin` at the last step. Also recorded: wall-clock per step, peak VRAM, peak host RSS, mean lookup seconds per step.
- **Class evaluation.** After the runs, `scripts/bucket_eval.py` re-evaluates each final checkpoint by the exact 4-token-history count class:

| Class | Meaning |
|---|---|
| not stored | never seen, or under c_min |
| 1-9 | |
| 10-99 | |
| 100-999 | |
| >=1000 | |

  Its overall loss must match `results.json` within 0.002, which checks that the re-evaluation reads the model the same way training did.

## 3. Procedure

Runs go under `systemd-run --user` with the GPU memory cap 0.75. `embeddinggemma` and `heretic-harness` are stopped for the night and restarted after, as before (approved by Nate 2026-11-02). Order: seed 0 Kenyon, seed 0 random, seed 1 Kenyon, seed 1 random, seed 2 Kenyon, seed 2 random, then analysis.

Before the first run, a smoke run of a few hundred steps checks that the bucket lookups keep up with the GPU step. If the lookup time exceeds the step time, the prefetch is restructured and the change is logged in `notes/deviations.md` before any registered run. Why: a lookup that stalls the GPU would make the cost numbers wrong, and a fix made after a registered run would be a change made with results in view.

## 4. Reference numbers from Stage 2B-0 (offline n-gram models, not decision inputs)

On held-out text at a 100M-token universe, averaged over positions whose 4-token history is not stored or seen 1-9 times, mixing in buckets lowered cross-entropy against the exact model alone by:

| Bucket type | Gain (nats) |
|---|---|
| Kenyon, two winners | 0.0248 |
| Kenyon, default (one winner) | 0.0205 |
| Kenyon from random codes | 0.0043 |
| Random buckets | 0.0007 |

These are n-gram models, not the network. The rare classes cover about two thirds of 4-token histories, so the expected gain for the network is of the order of 0.01 nats or less. That is why a small effect is registered as likely.

## 5. Decision rule

Paired by seed, as in Stage 2A: d_s = loss(COUNT + Kenyon) - loss(COUNT), computed by `clm.analyze.verdict` (the Stage 1 rule). SESOI 0.01 nats. Pairing by seed removes the noise the arms share.

- **X_BETTER:** mean d <= -0.01 and every d_s < 0.
- **X_WORSE:** mean d >= +0.01 and every d_s > 0.
- **NO_MEANINGFUL_DIFFERENCE:** |mean d| < 0.01 and every |d_s| < 0.02.
- **INCONCLUSIVE:** anything else. If |mean d| < 0.02 the five-seed rule applies: seeds 3 and 4 are added for COUNT, COUNT + Kenyon and COUNT + random-bucket (six more runs), and the rule is applied once more on five triples.

Also computed by the same rule: random-bucket against COUNT, and Kenyon against random-bucket.

**P4 is supported** if and only if, with mean losses over seeds in the 4-token-history classes:

- in each of "not stored" and "1-9", (Kenyon - COUNT) <= -0.01 and (Kenyon - random-bucket) <= -0.01; and
- in each of "100-999" and ">=1000", |Kenyon - COUNT| < 0.01.

Why the second part: the idea predicts a gain only where exact counts are missing, so a gain everywhere would point at something else. Why the first part compares with random buckets: a gain that a plain hash also gets would come from pooling, not from the fly-style similarity.

The verdict on the overall loss and the P4 reading are separate results. Either can come out without the other, and both are written up.

## 6. What a verdict does

Any outcome is written up with the per-class table. If Kenyon is X_BETTER against COUNT, the plan's gate 3 (a size-L check and table scaling) becomes available and goes to Nate with the numbers. Otherwise the work continues with Stage 2C and 2D, which do not depend on this result.

## 7. Conduct

A diverged run (non-finite loss or gradient) is rerun once from scratch. A second divergence is reported and gives no verdict. No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash. This stops the design drifting toward a result.

After any change to `src/`, the suite and `scripts/mutation_audit.py` must pass. The bucket fallback carries planted-bug mutants M75, M76, M66, M86 and M89 in `scripts/mutation_audit.py`, on top of Stage 2A's leave-one-out mutants M37 to M39: a row dropped by leave-one-out still blocking its bucket, a bucket overwriting an exact row, a flipped leave-one-out subtraction, among others. A test that cannot catch those bugs does not count.

## 8. Limits stated in advance

- The hash settings were chosen at a 100M-token universe and one projection seed. The tables here are built at 881M.
- The network may use the buckets differently from the offline mixture. Here the bucket scores fill an empty slot of COUNT's average. Offline, they were mixed by weights fit on held-out text.
- COUNT's learning rate sat at the top of its pilot grid (the pilot picked 0.032, the extension of 0.016), so the COUNT baseline may be under-tuned. All three arms share that rate, so the comparison stays fair, but the absolute losses may not be the best COUNT can do.
- One model size (13.8M non-embedding parameters), one corpus.
- A gain of 0.01 nats is at the SESOI. Three seeds can only just resolve it.

TL:DR. We test whether filling the empty n-gram slots of COUNT with fly-style Kenyon bucket counts lowers validation loss by at least 0.01 nats against COUNT itself, over three paired seeds, with a random-bucket arm as the control for pooling alone. P4 is supported only if the gain appears in the "not stored" and "1-9" history classes, beats the random buckets there, and is absent in the common classes. The offline result suggests a gain of about 0.01 nats or less, so a small effect is the expected case. Everything is fixed before any run. A gain against COUNT opens gate 3. Otherwise the work moves on to Stages 2C and 2D.
