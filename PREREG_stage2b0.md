# Stage 2B-0 pre-registration: an offline test of the Kenyon backoff

Dated 2026-11-03. Committed before any Stage 2B-0 run. Plan: `docs/superpowers/plans/2026-11-02-stage2-kenyon-memory.md` (§7.3, §9 P3 P5 P6, §11 gate 2). CPU only: no network is trained. This stage runs only if Stage 2A's gate 1 passed (`PREREG_stage2a.md` §5). It is registered now so that its settings cannot drift toward a result.

## 1. Questions

The idea comes from the fruit fly (Dasgupta, Stevens & Navlakha 2017; Litwin-Kumar et al. 2017). Its mushroom body takes an odour and sends it through a fixed random expansion into many Kenyon cells. Only the strongest cells stay active. That rule is k-winners-take-all: the k cells with the largest drive win and all others are silenced, here by a global inhibition shared across the cells. Similar odours switch on the same winners, so they land in the same bucket.

A language model's exact n-gram memory has nothing to say about a history it never saw. 52% of validation 4-token histories and 20% of 3-token histories never occur in the 200M training tokens (`notes/ngram_stats_2026-11-02.md`). The **Kenyon backoff** gives such a history the pooled continuation counts of the histories in its bucket. A history is the last few tokens of the text so far, and a continuation is a token that followed it.

Three predictions from the plan:

- **P3.** Adding Kenyon buckets lowers cross-entropy on unseen and rare (1-9) histories more than random buckets do. Cross-entropy is the model's average surprise per token, in nats.
- **P5.** Bucket quality improves with the number of inputs per cell, K, and then levels off. A dense cell (every input) is at least as good as K = 7 (Litwin-Kumar 2017 with global inhibition; Senge 2026, arXiv 2610.09114).
- **P6.** Without the homeostatic thresholds, some cells get used far more than others, and bucket quality is worse.

**Gate 2** (plan §11): the network arm 2B runs only if the bucket component lowers offline cross-entropy in the unseen and 1-9 history buckets by at least 0.02 nats more than random buckets do. Otherwise 2B is replaced by a write-up of this offline result.

## 2. Fixed design

Code: `src/clm/kenyon.py`, `src/clm/kenyon_lm.py`, `src/clm/ngram_lm.py`, `src/clm/counts.py`, `scripts/stage2b0.py`. Branch `stage2b0`.

- **Count corpus.** The 881,048,239-token training-split corpus of Stage 2A (`data/fineweb/count_corpus.bin`, end-of-text id 0). **Held-out slice H** is its last 10,000,000 tokens, cut at a document boundary, and no count includes it. Validation text (`data/fineweb/val.bin`) is never in any count. **Universe U** is the set of documents whose counts are used: the first about 100M tokens for the sweep, every token before H (about 871M) for the final run. H exists so that settings and weights can be fit on text that is not validation.
- **Exact model.** The `clm.ngram_lm.NgramLM` of Stage 2A. It uses absolute discounting (subtract a fixed 0.75 from every seen count and hand that freed probability to the shorter history), back-off to shorter histories, an add-one unigram, and is normalised. It runs over exact tables for histories of 1 to 4 tokens built from U, with minimum history counts 1, 1, 2, 2 as in Stage 2A and the top 16 continuations per history. Histories use only tokens inside the 512-token window and never contain end-of-text.
- **Token codes** (the fly's receptor neurons). 64-dimensional and frozen. Built in two steps from U. First PPMI (positive pointwise mutual information): for each pair of tokens, how much more often they appear within +-2 places of each other than chance predicts, with negative values set to zero, context smoothing 0.75, and end-of-text pairs excluded. Then truncated SVD (`torch.svd_lowrank`, q = 80, 4 iterations, seed 0), which squeezes that table down to its 64 strongest directions. Code = U times sqrt(S) on the first 64 components. The control uses random Gaussian codes of the same size (**KM-rand**): it keeps the hash and drops the similarity, so tokens that mean alike no longer get alike codes.
- **Kenyon hash.** One per history length 2, 3 and 4. A 1-token history is its own bucket.
  - A history's odour is its tokens' codes concatenated, centred by the frequency-weighted mean code and L2-normalised (scaled to length 1).
  - A fixed random projection (Gaussian, seed 0) expands it into two sub-codebooks of 2,048 cells each.
  - After subtracting per-cell thresholds, the winner of each sub-codebook names the bucket. One winner per sub-codebook gives 4,194,304 possible buckets.
  - Because the odour is a concatenation, a cell's drive is a sum over history positions of a per-token table, and the odour's norm is the root of the sum of per-token squared norms. The code computes it that way, which avoids building every odour, and a test checks it against the explicit projection.
  - Default settings: dense Gaussian projection, one winner per sub-codebook, thresholds on.
- **Homeostatic thresholds** (the fly's intrinsic plasticity; the same idea as DeepSeek's auxiliary-loss-free bias rule). Per-cell thresholds start at 0. For 100 steps, a cell that wins more often than its share (winners / 2048 of positions) has its threshold raised by 0.03 times the standard deviation of the drive, and a cell that wins less often has it lowered. The rule runs on 40,000 histories sampled at random positions of U, so frequent tokens are as common as in real text. Cell-use balance is reported as the max share and as entropy as a fraction of the maximum.
- **Bucket tables.** For each history length 2, 3, 4: the continuation counts of every distinct history in U (minimum count 1, so histories seen once are included) pooled by bucket, keeping the top 16 continuations per bucket plus the bucket's total. A history containing end-of-text gets no bucket.
- **Random-bucket control.** Identical, except the bucket is a plain hash of the history's token ids (splitmix64, a standard integer-mixing hash) into the same 4,194,304 buckets. It pools unrelated histories, so it shows what pooling alone does.
- **Bucket model.** The exact model with its 2-, 3- and 4-token tables replaced by bucket tables. The 1-token table and the unigram stay exact.
- **Mixture.** p = (1 - lambda_c) p_exact + lambda_c p_bucket. There is one weight lambda_c for each class c of the position's exact 4-token history, by its stored count:

| Class | Meaning |
|---|---|
| not stored | never seen, or seen once (the 4-token table keeps no history seen once) |
| 1-9 | stored counts 2 to 9 |
| 10-99 | |
| 100-999 | |
| >=1000 | |

  A sixth class (window too short to hold a 4-token history) is never mixed. The weights are fit by EM (repeated re-weighting until the mixture's likelihood stops improving) on H, never on validation.
- **Scoring.** The 4,096 fixed windows of 512 (first 2,097,152 tokens) of H and of validation.

## 3. Sweep

`scripts/stage2b0.py sweep`. Universe about 100M tokens. It reads H only and no validation.

Why a smaller universe: each configuration needs a pass over the whole corpus. The sweep chooses settings and the final run confirms them at the real scale.

One factor changes at a time around the default (PPMI codes, dense, one winner, thresholds on):

- inputs per cell K in {1, 3, 7, 15, 31} (each cell then connects to K random dimensions of the odour; dense is the default);
- winners per sub-codebook in {2, 4};
- thresholds off;
- random codes.

The random-bucket control is run as well. For each configuration the sweep records the mixture's cross-entropy on H by class, the fitted weights, cell-use balance and bucket-use entropy.

**Selection rule (fixed here).** Among Kenyon configurations with PPMI codes, take the one with the lowest mixture cross-entropy on H, averaged over positions whose 4-token history is not stored or has stored count 1-9. Call that the **rare loss**. The random-codes configuration is a control and is not eligible. The choice is written to `results/stage2b0_sweep.json` before any validation score exists.

## 4. Final run

`scripts/stage2b0.py final`. Universe about 871M.

The selected configuration, its random-bucket control and its random-codes control are each scored on validation, with weights fit on H.

**Gate 2.** For each of the classes "not stored" and "1-9" of the 4-token history, compute

(exact loss minus Kenyon-mixture loss) minus (exact loss minus random-bucket-mixture loss).

It must be at least 0.02 nats in both classes. PASS or FAIL is written to `results/stage2b0_final.json`.

Also reported: the three models' losses in all classes, and how much of the Kenyon gain survives with random codes. If the random-codes control gains as much as Kenyon, similarity is not the source of the gain.

## 5. Reading P3, P5, P6

- **P3** is supported if and only if gate 2 passes.
- **P5** is supported if and only if, in the sweep, the rare loss at K = 31 is at least 0.005 below that at K = 1, and the dense default's rare loss is no more than 0.002 above that at K = 7.
- **P6** is supported if and only if, with thresholds off, the entropy fraction of cell use is at least 0.02 lower than with thresholds on (averaged over the three history lengths and both sub-codebooks), and the rare loss is at least 0.002 higher.

All three are written up whatever they show.

## 6. What a verdict does

Gate 2 PASS: Stage 2B (the network with the Kenyon backoff, plus the random-bucket arm, 3 seeds at M) is registered and run with the selected settings fixed.

Gate 2 FAIL: 2B is replaced by a write-up of this offline result, and the work goes on to Stage 2C with COUNT alone.

## 7. Conduct

CPU only, under `systemd-run --user` with memory and CPU limits (4 threads), so it does not disturb a GPU run. No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash. This stops the design drifting toward a result.

After any change to `src/`, the suite and `scripts/mutation_audit.py` must pass. This stage carries mutants M62 to M74 (a flipped threshold update, a missing norm, a history with end-of-text given a bucket, mixture weights swapped, short windows counted as unseen, among others). A test that cannot catch those bugs does not count.

## 8. Limits stated in advance

- Settings are chosen at a 100M-token universe and confirmed at 871M, which is a different table density.
- There is one random projection seed (0) and one calibration sample, so seed-to-seed variation in bucket quality is not measured. Differences under about 0.005 nats between sweep rows should not be read as real.
- The offline mixture is not the network. In 2B the buckets' scores would join COUNT's average, and the network can use them differently.
- Weights and settings are fit on the same slice H, with 5 weights and 10 configurations, so a small optimism in H losses is expected. It is the same for every configuration.
- The universe differs from the real tables only by removing H.
- 52% of validation 4-token histories are unseen in the 200M stream, but fewer are unseen against 871M. The final run measures this at the real density.

TL:DR. We test, on CPU and with no network trained, whether giving an unseen or rare n-gram history the pooled counts of its fly-style Kenyon bucket lowers loss more than pooling by a random hash. Every setting, the tuning sweep on a 100M-token universe, the selection rule and the 871M-token final run are fixed before any run. Gate 2 passes only if the Kenyon gain over random buckets is at least 0.02 nats in both the "not stored" and "1-9" history classes on validation. A pass registers the network arm 2B. A fail replaces 2B with a write-up and moves on to 2C with COUNT alone. P3, P5 and P6 are written up whatever they show.
