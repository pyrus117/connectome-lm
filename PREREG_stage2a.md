# Stage 2A pre-registration: COUNT, a counted n-gram memory read through the unembedding

Dated 2026-11-03. Committed before the first Stage 2A run. Its sha256 is recorded in every Stage 2A `results.json`. Plan: `docs/superpowers/plans/2026-11-02-stage2-kenyon-memory.md` (§7.2, §11). Notes of step 1 (the count tables): `notes/stage2a_count_table_2026-11-03.md`.

## 1. Question

Does adding COUNT (Magnusson, Smith & Koh 2026, arXiv 2610.20456, Eq. 1-2) to B0-M lower validation loss, at equal tokens, by at least the SESOI of 0.01 nats?

COUNT is a lookup table of n-gram counts kept in host RAM. For the last few tokens of the text so far, it finds which tokens most often came next in the training text. It takes the model's own unembedding rows for those tokens, weights each row by how often that token followed, and adds the result to the token embedding. It adds no learned parameters.

Gate 1 of the plan: if COUNT is not X_BETTER than B0, the memory track (2B-0, 2B, most of 2C) stops and the null is written up. 2D (looped depth) still runs.

## 2. Fixed design

- **Baseline arm B0-M.** The frozen Stage 2.0 runs `runs2/b0-M-0`, `-1`, `-2`: final validation loss 3.405540, 3.405689, 3.400939 (mean 3.4040), 13.8M non-embedding parameters, Muon lr 8e-3, 200M tokens (12,207 steps of 32 x 512). No rerun, so the baseline cannot be nudged after COUNT's results are known. Further B0-M seeds, if section 5 calls for them, use the same code path and recipe.
- **COUNT arm.** The same model and recipe: Muon lr chosen by the pilot in section 3, weight decay 0.1, warmup 200 steps then linear decay to zero, clip 1.0, bf16 autocast, GPU memory cap 0.75, context 512, the Stage 1 tokenizer (8,192 tokens). Same data order per seed and same name-keyed initialisation, so every tensor starts identical to B0-M of the same seed and any gap comes from COUNT, not from a lucky start. Code: `src/clm/countread.py`, `src/clm/counts.py`, `src/clm/s2model.py` (`B0GPT.forward`, `count`), `src/clm/s2train.py`. Config `configs/stage2a.json`. Driver `scripts/stage2a_driver.py`. Analysis `scripts/stage2a_analyze.py` and `scripts/bucket_eval.py`.
- **Count corpus.** Every training-split document of FineWeb-Edu shard 000: 881,048,239 tokens in 718,756 documents, which is 4.405 times the 200,000,154-token training stream. End-of-text id 0 closes each document. Validation documents are excluded, so the counts never contain the text we score on. The corpus's first 200,000,154 tokens equal `data/fineweb/train.bin` exactly (checked), so each training position's own occurrence is in the counts exactly once. That fact is what makes the leave-one-out rule below possible. sha256 in `data/fineweb/count_corpus.json`.
- **Tables.** Orders 2 to 5 (histories of 1 to 4 tokens), built under a 6 GiB budget by a size-only rule, so no validation number touched the choice. Minimum history counts c_min are 1, 1, 2, 2 (a history seen too rarely is dropped to save space). Rows: 8,142 / 12,973,613 / 53,590,439 / 85,100,508; 5.15 GiB in all. Each history keeps its 16 most frequent continuations with their counts, plus its total count. A history never contains end-of-text, so no history runs across a document boundary.
- **The read.** Fixed in advance. Where the paper is not explicit this is our reading of it.
  - One slot per order, 4 orders (J = 4). Each slot holds the first K = 8 stored continuations of that order's history. Each continuation's score is C(h,w)/C(h) divided by J (Eq. 1's 1/J). K = 8 follows the paper's training setting. K and J are not tuned.
  - A history that is not stored (never seen, under c_min, or containing end-of-text) leaves its slot empty, and its 1/J stays, so the other slots do not get louder.
  - A history uses only tokens inside the 512-token window, so the first k-1 positions of a window have no order-(k+1) slot. This keeps training and evaluation windows independent.
  - Eq. 2's 0.4 back-off applies only to an unobserved continuation. A stored top-K continuation is always observed, so scores are plain count ratios.
  - The embedding for position t is f_t = the sum over slots of score x (row of the untied head). The model input is rms_norm(embedding + f_t). B0 already applies RMSNorm without gain to its input, which is the paper's "embedding amplification" without learned gain and bias, so no extra layer is added. The head is zero-initialised as in B0 and is trained through f_t as well as through the logits.
- **Leave-one-out.** Our addition, because the count corpus is only 4.4 times the training stream. Without it, a training position's own next token sits in the counts and hands the model part of the answer it is being scored on, an advantage that validation text never gets. So during training the position's own occurrence is subtracted from C(h,w) and C(h), and a history whose count then falls below c_min is dropped. The first K continuations are taken before the subtraction and are not re-ranked after it. Validation reads need no correction (validation documents are not in the corpus) and use none.
- **Lookups.** They run on a CPU thread one step ahead of the GPU, so the GPU rarely waits. The GPU applies them with one weighted sum over head rows (`embedding_bag`).
- **Metric.** Mean cross-entropy (nats/token) on the first 2,097,152 tokens of `data/fineweb/val.bin` at the last step (4,096 windows of 512). Also recorded: training wall-clock (sum of step times, evaluation excluded, including any wait for lookups), peak VRAM, peak host RSS, mean lookup seconds per step.

## 3. Procedure

a. **Pilot.** COUNT-M, seed 0, 50,000,000 tokens with the schedule compressed to that length, Muon at lr 4e-3, 8e-3, 16e-3. The lowest final validation loss wins. If the winner is on the grid edge, one further run at half or double that rate is added once, then the rule is applied again. The choice goes to `runs2/pilot_count/choice.json` and applies to all COUNT runs. COUNT gets its own rate because B0's rate was tuned for B0, and a fair test tunes both. The Stage 2.0 B0-M pilot at Muon 8e-3 reached 3.6649; it is reported for orientation and is not a decision input. Pilots touch only B0 and COUNT.
b. **Main runs.** COUNT-M seeds 0, 1, 2 at the chosen rate, 200M tokens each (`runs2/count-M-0`, `-1`, `-2`).
c. **Leakage diagnostic.** COUNT-M seed 0 with leave-one-out switched off in training (`runs2/count-noloo-M-0`), same rate. It shows how much the leave-one-out rule matters. Reported, not a decision input.
d. **Bucket evaluation, after the runs.** `scripts/bucket_eval.py` re-evaluates every final checkpoint and records validation loss by history-count bucket. It uses 3-token and 4-token histories and the buckets not stored / 1-9 / 10-99 / 100-999 / >=1000. Orders 4 and 5 store no history seen once, so "not stored" includes those. This shows which kind of history any gain comes from. Its overall loss must match `results.json` within 0.002, which checks that the re-evaluation reads the model the same way training did.

Runs run under `systemd-run --user` with the GPU memory cap 0.75 (the harness kills long background jobs, and the laptop GPU needs headroom). `embeddinggemma` and `heretic-harness` are stopped for the night and restarted after (approved by Nate 2026-11-02). `nvidia-smi` is checked before a job.

## 4. Reference numbers reported alongside (not decision inputs)

- **Pure n-gram language model over the same tables.** Absolute discounting 0.75 with back-off to shorter histories and an add-one unigram (`src/clm/ngram_lm.py`). The discount was fixed before any validation read, from the textbook value (Chen and Goodman 1999). Validation loss 4.7455 nats on the same 4,096 windows (`results/stage2a_ngram.json`). This shows how far the tables alone get without a transformer.
- **Measured smoke run of the COUNT arm before registration.** 0.399 s per step against 0.375 for B0-M (about 6% slower), lookups 27 ms per step median, peak VRAM 1.6 GB, host RSS peak 7.4 GB.

## 5. Decision rule

Paired by seed: d_s = loss(COUNT-M) - loss(B0-M), computed by `clm.analyze.verdict` (the Stage 1 rule). SESOI 0.01 nats (Stage 2.0). Pairing by seed removes the shared seed-to-seed noise from the comparison.

- **X_BETTER:** mean d <= -0.01 and every d_s < 0. Gate 1 passes; Stage 2B-0 proceeds.
- **X_WORSE:** mean d >= +0.01 and every d_s > 0. Gate 1 fails.
- **NO_MEANINGFUL_DIFFERENCE:** |mean d| < 0.01 and every |d_s| < 0.02. Gate 1 fails.
- **INCONCLUSIVE:** anything else. If |mean d| < 0.02 the plan's five-seed rule applies: seeds 3 and 4 are added for both COUNT-M and B0-M (four more runs, B0-M with the Stage 2.0 recipe), and the rule is applied once more on five pairs. Otherwise gate 1 fails.

Cost is reported with the verdict: seconds-per-step ratio, VRAM, host RAM. The plan's headline is loss at equal wall-clock and VRAM. This test is at equal tokens because the two arms share data order and schedule. COUNT's per-step cost of about 6% is stated next to the verdict. To convert the extra time into nats, the analysis uses the one token-scaling measurement we have: B0-M seed 0 reached 3.664853 at 50M tokens (the Stage 2.0 pilot, `runs2/pilot/muon-0.008`) and 3.405540 at 200M, a loss-against-tokens exponent b = ln(3.664853 / 3.405540) / ln 4 = 0.0529. If COUNT takes r times as long per step as B0-M, giving back that time costs B0-M's mean loss x (r^b - 1) in nats; at r = 1.06 that is 0.0105. The "equal wall-clock" gap is mean d plus that cost. If the verdict is X_BETTER at equal tokens but the equal wall-clock gap is above -0.01, this is flagged to Nate for a decision before 2B-0 starts (it is a re-evaluation point, not a failure). The exponent comes from one seed at two schedule lengths, so it is a rough conversion, stated as such in the result.

## 6. What a verdict does

X_BETTER: Stage 2B-0 (offline Kenyon-backoff test, CPU only) starts. Any other verdict: the memory track stops and the null is written up with the bucket results. 2D still runs.

## 7. Conduct

A diverged run (non-finite loss or gradient) is rerun once from scratch with the same seed. A second divergence is reported and gives no verdict. No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash; this stops the design drifting toward a result. Interrupted runs resume from their checkpoint, and that does not change results (tested for the COUNT arm: `tests/test_s2count.py::test_interrupted_and_resumed_count_run_ends_bit_identical`). After any change to `src/`, the suite and `scripts/mutation_audit.py` must pass. The COUNT code carries mutants M37 to M52 (a leaked own-occurrence, leave-one-out applied at validation, a stale prefetch, a history read from the wrong look-back window, among others), so a test that cannot catch those bugs does not count.

## 8. Limits stated in advance

- One backbone size (13.8M non-embedding parameters), one corpus, one table size. The paper's gains come from tables of 17 GB to 797 GB; ours is 5.15 GiB.
- B0's learning rate was tuned on B0 and COUNT's on COUNT, each on seed 0 at 50M tokens. Seed 0 is therefore both a tuning seed and a comparison seed for both arms. The bias is symmetric.
- The reading of Eq. 1 and 2 in section 2 is ours where the paper is silent. K and J are fixed, not tuned.
- Leave-one-out is our addition and has no published check. The diagnostic in 3c is the only evidence on it.
- Laptop wall-clock varies with background load.
- 50M-token pilots are a short-horizon proxy.

TL:DR. We test whether a 5.15 GiB n-gram count table, read through the model's own unembedding, beats the frozen B0-M baseline (mean 3.4040) by at least 0.01 nats at equal tokens, over three paired seeds. Everything about the read, the leave-one-out correction and the learning-rate pilot is fixed before any run. If COUNT is not clearly better, the memory track stops and the null is written up.
