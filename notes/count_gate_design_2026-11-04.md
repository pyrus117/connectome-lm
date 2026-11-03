# Gated COUNT read: design and why (2026-11-04)

This is a design record, not a result. No run has been made with the gated read and no validation loss has been read. Code is on branch `count-gate` (worktree `/home/nate/Projects/connectome-lm-countgate`), not committed.

Terms, once. A *history* is the last few tokens of the text so far. An *order* is a history length; COUNT uses J = 4 orders, histories of 1 to 4 tokens, and keeps the top 8 continuations of each. A *slot* is one order's stored answer for the current position. A *bucket fallback* is a slot filled from a group of similar histories (Stage 2B) because the exact history was never stored. A *gate* is a number that scales how much a slot is trusted. *Cross-entropy* (validation loss) is the model's average surprise per token, in nats; lower is better.

## What COUNT does now, and the weakness

COUNT (Magnusson, Smith & Koh 2026, arXiv 2610.20456) adds to each token embedding

f_t = (1/J) Σ_j Σ_k S(w_k|h_j) B[w_k]

That is the count-ratio-weighted sum of the model's own head rows B over the stored continuations w_k of each order j. Three things follow from the formula:

- Every order gets the same fixed weight, 1/J.
- An exact slot and a Stage 2B bucket slot look the same to the network.
- An order with no stored history adds a zero vector, so "no information" and "weak information" cannot be told apart.

## Two proposals

Nate proposed two fixes.

- **Avenue A.** A learned scalar weight per order (alpha_j), a separate one for bucket fallbacks (beta_j), and a learned "missing-history" vector for an empty slot.
- **Avenue B.** Replace the 2-winner Kenyon bucket ID with 1 winner per sub-codebook, and read the top-m cells independently (multi-probe LSH, Lv et al. 2007). The aim is a smaller table and more sharing of a cell between similar words.

## Evidence from earlier stages

Stage 2A (`notes/stage2a_result_2026-11-03.md`). Loss by stored count of the exact 4-token history, COUNT minus B0, mean of 3 seeds. Negative means COUNT is better.

| Stored count | Share of positions | COUNT minus B0 |
|---|---|---|
| not stored | 47.3% | +0.028 |
| 1-9 | 17.4% | -0.222 |
| 10-99 | 19.5% | -0.224 |
| 100-999 | 11.5% | -0.106 |
| >=1000 | 4.2% | +0.027 |

For 3-token histories the not-stored class is +0.088. COUNT overall beat B0 by 0.0798.

Stage 2B (`notes/stage2b_result_2026-11-04.md`). Filling empty slots with bucket counts made no meaningful difference. Kenyon was +0.0046 and random buckets +0.0040 against COUNT; Kenyon against random was +0.0007. In the not-stored class both were worse (+0.0085, +0.0073). The offline gain of 0.0116 nats in Stage 2B-0 did not carry into the network.

Stage 2B-0 sweep (`notes/stage2b0_result_2026-11-03.md`). Gain on rare histories at a 100M-token universe:

| Setting | Gain (nats) | Buckets |
|---|---|---|
| winners2 | 0.0248 | 55.5M (104.6M at 871M tokens, too big for RAM) |
| K31 | 0.0207 | 2.9M |
| K1 | 0.0155 | 0.4M |
| random buckets | 0.0007 | 4.2M |

Stage 2C (`notes/stage2c_result_2026-11-04.md`). 6.2% of twin pairs share a bucket, so 3% of twins reach the stored answer. That hash was a one-token-history overlay hash, never calibrated for one-token histories.

## Avenue B: assessed, not built

1. "One winner per sub-codebook" is K31, the setting Stage 2B already ran. Its bucket ID is the pair of winners: 2,048 x 2,048 = 4.19M possible IDs, 3.58M used at 4-token histories.
2. Reading each sub-codebook's winning cell on its own gives 2 x 2,048 = 4,096 cells for roughly 400M distinct histories, about 100,000 histories per cell. The sweep shows that fewer buckets pool more and gain less (K1 below K31). So 4,096 cells are expected to do worse. This is an inference, not a measurement.
3. The 6.2% twin rate comes from the Stage 2C overlay hash, not from the COUNT bucket tables. Coarser cells raise sharing between twins, and between unrelated histories by the same amount.
4. The main reason: Stage 2B showed the network does not use bucket fills at all (Kenyon equal to random). While the injection gives the network no way to tell a bucket slot from an exact one, or to turn it down, a better bucket table has no channel to show its value. Avenue B waits on Avenue A.

Multi-probe reading stays a valid idea. Revisit it once a gated read shows the network can use bucket slots.

## Avenue A: built, with one change

The diagnosis is right. The change is that the scalar depends on order, source (exact or bucket) and the history's count bin (1-9, 10-99, 100-999, >=1000), not on order and source alone.

Why. Stage 2A is worse at both ends of the count range (not stored +0.028, >=1000 +0.027) and better in between. One constant per order cannot treat a history seen 3 times differently from one seen 3,000 times. Interpolation weights bucketed by history count are the standard fix in n-gram smoothing (Bahl, Jelinek & Mercer 1983, as described in the Jelinek-Mercer section of Chen & Goodman 1999).

The missing-history vector is built as Nate proposed: one learned vector per order, added where that order has no stored history. That includes window positions too early to hold that history length.

Considered and not built: an Engram-style gate computed from the hidden state (Cheng et al. 2026, arXiv 2601.07372). Two independent small-scale logs found gating gave no gain: Karpathy's nanochat log (2026-01, d12 to d25) and modded-nanogpt PR 201 (124M class). Both are recorded in the neural vault's 2026-11-02 memory-layers research log. Also, COUNT injects at the embedding, where no hidden state exists yet.

An untested idea, not built: the head rows serve both as output weights and as the memory's values. A separate projection for the memory might matter.

## What was built

- `src/clm/countread.py`: `CountReader.lookup(x, y, states=True)` also returns an int8 array of shape (batch, time, J). 0 means no stored history for that order. 1 + bin means an exact history. 1 + 4 + bin means a bucket fallback. The bin is taken from the history count after leave-one-out.
- `src/clm/s2model.py`: `B0Config.count_orders` (0 = off). Two parameters. `count_gate`, shape (J, 8), starts at 1.0. `count_absent`, shape (J, d_model), starts at 0. Each order's scores are multiplied by `count_gate[order, state - 1]`; an order with state 0 adds `count_absent[order]`. At size M (d_model 384) that is 32 + 1,536 = 1,568 parameters. At the start of training the gated model gives exactly the same output as plain COUNT, and a test checks bit-equality. Both go to the AdamW group with the other non-matrix parameters (weight decay 0, same learning rate).
- `src/clm/s2train.py`: train config `count_gate` (default false) and CLI flag `--count-gate`. Arm names gain a `_gated` suffix (`count_gated`, `count_kenyon_gated`, `count_random_gated`). `results.json` records the learned gate table (`count_gate`) and the norm of each absent vector (`count_absent_norm`). `count_orders` can only be set through `count_gate`; setting it in a size config is refused.
- `scripts/bucket_eval.py` reads states for gated runs.
- Defaults leave every existing configuration and run unchanged.
- `scripts/mutation_audit.py` now takes the environment variables `CLM_MUT_WORK` and `CLM_MUT_RESULTS`, so two audits can run at once without sharing a scratch directory.

Verification. Test suite: 260 passed, 1 skipped (the skip existed before). New tests compare the reader's states with a brute-force count, compare the gated input with a by-hand sum, and check that only the used gates get gradient. Mutation audit: mutants M100 to M108 added; `scripts/mutation_audit.py` passed on the branch. All 98 mutants (M1 to M89 and the new M100 to M108) were killed by an assertion. Control C1 survived and C2 errored at collection, both as required. It ran with its own scratch directory because another session was working in the main tree.

## Check on the real tables

Conditions: 2026-11-04, CPU only, the first 64 validation windows of 512 tokens (32,768 positions), the full exact tables `data/count` and the Kenyon K31 bucket tables `data/count_kenyon`. No model, no loss read. The table gives the measured share of positions in each state. Window starts too short to hold the history are excluded.

| History length | No slot, exact COUNT | No slot, COUNT + Kenyon | Bucket fills by pooled count: 1-9 / 10-99 / 100-999 / >=1000 |
|---|---|---|---|
| 1 token | 0.001 | 0.001 | 0 / 0 / 0 / 0 |
| 2 tokens | 0.006 | 0.002 | 0.000 / 0.001 / 0.001 / 0.002 |
| 3 tokens | 0.144 | 0.004 | 0.003 / 0.018 / 0.049 / 0.071 |
| 4 tokens | 0.439 | 0.005 | 0.004 / 0.044 / 0.171 / 0.216 |

Exact 4-token histories by count bin: 1-9 0.178, 10-99 0.212, 100-999 0.126, >=1000 0.045. The 0.439 no-slot share at 4 tokens is close to Stage 2A's 47.3% not-stored class. This is a 64-window sample, so the two are not identical.

Two readings. These are interpretation, not measured results.

- With Kenyon buckets the network almost never saw an empty slot: 0.5% of positions at 4 tokens, against 43.9% with exact COUNT. So in Stage 2B the "no information" signal was nearly erased. This fits the second possible reason in the 2B note (the clean "no information" signal was lost) but does not prove it.
- Bucket fills fall mostly in the top two pooled-count bins, because a pooled count sums many histories. So the count bins say little about how good a bucket slot is, and for bucket slots the per-source gate does most of the work. A better confidence measure for buckets is a possible later change, not built. One example is pooled count per distinct history, the bucketing Chen & Goodman 1999 found better.

Lookup cost: the states add no measurable time. Best of 6 warm-cache passes for one step's 16,384 tokens: exact COUNT 0.030 s with or without states; COUNT + Kenyon 0.083 s with or without states.

## Next steps, each needing Nate's decision

- A pre-registration ("Stage 2E") before any run. Suggested arms: COUNT-gated against COUNT (does the gate help on exact counts?), then Kenyon-gated against COUNT-gated (do buckets help once the network can turn them down?).
- Cost: about 1.4 h per 200M-token run at size M on the laptop (Stage 2B: 6 runs in 8 h 30 min), so about 4.2 h per arm of 3 seeds. No money.
- COUNT's learning rate (0.032) sat at the top of its pilot grid, and the gate parameters share it.

TL:DR. A gated COUNT read is built but not run: a gate per order, source and count bin, plus a learned absent vector per order, starting exactly equal to plain COUNT, with the mutation audit passed. Multi-probe cells (Avenue B) are not built, because Stage 2B showed the network ignores bucket fills until it can turn them down. Next is a Stage 2E pre-registration, about 4.2 h per arm of 3 seeds, no money.
