# Stage 2A result: COUNT beats B0 (2026-11-03)

Pre-registration `PREREG_stage2a.md` (commit `a82f04d`). Analysis `scripts/stage2a_analyze.py` -> `results/stage2a.json` (main branch, commit `744cdce`). Runs in `runs2/`: pilot `runs2/pilot_count/`, main `count-M-{0,1,2}`, diagnostic `count-noloo-M-0`. Conditions: laptop RTX 3050 4 GB, bf16, GPU memory cap 0.75, `embeddinggemma` and `heretic-harness` not running; CPU-only jobs (the Stage 2B-0 sweep, an audit) ran alongside at low priority and did not change the step time (median 0.400 s in the pilots with and without them). Every run recorded `git_dirty: false`.

## Verdict: X_BETTER, gate 1 passes

| Seed | B0-M | COUNT-M | d = COUNT - B0 |
|---|---|---|---|
| 0 | 3.405540 | 3.3221 | -0.0835 |
| 1 | 3.405689 | 3.3206 | -0.0851 |
| 2 | 3.400939 | 3.3300 | -0.0709 |

Mean d = **-0.0798** (SD of d 0.0077; SD of the three COUNT losses 0.0051), eight times the SESOI of 0.01. Every seed is negative.

## Cost

- 0.4044 s per step against 0.3874 for B0-M: **COUNT is 4.4% slower per step** (the smoke run had said 6%). Lookups 0.046 s per step on average, hidden by the prefetch thread.
- Peak VRAM 1,593 MiB against 1,591 (the memory is on the host); peak host RSS 7.4 GB (memory-mapped tables, mostly page cache).
- Equal-wall-clock conversion (PREREG section 5, exponent 0.0529 from B0-M's 50M and 200M-token losses): the 4.4% extra time costs 0.0078 nats, so the equal-wall-clock gap is **-0.0721**; no flag.
- Against the Stage 2.0 size ladder (L = 8.031 x N^-0.0524), a gain of 0.0798 at size M equals a B0 with **1.57 times the non-embedding parameters** (about 21.7M instead of 13.8M), so about 1.5 times the compute after COUNT's own 4.4%. The paper reports 46-89% more compute for a standard transformer to match COUNT (60M-1B); this is inside that range at the small end.

## Learning-rate pilot (COUNT-M seed 0, 50M tokens; B0-M's Muon pilot reached 3.6649)

| lr | 4e-3 | 8e-3 | 16e-3 | 32e-3 |
|---|---|---|---|---|
| val loss | 3.5653 | 3.5249 | 3.4756 | 3.4640 |

The pilot kept improving to the top of the grid and the single extension (32e-3, the rule's one extra point) won, so the chosen rate is at the edge. The optimum may lie above it. The COUNT baseline of Stage 2B and every Stage 2B arm use this rate; B0's own rate (8e-3) was interior. If anything this makes COUNT's gain a lower bound.

## Leave-one-out matters (leakage diagnostic, seed 0)

| | validation loss | training loss, last 100 steps |
|---|---|---|
| leave-one-out (the arm) | 3.3221 | 3.2827 |
| no leave-one-out | **4.1612** | **2.3967** |

Without the correction the model learns to read its own answer out of the counts (training loss 0.9 nats lower) and then does 0.84 nats worse on text that is not in the count corpus. With it, the train/validation gap is 0.04 (seeds 1 and 2: 0.032 and 0.045), the size expected from ordinary generalisation.

## Where the gain sits (validation loss by the stored count of the exact 4-token history; mean of three seeds)

| Class | share | B0-M | COUNT-M | COUNT - B0 |
|---|---|---|---|---|
| not stored (never seen, or seen once) | 47.3% | 3.3395 | 3.3672 | **+0.028** |
| 1-9 | 17.4% | 3.3823 | 3.1600 | -0.222 |
| 10-99 | 19.5% | 3.4589 | 3.2346 | -0.224 |
| 100-999 | 11.5% | 3.4903 | 3.3846 | -0.106 |
| >=1000 | 4.2% | 3.5050 | 3.5324 | +0.027 |

For 3-token histories the "not stored" class (16% of positions) is worse by +0.088 (3.3338 against 3.2455). So COUNT's gain comes from histories that have stored counts, and where the exact history has none the model is slightly worse than a model without the memory. That is the case the Kenyon backoff (Stage 2B) is designed for; this table is the observation that motivates its test and is not a result of it. (The >=1000 class is also slightly worse: very common histories carry little information per count.)

## Reference

Pure n-gram model over the same tables, validation: 4.7455 nats (`results/stage2a_ngram.json`; B0-M 3.404, COUNT-M 3.324).
