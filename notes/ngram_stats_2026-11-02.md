# N-gram statistics of the training and validation tokens, 2026-11-02

Planning measurement for Stage 2 (`docs/superpowers/plans/2026-11-02-stage2-kenyon-memory.md`). Script: `scripts/ngram_stats.py` (numpy 2.5.4, laptop CPU, 15 s). Data: `data/fineweb/train.bin` (200,000,154 tokens) and `val.bin` (2,098,135 tokens), own BPE, 8,192 tokens. Keys are exact token n-grams; documents are not separated, so a few keys span an end-of-text token.

For each validation position t, the key is the n-gram ending at t, the key an n-gram memory would use to predict token t+1. The buckets give the share of validation keys by how often that key occurs in the training tokens.

| n | distinct n-grams in train | seen once (share of distinct) | unseen | 1–9 | 10–99 | 100–999 | ≥1000 |
|---|---|---|---|---|---|---|---|
| 2 | 7,044,748 | 43.1% | 1.7% | 5.5% | 13.0% | 28.8% | 50.9% |
| 3 | 52,624,767 | 68.7% | 19.9% | 23.1% | 26.3% | 21.8% | 8.9% |
| 4 | 117,443,950 | 83.2% | 52.4% | 25.3% | 15.1% | 6.2% | 1.2% |

Reading: an exact-hash memory (DeepSeek's Engram) has a trained row for almost every validation bigram, but for 52% of validation 4-grams and 20% of trigrams the row only holds collision noise from other n-grams, and a further 25% (4-grams) and 23% (trigrams) were seen fewer than 10 times. A similarity-based address could reuse rows learned from similar n-grams in exactly these buckets. The buckets are the planned unit of analysis for the Stage 2 memory comparison.
