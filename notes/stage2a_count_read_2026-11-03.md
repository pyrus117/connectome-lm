# Stage 2A step 2: COUNT read, n-gram reference, COUNT arm (2026-11-03)

Pre-registration `PREREG_stage2a.md` (commit `a82f04d`). Code `src/clm/countread.py`, `src/clm/ngram_lm.py`, the `count` input in `src/clm/s2model.py`, the arm in `src/clm/s2train.py`. Tests `tests/test_countread.py`, `tests/test_ngram_lm.py`, `tests/test_s2count.py`; mutants M37–M52.

## What the read does (from the paper, `arXiv 2610.20456` §2.2–2.3, read in full on 2026-11-03)

f_t = (1/J) Σ_j Σ_k S(w_k | h_t,j) B[w_k]; the input is h0 = f_t + e_t. B is the model's untied head, so the table adds no learned parameters. The paper trains with K = 8 continuations and J = 6 orders; the tables here have J = 4 (orders 2–5).

Choices we made where the paper is silent (all in the pre-registration):
- A missing history leaves its order's slot empty and keeps the 1/J. The paper's "longest stored suffix at the j-th highest order" can be read as duplicating a shorter order's row into the empty slot; we do not.
- Eq. 2's 0.4 back-off never fires for stored top-K continuations (they are all observed), so scores are plain count ratios.
- B0 already applies RMSNorm to its input, which is the paper's "embedding amplification" without learned gain and bias.
- Leave-one-out subtracts the position's own occurrence from C(h,w) and C(h) and drops a history left under c_min. The first K continuations are taken before the subtraction.

## Found while building

- Plain count ratios give a normalised model probability 0 for any unseen continuation after a history whose continuations are all stored, so the first pure n-gram evaluation returned NaN. Fixed with absolute discounting (d = 0.75, textbook, fixed before any validation read).
- The mutation audit's M40 (the "no end-of-text in history" mask in the lookup) was an equivalent mutant: tables never store a history containing end-of-text, so the mask changed nothing. The mask was removed; M40 now plants a different bug.
- The analysis dry run caught a wrong token count: a 200M-token run records 199,999,488 tokens (12,207 whole steps).
- A test of the leakage that relied on a tiny model learning to read its own answer could not separate the arms in 300 steps (4.12 against 4.15 uniform). Replaced with a direct check of what the trainer feeds the model: the true next token is among the scored continuations 100% of the time without leave-one-out and 31% of the time with it.

## Measurements (laptop, conditions as noted)

- **Pure n-gram model over the tables, validation:** 4.7455 nats on the 4,096 x 512 windows (`results/stage2a_ngram.json`; absolute discount 0.75, back-off to shorter histories, add-one unigram). B0-M is 3.4040 (mean of three seeds). By 4-token-history bucket the n-gram loss is 4.87 (not stored, 47.3% of positions), 4.59 (1–9), 4.55 (10–99), 4.68 (100–999), 4.98 (≥1000).
- **COUNT arm smoke run** (size M, 4M tokens, 244 steps, GPU memory cap 0.75, nothing else on the GPU but a 158 MB reranker): 0.399 s per step against 0.375 for B0-M (about 6% slower); lookup 27 ms per step median, max 169 ms, hidden behind the GPU step by the prefetch thread; peak VRAM 1.6 GB; host RSS peak 7.4 GB (memory-mapped tables, mostly reclaimable page cache).
- **Token scaling for the equal-wall-clock conversion:** B0-M seed 0, 3.664853 at 50M tokens (pilot) and 3.405540 at 200M, exponent b = 0.0529; at a 1.06 time ratio the give-back is 0.0105 nats.
