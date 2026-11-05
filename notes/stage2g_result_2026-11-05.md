# Stage 2G result: gate 1 holds on equally tuned recipes, and the gap grows to 0.122 nats (2026-11-05)

Pre-registration `PREREG_stage2g.md` (commit a191a36). Code `src/clm/rules2g.py`, `scripts/stage2g_driver.py`, `scripts/stage2g_analyze.py`; config `configs/stage2g.json`; results `results/stage2g.json`, `runs2/pilot_2g/choice.json`. Laptop RTX 3050 (4 GB), GPU memory cap 0.75, 2026-11-05 08:24 to 13:05 NZDT. No crash, no deviation.

Terms, once. *Muon* trains the 2-D block matrices and *AdamW* the embeddings and head. *Separate rates* means AdamW runs at a multiple of the Muon rate (Stage 2F). *Cross-entropy* (validation loss) is the model's average surprise per token, in nats; lower is better. *SESOI* is the smallest difference worth caring about, 0.01 nats. *Gate 1* is the Stage 2 plan's first check: COUNT must beat B0 at size M.

## Verdict: P13 supported. X_BETTER, mean d = -0.1224. Gate 1 passes.

COUNT-M on separate rates (Stage 2F's runs, Muon 8e-3, AdamW 64e-3) against B0-M on separate rates (Muon 8e-3, AdamW 64e-3), 200M tokens, three paired seeds, `clm.analyze.verdict` with SESOI 0.01:

| Seed | B0-M shared rate (Stage 2.0, not rerun) | B0-M separate rates | COUNT-M separate rates | d_s (COUNT - B0 separate) |
|---|---|---|---|---|
| 0 | 3.4055 | 3.3872 | 3.2362 | -0.1509 |
| 1 | 3.4057 | 3.3851 | 3.2842 | -0.1009 |
| 2 | 3.4009 | 3.3836 | 3.2681 | -0.1155 |

Mean d = -0.1224, every seed negative and below -0.01: X_BETTER. Stage 2A's gap on shared rates was -0.0798, so equal tuning widened the gap by 0.043. Cost: COUNT 0.397 s per step against B0's 0.384, 3.4% slower.

## B0's own gain (described, no decision)

B0-M on separate rates against B0-M on the shared rate: mean d = -0.0188, verdict X_BETTER by the rule. COUNT gained 0.0614 from the same change (Stage 2F), about 3.3 times B0's gain. The pilot had pointed to a smaller difference (B0 +0.024 against COUNT +0.037 at 50M tokens); at 200M tokens COUNT's advantage from the fast AdamW rate is much larger.

## Pilot (B0-M, Muon 8e-3, seed 0, 50M tokens)

| AdamW rate | validation loss |
|---|---|
| 8e-3 (shared, Stage 2.0) | 3.6649 |
| 32e-3 | 3.6475 |
| 64e-3 | 3.6409 (best) |
| 128e-3 | 3.6752 |

64e-3 was inside the grid, so no edge extension ran. It beat the shared rate by 0.0240, over the 0.01 rerun gate.

## Reading (interpretation, not a measured result)

- A plausible reason COUNT gains more than B0 from a fast AdamW rate: COUNT reads its memory through `head.weight`, which AdamW trains, so the memory read depends on the head learning fast. Not tested.
- For orientation only: COUNT-M on separate rates (mean 3.2629, 13.8M non-embedding parameters) is slightly below B0-L on the shared rate (3.261 in Stage 2.0, 29.4M parameters). B0-L was not retuned, so this is not a like-for-like size comparison; it only says the memory gain is large next to the model-size ladder.
- Gate 1 is no longer in question. Stage 2A's -0.0798 was not an artefact of untuned B0.

## What the verdict does (PREREG section 5)

X_BETTER: gate 1 holds on equally tuned recipes and the memory track stands. The next steps are Stage 2F section 5's: repeat the gated read (Stage 2E) and the scale-only control on the separate-rate recipe, each as a new pre-registration.

## Limits (as registered)

One seed at 50M tokens picked B0's rate; B0 got a thinner search than COUNT (one Muon rate, four AdamW rates, 16e-3 not tried); both baselines come from earlier commits and are not rerun; one size, one corpus.
