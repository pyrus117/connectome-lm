# Stage 2F result: separate rates for Muon and AdamW lower COUNT-M's loss by 0.061 nats (2026-11-05)

Pre-registration `PREREG_stage2f.md` (commit bca00eb). Code `src/clm/s2model.py` (`param_groups`, `adam_mult`), `src/clm/s2train.py`, `src/clm/rules2f.py`, `scripts/stage2f_driver.py`, `scripts/stage2f_analyze.py`; config `configs/stage2f.json`; results `results/stage2f.json`, `runs2/pilot_2f/choice.json`. Laptop RTX 3050 (4 GB), GPU memory cap 0.75, 2026-11-04 21:22 to 2026-11-05 05:43 (8 h 21 min). One crash fix in the analysis script, `notes/deviations.md` (45c17a5); no run was affected.

Terms, once. *Muon* trains the 2-D block matrices and *AdamW* trains the embeddings, head and gates; a *cell* is a (Muon rate, AdamW rate) pair. A *shared rate* is both at one value, as in Stages 2A to 2E. *Cross-entropy* (validation loss) is the model's average surprise per token, in nats; lower is better. *SESOI* is the smallest difference worth caring about, 0.01 nats.

## Verdict: P12 supported. X_BETTER, mean d = -0.0614

Separate rates at Muon 8e-3 and AdamW 64e-3 (AdamW 8 times Muon) against COUNT-M at the shared 0.032, 200M tokens, three paired seeds, `clm.analyze.verdict` with SESOI 0.01:

| Seed | COUNT shared rate (Stage 2A runs, not rerun) | COUNT separate rates | d_s |
|---|---|---|---|
| 0 | 3.3221 | 3.2362 | -0.0858 |
| 1 | 3.3206 | 3.2842 | -0.0364 |
| 2 | 3.3300 | 3.2681 | -0.0619 |

Mean d = -0.0614, every seed negative and below -0.01, so X_BETTER. The five-seed rule does not apply (it only applies to INCONCLUSIVE). Cost: 0.397 s per step against 0.404 for the shared rate (noise), same peak VRAM (1,593 MiB). The seed-to-seed spread of d (0.036 to 0.086) is wider than Stage 2E's, so the size of the gain is less certain than its sign.

For scale: Stage 2E's gated read gained 0.0111 and Stage 2A's COUNT over B0 gained 0.0798. This single change to the learning rates is worth about 5.5 times the gate.

## Pilot (COUNT-M, seed 0, 50M tokens, validation loss)

Rows are the Muon rate, columns the AdamW rate. `a` marks a Stage 2A pilot (equal rates, not rerun); `x` marks a cell added by the edge rule. The Muon 4e-3 row also has the Stage 2A cell with AdamW at 4e-3: 3.5653.

| Muon \ AdamW | 8e-3 | 16e-3 | 32e-3 | 64e-3 | 128e-3 |
|---|---|---|---|---|---|
| 4e-3 | | | | 3.4623 (x) | |
| 8e-3 | 3.5249 (a) | 3.4785 | 3.4305 | **3.4272** | 3.4997 (x) |
| 16e-3 | 3.5546 | 3.4756 (a) | 3.4450 | 3.4428 | |
| 32e-3 | 3.5868 | 3.5164 | 3.4640 (a) | 3.4620 | |

- The AdamW rate drives the gain. Down any column the Muon rate matters little once AdamW is at 32e-3 or higher; along any row loss falls steeply with the AdamW rate until 32e-3 to 64e-3.
- The best cell, Muon 8e-3 (B0's own rate) with AdamW 64e-3, sat on the grid edge, so the edge rule ran two more cells. AdamW 128e-3 is worse (3.4997), and Muon 4e-3 is worse (3.4623). The optimum is inside the extended grid.
- The pilot gate (best separate cell beats the best equal-rate cell by 0.01) was met by a wide margin: 3.4640 - 3.4272 = 0.0368.

Both Stage 2A's pilot edge (32e-3 equal rates "still improving") and the audit's guess are supported: COUNT wanted a fast AdamW rate, and the shared rate pushed Muon up with it.

## B0 control (described, no decision)

B0-M, seed 0, 50M tokens, at the same cell (Muon 8e-3, AdamW 64e-3): 3.6409 against 3.6649 at the shared 8e-3 (Stage 2.0 pilot). B0 gains 0.0240 from separate rates at this single untuned cell; COUNT gains 0.0368 over its best shared-rate cell (0.0977 over the shared 8e-3 cell). So most of the effect is not specific to COUNT, and COUNT's own surplus at the pilot is about 0.013. B0 was tried at this one cell only, so its best separate-rate gain is not known and may be larger.

## What this changes (reading, not a measured result)

- **Gate 1 needs rechecking.** Stage 2A's COUNT-over-B0 gap (-0.0798) compared COUNT at a tuned shared rate with B0 at its own shared rate. If B0 also gains from separate rates, the gap shrinks by an amount not yet measured at 200M tokens. The one B0 pilot gained 0.0240 against a gap of 0.0798, which would leave COUNT ahead, but that is one cell at 50M tokens and says little about 200M.
- **The Stage 2E gate result may move.** The gated read's gates also train under AdamW at the shared 0.032, and its 0.0111 gain is small next to this one. The gain may shrink, vanish or grow once AdamW runs at its own rate. The scale-only control from `PREREG_stage2e.md` section 6 is better run at the separate-rate recipe than at the shared one, or run at both.
- **Stages 2B, 2C and 2D used the shared rate.** Their null and small results (Kenyon buckets, consolidation, the looped model) were measured at a rate that this stage shows to be far from the best AdamW rate. They are not refuted, but they were measured on a handicapped recipe.

## What the verdict does (PREREG section 5)

P12 supported: the pair (Muon 8e-3, AdamW 64e-3) becomes the COUNT recipe for ungated COUNT. The next test repeats the decoupling on the gated read, as a new pre-registration. Stage 2A's gap is described against the B0 control, not recomputed.

## Limits (as registered, plus one)

One seed at 50M tokens picked the cell; weight decay moves with the Muon rate; COUNT has been tuned over more cells than B0; ungated COUNT only; one size, one corpus. Added after seeing the result: the 0.061 gain is measured against Stage 2A's COUNT runs, which ran at an earlier commit, and its seed spread is wide, so it is better read as "large" than as "0.061".
