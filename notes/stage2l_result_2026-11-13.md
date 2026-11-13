# Stage 2L result: the COUNT read helps the looped model by 0.17 nats, P21 is supported, but the loop passes did nothing in either arm (2026-11-13)

Pre-registration `PREREG_stage2l.md` (dated 2026-11-10, committed 2026-11-11 07:55 as `2ab68af`, before any run). Code `src/clm/s2model.py`, `src/clm/s2loop.py`, `src/clm/s2train.py`, `scripts/stage2l_driver.py`, `scripts/stage2l_analyze.py`, decision rules `src/clm/rules2h.py` (unchanged from Stage 2H); config `configs/stage2l.json`; tests `tests/test_s2l.py`, mutants M199 to M205; results `results/stage2l.json`. Run on q-pc, six runs (two arms, seeds 0 to 2), all at commit `f96d054`, tree clean, status complete. The batch launched at 11:25 on 2026-11-11 and the last run finished at 23:44 on 2026-11-13 (file times of `results.json` on q-pc; seed 0 LOOP 12:58 and seed 0 LOOP+COUNT 20:05 on 2026-11-12, seed 2 LOOP 16:38 on 2026-11-13). Deviations: one, logged in `notes/deviations.md` (entry dated 2026-11-12, commit `6c926e7`), see Events.

Terms, once. *Cross-entropy* (validation loss) is the model's average surprise per token, in nats; lower is better. The *looped model* reuses one block pair several times in a row on each token. Each reuse is a *pass*, and its *depth* is how many passes were used (1 to 6). *Best-depth loss* is the lowest of the six per-depth validation losses. *LOOP* is the looped model with no memory read. *LOOP+COUNT* is the same model with the COUNT read added at the input. The *COUNT read* looks up how often the recent words followed each other in the training text and feeds that to the model. The read used is plain COUNT, which Stage 2K2 carried forward. The *separate-rate recipe* trains the block matrices with Muon at 8e-3 and everything else with AdamW at 64e-3. *Paired by seed* means each LOOP+COUNT run is compared with the LOOP run of the same seed; d_s is LOOP+COUNT minus LOOP, so negative means the read helps. *SESOI* is the smallest difference worth caring about, 0.01 nats. The verdict rule is the Stage 1 rule. *X_BETTER*: mean d at or below -0.01 and every seed below zero. *X_WORSE*: the mirror image. *NO_MEANINGFUL_DIFFERENCE*: mean under 0.01 in size and every seed under 0.02 in size. *INCONCLUSIVE*: everything else.

## Verdict

- **P21 is supported.** The read lowered the looped model's best-depth loss by 0.01 or more on all three seeds. Verdict X_BETTER, mean d = -0.1722 nats (seed 0 -0.1733, seed 1 -0.1703, seed 2 -0.1731).
- **Same verdict at depth 1 and at depth 6.** X_BETTER both times, mean d -0.1722 at each depth.
- **The loop passes did nothing in either arm.** This limits what P21 means, and it is the main finding. Details below.
- **Against the stated expectation (PREREG section 4).** The expectation was a gain from the read in the loop smaller than its gain in the plain model (0.1224 at Stage 2G), but past 0.01. The gain is past 0.01. It is not smaller. It is larger in the loop arm (0.172 against 0.122), because LOOP is slightly worse than B0-M and LOOP+COUNT is better than plain COUNT.
- **Futility check.** After seed 0 it printed "FUTILITY continue", so seeds 1 and 2 ran.

Best-depth validation loss (nats per token, first 2,097,152 tokens of `data/fineweb/val.bin`, last step; size M, 200M tokens, context 512). The best depth is in brackets.

| Seed | LOOP | LOOP+COUNT | d_s |
|---|---|---|---|
| 0 | 3.4007 (depth 5) | 3.2274 (depth 3) | -0.1733 |
| 1 | 3.3990 (depth 4) | 3.2287 (depth 4) | -0.1703 |
| 2 | 3.3982 (depth 2) | 3.2251 (depth 3) | -0.1731 |
| Mean | 3.3993 | 3.2271 | -0.1722 |

The three seeds of LOOP+COUNT sit within 0.0036 of each other, and the three of LOOP within 0.0025.

## The loop passes are inert

Within every run, all six per-depth losses are equal to within 0.00006 nats. The best-depth loss and the depth-1 loss differ by 0.00005 or less on every run. So the "best depth" labels above are noise among six equal numbers. Depth 1 equals depth 6.

Mean gain from each extra pass (nats; positive means the extra pass helped), passes 2 to 6:

| Arm | pass 2 | pass 3 | pass 4 | pass 5 | pass 6 |
|---|---|---|---|---|---|
| LOOP | +0.000034 | -0.000006 | -0.000002 | -0.000003 | +0.00000004 |
| LOOP+COUNT | +0.000018 | +0.000002 | -0.0000004 | -0.0000037 | -0.0000052 |

For comparison, Stage 2H (`results/stage2h.json`, one shared rate, same looped model) got about 0.05 nats from the first extra pass (depth 1 about 3.39, best about 3.33), and its reference loop got 0.042. On the separate-rate recipe the looped model behaves like a plain six-block model.

- LOOP here (mean 3.3993) is 0.066 worse than Stage 2H's gained all-depth loop (mean about 3.3336, shared rate). That compares across recipes (PREREG section 8, limit 4).
- LOOP is 0.014 worse than B0-M on separate rates (Stage 2G, `results/stage2g.json`, seeds 0 to 2: 3.3872, 3.3851, 3.3836, mean 3.3853).
- **Learned loop gain.** I read the learned `loop_gain` tensor in the seed 0 checkpoints on q-pc after the runs. Shape (6, 384) in both arms. Mean 0.052 (LOOP) and 0.047 (LOOP+COUNT). Largest absolute value 14.8 and 18.5. So the gain did not shrink to zero.
- **Why the passes add nothing is not established.** I did not test it. Treat this as an unexplained observation, not a finding about cause.

## What P21 does and does not say

P21 asks only whether the read helps the looped model. It does, by about 0.17 nats on every seed. But the loop is inert here, so this is the read's effect on a six-block model. It says nothing about whether the read and real looping add up.

## Described comparisons (no decision)

**LOOP+COUNT against plain COUNT with the same read** (`reference_read_loss`, the Stage 2K and 2K2 COUNT runs, seeds 0 to 2):

| Seed | COUNT | LOOP+COUNT | LOOP+COUNT minus COUNT |
|---|---|---|---|
| 0 | 3.2354 | 3.2274 | -0.0080 |
| 1 | 3.2886 | 3.2287 | -0.0599 |
| 2 | 3.2969 | 3.2251 | -0.0718 |
| Mean | 3.2737 | 3.2271 | -0.0466 |

Verdict X_BETTER by the same rule. Read it with care.

- Plain COUNT's seed spread on this recipe is large (SD about 0.03 to 0.05, Stage 2K and 2K2). LOOP+COUNT's three seeds are much tighter.
- Seed 0's gap (-0.0080) is under the 0.01 bar by itself.
- Because the loop is inert, this gap is not evidence that looping helps on top of COUNT. The likeliest reading is plain COUNT's seed instability, but that was not tested.
- Plain COUNT in Stage 2G (different runs, same config; seeds 3.2362, 3.2842, 3.2681) gives the same picture.

**The four effects** (mean paired difference in best-depth loss; the plain model's loss is its final loss):

| Effect | Mean difference |
|---|---|
| Read added to the loop (LOOP+COUNT minus LOOP) | -0.1722 |
| Loop added to the read (LOOP+COUNT minus COUNT) | -0.0466 |
| Read added to B0 | not computed |
| Loop added to B0 | not computed |

The two B0 rows are empty because the Stage 2G B0 runs came from the laptop and the analysis script had no path to them. The fields are null in `results/stage2l.json`. I did not compute them by hand. Stage 2G's own figure for the read in the plain model is 0.1224.

## Cost

| | LOOP | LOOP+COUNT |
|---|---|---|
| s per step (q-pc) | 1.854 | 1.921 |
| peak VRAM MiB | 2,670 | 2,586 |
| peak host RSS MiB | 1,902 | 7,239 |

LOOP+COUNT is about 3.6% slower per step and uses about 5,300 MiB more host memory. Its peak VRAM is slightly lower than LOOP's. This note does not explain that.

## Events

q-pc was reset by a Windows update at 17:47 on 2026-11-11. `loop_gain_all_sep-M-0` resumed from its step-11,000 checkpoint (`notes/deviations.md`, 2026-11-12, commit `6c926e7`). Steps 11,000 to about 11,238 ran twice. No other deviation.

## What the verdict does (PREREG section 5)

The rule says: COUNT plus the gained loop is the recipe for the next scaling test, and whether to open gate 3 (a size-L check) with it goes to Nate with the numbers and a cost estimate. The trigger fired. The evidence above argues against using it as is. The looped model gave no depth gain here, so a size-L check built on this recipe would test a six-block model with COUNT. The decision belongs to Nate. Two options, no recommendation:

- **(a) Find out why the passes are inert** on the separate-rate recipe. One way is to run the loop at the shared rate with the read, which is Stage 2H's loop with COUNT.
- **(b) Skip the loop** and carry plain COUNT forward.

[TK: size-L cost estimate not made]. Nothing here changes gate 1 (Stage 2G).

## Limits

PREREG section 8, limits 1 to 7, apply: one read, one size, one corpus; equal tokens, not equal compute (the looped arms do about 2.4 times the block passes per step); seed spread; the loop arms were rerun on the separate-rate recipe, so any comparison with Stage 2H's numbers crosses recipes; one placement of the read (the input); B0-M on separate rates was trained on the laptop; the futility stop did not fire. Also:

- **The inert loop** means the three-seed X_BETTER is a statement about the read in a six-block model.
- **The learned-gain check used seed 0 only.**

## Files

`PREREG_stage2l.md`, `src/clm/s2model.py`, `src/clm/s2loop.py`, `src/clm/s2train.py`, `scripts/stage2l_driver.py`, `scripts/stage2l_analyze.py`, `tests/test_s2l.py`, `results/stage2l.json`, `notes/deviations.md`. Run directories: `runs2/loop_gain_all_sep-M-s` and `runs2/loop_gain_all_count_sep-M-s`, seeds 0 to 2.

TL:DR. We added the plain COUNT read to the looped model on q-pc, three seeds, and P21 is supported: the read lowered best-depth loss on every seed (mean d -0.1722 nats, X_BETTER, same at depth 1 and depth 6). That is bigger than the read's 0.1224 gain in the plain model, not smaller as expected. But the loop passes did nothing in either arm: all six depths agree to within 0.00006 nats, so the result is about the read in a six-block model. Why the passes are inert is unexplained, since the learned gain did not shrink to zero (seed 0 only). LOOP+COUNT beats plain COUNT by 0.0466 on average, but plain COUNT's seed spread is large and the loop is inert, so this is not evidence that looping helps. The rule's trigger for a size-L check fired, but the evidence argues against using it as is, and the decision is Nate's: find out why the passes are inert, or skip the loop and carry plain COUNT forward. No size-L cost estimate exists.
