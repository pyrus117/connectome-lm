# Stage 2H oracle repeat: the effort law holds on the gain model, the RL gate is still OPEN, and the headroom did not shrink (2026-11-09)

This is a repeat analysis of an existing checkpoint under rules that were already registered. It is not a new pre-registration. `PREREG_stage2h.md` section 7 (dated 2026-11-06, commit `eabc1c3`) required the Stage 2D oracle analysis to be run again on `loop_gain_all-M-0` before any halting or RL decision, because Stage 2D's RL gate (OPEN) was measured on the old looped model. The rules are those of `PREREG_stage2d.md` (dated 2026-11-03, commit `b4dcde9`), unchanged: the section 3 gate, the price schedule k(b) = 0.2 x exp(-b / tau) with tau 2.171, efforts 0 to 10, and the free-exit quantiles.

Code `scripts/stage2d_analyze.py`, `src/clm/depth_law.py`, `src/clm/depth_eval.py`; run `runs2/loop_gain_all-M-0`; results `results/stage2h_oracle.json` (committed with this note). Stage 2D's matching file is `results/stage2d.json`. Deviations: one entry for this analysis, dated 2026-11-09 in `notes/deviations.md`. The script now builds the model with the run's own `loop_gain` setting and takes an optional output file name, so this run did not overwrite `results/stage2d.json`. For Stage 2D's own run both defaults give the old behaviour.

## Verdict: P9 is supported. The RL gate is still OPEN. No RL work was started.

One seed, one model. This is not a verdict on RL.

1. **P9 (does the best depth rise in a straight line with effort?): supported.** Over the nine efforts whose mean best depth sits in the registered middle band, the straight-line fit has R^2 0.993 (bar 0.9) and a positive slope of 0.149 passes per effort level.
2. **Frontier (does a per-token optimal depth beat fixed depth and the free-exit rule by at least 0.01 nats at equal mean depth?): yes, for the oracle.** It wins by 0.0623 nats over fixed depth and 0.0619 over free exit, both executed. The oracle uses each token's true loss, so it is an upper bound and nobody could deploy it. The free-exit rule, which is deployable, has no real edge over plain fixed depth (0.0004 on average).
3. **RL gate: OPEN.** The oracle beats every executable policy by more than the registered threshold of 0.01 nats. Stage 2D found the same on the old model (0.0631 against 0.0619 now).

Under PREREG_stage2d.md section 5, an open gate means stop and report to Nate before any RL work. RL has been paused since 2026-11-03 ("We will pause 2D here"). Nothing in this note starts it.

## What was run

No training. The checkpoint is `loop_gain_all-M-0`, the Stage 2H `both` model, seed 0. It is Stage 2D's looped model with two changes: a per-pass gain (a table of numbers that scales how big a step each pass may take) and a training loss averaged over all six depths. The Stage 2H result note (`notes/stage2h_result_2026-11-09.md`) describes it in full. In short, it has 2 prelude blocks, a 2-block loop reused on every pass, then 2 coda blocks. "Depth" is the number of passes through the loop, from 1 to 6, so depth r executes 4 + 2r blocks. It has 13.8M non-embedding parameters and was trained for 12,207 steps of 32 x 512 tokens (200M tokens) on the Linux laptop RTX 3050.

The analysis ran on q-pc (RTX 4060), not the laptop, over the same 4,096 validation windows of 512 tokens (2,097,152 tokens from `data/fineweb/val.bin`) that Stage 2D used. The script re-evaluates the model first and stops if its depth-6 loss differs from `results.json` by more than 2e-3. It passed. In fact all six depth losses agree with `results.json` to within 0.000013 (the largest gap is at depth 2).

All losses below are validation loss in nats per token (lower is better; a nat is the unit the loss is measured in). One forward pass reads out every depth, so each token's loss at every depth is known exactly.

## Fixed depth against B0-M

B0-M is the plain six-block model with no loop. Its three Stage 2.0 final losses average 3.4041. "Gain" below is B0-M's mean minus the loop's loss at that depth, so a positive number means the loop is better.

| Depth | Blocks run | Val loss | Gain over B0-M mean (3.4041) |
|---|---|---|---|
| 1 | 6 | 3.3934 | 0.0107 |
| 2 | 8 | 3.3406 | 0.0634 |
| 3 | 10 | 3.3331 | 0.0710 |
| 4 | 12 | 3.3326 | 0.0714 |
| 5 | 14 | 3.3326 | 0.0714 |
| **6** | 16 | **3.3326** | **0.0714** |

The loss is flat after depth 4. Depths 4, 5 and 6 differ by less than 0.00001, so the label "best depth 6" carries no meaning. Stage 2D's looped model was best at depth 4 (3.3705) and got worse after it. This one does not get worse.

Depth 1 is now 0.0107 better than B0-M, where Stage 2D's loop was 0.0180 worse. Depth 1 runs the same six blocks as B0-M, so the all-depth loss appears to have fixed the depth-1 problem (the Stage 2H note shows it in the ablations). This is one run, not compute-matched: depth 6 runs 16 blocks per token against B0's 6.

## Marginal gains and the geometric decay fit

The marginal gain of an extra pass is the loss at depth r minus the loss at depth r + 1, on the mean over all tokens.

| Extra pass | 1 to 2 | 2 to 3 | 3 to 4 | 4 to 5 | 5 to 6 |
|---|---|---|---|---|---|
| Gain (nats) | 0.0527 | 0.0075 | 0.00045 | -0.0000014 | 0.0000081 |

The first pass beyond depth 1 does most of the work. The gain then falls by a factor of about seven, then about seventeen, then to nothing. In Stage 2D the last two gains were negative (-0.0010 and -0.0022), meaning extra passes hurt. Here they are zero for practical purposes, in the sense that an extra pass neither helps nor hurts.

The last two gains are noise-sized, and their signs are not stable. `results.json`, evaluated on the laptop, gives +0.000003 for both. This run, on q-pc, gives -0.0000014 and +0.0000081. A rounding-level difference between the two GPUs flips a sign. Read both as zero.

DeepSeek's straight-line law follows if gains shrink by a constant factor per pass (geometric decay). The diagnostic fits log gain = log g0 - r / sigma over the positive gains. R^2 says how much of the variation the fit explains (1 is perfect).

| Quantity | Value (Stage 2D) |
|---|---|
| sigma | 0.449 passes (0.572) |
| g0 | 0.502 nats (0.250) |
| R^2 | 0.996 (0.999) |

Only positive gains can be fitted, because negative ones have no logarithm. In Stage 2D that was three points. Here the code kept four: 1 to 2, 2 to 3, 3 to 4, and 5 to 6. The fourth is 0.0000081, a noise-sized number, and 4 to 5 was left out only because its sign came out negative. Only three gains are clearly positive, so the fit is weak evidence. With the laptop's values in `results.json` all five gains would be positive and the fit would change. Read it as "consistent with fast decay", not as proof of geometric decay. The law predicts a slope of sigma / tau = 0.449 / 2.171 = 0.207 passes per effort level. The oracle's measured slope, 0.149, is lower, as in Stage 2D (predicted 0.264, measured 0.190).

## Oracle depth by price of effort

The price of one extra pass is k(b) = 0.2 x exp(-b / tau) nats, with tau 2.171, at effort b = 0 to 10. The price runs from 0.2 nats per pass (almost nobody goes deeper) down to 0.002 (everybody who benefits goes deeper). The oracle gives each token the depth that minimises its own loss plus price x depth.

- **Oracle loss** is the loss if each token is read at its own best depth, assuming no token's early exit changes what other tokens see.
- **Executed loss** runs the policy for real. A token that exits keeps its state frozen, and later tokens read that frozen state in the passes the exited token skips. This is the loss that counts for the gate.
- **Cross-token effect** is executed loss minus oracle loss.

| Effort | Price (nats/pass) | Mean depth | Oracle loss | Executed loss | Cross-token effect |
|---|---|---|---|---|---|
| 0 | 0.2000 | 1.25 | 3.2870 | 3.3220 | 0.0349 |
| 1 | 0.1262 | 1.38 | 3.2652 | 3.3071 | 0.0419 |
| 2 | 0.0796 | 1.53 | 3.2500 | 3.2964 | 0.0464 |
| 3 | 0.0502 | 1.69 | 3.2399 | 3.2890 | 0.0491 |
| 4 | 0.0317 | 1.85 | 3.2333 | 3.2841 | 0.0508 |
| 5 | 0.0200 | 2.02 | 3.2291 | 3.2808 | 0.0518 |
| 6 | 0.0126 | 2.19 | 3.2264 | 3.2788 | 0.0524 |
| 7 | 0.0080 | 2.35 | 3.2248 | 3.2775 | 0.0527 |
| 8 | 0.0050 | 2.49 | 3.2239 | 3.2767 | 0.0528 |
| 9 | 0.0032 | 2.60 | 3.2234 | 3.2762 | 0.0528 |
| 10 | 0.0020 | 2.69 | 3.2232 | 3.2759 | 0.0527 |

The mean cross-token effect is 0.0489 nats (Stage 2D: 0.0536). About half of the oracle's apparent edge over fixed depth is lost once its exits are actually carried out (at effort 10, 0.1122 predicted against 0.0595 executed, against fixed depth at the same mean depth), and the other half survives (next section).

The oracle never gets near the depth cap. Even at the cheapest price its mean depth is 2.69 (Stage 2D: 3.10), because extra passes past depth 3 are worth almost nothing on average. At effort 0, 79% of tokens stay at depth 1. At effort 10, 39% are still at depth 1, but about 9% go to depth 6 (the rest are spread over depths 2 to 5, at 17%, 10%, 14% and 12%). So although the mean gain of passes 4 to 6 is zero, some individual tokens still gain from them. Which tokens, and why, was not tested.

## P9: the effort law

The registered test takes the efforts whose mean best depth lies in [1.5, 5.5] and fits a straight line, depth = slope x b + intercept. Effort 0 (1.25) and effort 1 (1.38) fall below the band, so efforts 2 to 10 are used.

| Quantity | Value (Stage 2D) | Registered bar |
|---|---|---|
| Points used | 9, efforts 2 to 10 (9) | at least 4 |
| Slope | 0.149 (0.190) | positive |
| Intercept | 1.264 (1.365) | none |
| R^2 | 0.993 (0.957) | at least 0.9 |
| P9 | **supported** | |

Reading, beyond the rule. The fit passes, but the "middle" it covers is mean depth 1.53 to 2.69, not the whole 1.5 to 5.5 band, because the oracle never goes deeper. This is the same caveat as Stage 2D, with a narrower range. The curve is closer to straight than before: the step in mean depth per effort level is about 0.16 at the cheap end of the fit (effort 2 to 3) and about 0.09 at the other (effort 9 to 10), against 0.27 and 0.07 in Stage 2D. That is why R^2 is higher. A straight line over a range this short does not test the law over the depths DeepSeek's law was written for.

## Frontier and the RL gate

At each oracle effort, the free-exit rule and fixed depth are compared at the same mean depth by linear interpolation. Gap is the free-exit (or fixed-depth) loss minus the oracle's executed loss, so a positive gap means the oracle is better.

| Effort | Mean depth | Free exit | Fixed depth | Oracle executed | Gap to free exit | Gap to fixed depth |
|---|---|---|---|---|---|---|
| 0 | 1.25 | 3.3791 | 3.3804 | 3.3220 | 0.0571 | 0.0584 |
| 1 | 1.38 | 3.3712 | 3.3732 | 3.3071 | 0.0641 | 0.0661 |
| 2 | 1.53 | 3.3625 | 3.3652 | 3.2964 | 0.0662 | 0.0688 |
| 3 | 1.69 | 3.3556 | 3.3569 | 3.2890 | 0.0666 | 0.0679 |
| 4 | 1.85 | 3.3501 | 3.3483 | 3.2841 | 0.0660 | 0.0642 |
| 5 | 2.02 | 3.3443 | 3.3405 | 3.2808 | 0.0635 | 0.0596 |
| 6 | 2.19 | 3.3386 | 3.3392 | 3.2788 | 0.0598 | 0.0604 |
| 7 | 2.35 | 3.3372 | 3.3380 | 3.2775 | 0.0597 | 0.0605 |
| 8 | 2.49 | 3.3363 | 3.3370 | 3.2767 | 0.0596 | 0.0603 |
| 9 | 2.60 | 3.3355 | 3.3361 | 3.2762 | 0.0593 | 0.0599 |
| 10 | 2.69 | 3.3348 | 3.3354 | 3.2759 | 0.0589 | 0.0595 |
| **Mean** | | | | | **0.0619** | **0.0623** |

The gate rule (PREREG_stage2d.md section 3): OPEN if the mean, over oracle efforts whose mean depth lies inside the free-exit frontier's range, of (free-exit executed loss minus oracle executed loss at the same mean depth) is at least 0.01 nats. The free-exit range is mean depth 1.18 to 6.0, and all eleven oracle efforts (1.25 to 2.69) sit inside it. The mean gap is 0.0619, more than six times the threshold. **The gate is OPEN.**

The gap to free exit peaks at 0.0666 (effort 3) and falls to 0.0589 (effort 10). In Stage 2D it sat flat near 0.065 from effort 3 on.

The oracle sees each token's true loss, so this is the most a perfect halting rule could win, not what any real one would. The plan rates RL as a weak lever at this scale (PREREG_stage2d.md section 1).

## Free-exit policies

The free-exit rule stops a token at the first pass where its loop state changes by less than a threshold (relative change norm(x_r - x_(r-1)) / norm(x_r)), and runs to depth 6 otherwise. The threshold is set at a quantile of all observed change values; a higher quantile means more tokens exit early.

| Quantile | Mean depth | Executed loss |
|---|---|---|
| 0.02 | 6.00 | 3.3326 |
| 0.05 | 6.00 | 3.3326 |
| 0.1 | 6.00 | 3.3326 |
| 0.2 | 5.80 | 3.3326 |
| 0.3 | 5.20 | 3.3326 |
| 0.4 | 4.60 | 3.3326 |
| 0.5 | 4.00 | 3.3326 |
| 0.6 | 3.40 | 3.3327 |
| 0.7 | 2.80 | 3.3341 |
| 0.8 | 2.20 | 3.3383 |
| 0.9 | 1.60 | 3.3588 |
| 0.97 | 1.18 | 3.3828 |

The free-exit rule now loses nothing until it exits very early. At quantile 0.6 (mean depth 3.4) its loss is 3.3327 against 3.3326 at full depth.
That is only half the story, because fixed depth is just as flat: depth 3 gives 3.3331 and depth 4 gives 3.3326. So the free-exit rule still has no edge over plain fixed depth. At equal mean depth its gap to the oracle is 0.0619, against 0.0623 for fixed depth. The free-exit loss is lower than fixed depth's at nine of the eleven oracle efforts, by 0.0006 to 0.0027, and higher at efforts 4 and 5, by 0.0018 and 0.0039. The average edge is 0.0004. This is far under the 0.01 that the registered rules treat as meaningful. The state-change signal buys nothing over a fixed depth.

## Reading beyond the rule

These points are read from the tables and were not registered as tests.

- **A cheap fixed depth is now almost free.** Depth 3 gives 3.3331, within 0.0005 of the full six passes (3.3326), at half the passes (3 against 6, or 10 blocks against 16). Depth 2 is within 0.0080. This matters for compute at inference, separately from halting. Training this model cost about 2.4 times the work per step of Stage 2D's (see the Stage 2H note).
- **The oracle headroom did not shrink meaningfully.** The gap to free exit is 0.0619 against Stage 2D's 0.0631, and to fixed depth 0.0623 against 0.0635. These are single-model figures, and a difference of 0.0012 is inside the noise. The Stage 2H note guessed that a halting rule would have least to win once every depth beyond 4 gives the same loss. For the oracle that did not hold. One reading is that the oracle's gain comes from tokens differing from each other, not from the mean curve. This was not tested.
- **The oracle lead is not a result anyone can reach.** It still sees each token's true loss.

## What this means for the next step

RL stays paused (Nate, 2026-11-03). Nothing about this result starts it. The audit note's follow-up 5B is a non-RL halting probe. It has three parts: an entropy or margin exit rule, a supervised linear probe on per-token marginal gains, and COUNT plus LoopGPT with the stored n-gram count as a prior on depth. Each gets its own pre-registration.

## Limits

- One seed, one model. B0-M's seed standard deviation at 200M tokens is about 0.0027 nats (the sample SD of its three losses), so differences under 0.01 between frontiers are inside the noise. Stage 2D's note quotes 0.0054 for this figure; PREREG_stage2h.md section 1 recomputed it from the three B0-M losses (3.405540, 3.405689, 3.400939) as 0.0027, and 0.0027 is the figure used here.. The 0.062 gap is far above either, but it is still one trained model.
- One size (13.8M non-embedding parameters), one corpus.
- The oracle uses the future loss of each token, so it is not executable. The gate measures headroom, not an achievable result.
- The executed oracle losses are worse than the oracle losses (cross-token effect 0.0489 on average) because frozen exited states change what later tokens read. Whether a learned policy could close any of that gap is untested.
- The price schedule is our choice. The law is DeepSeek's "local reward-level approximation", and the depth cap of 6 bends both ends. The oracle's mean depth reached only 2.69, so P9 was tested over a narrower middle than the registered band.
- Seed 0 was trained on the Linux laptop and analysed on q-pc. Floating-point results are not bit-identical across the two GPUs. The depth-6 loss matched `results.json` within the script's 2e-3 check, and all six depth losses matched within 0.000013. The two noise-sized gains still change sign between the two evaluations, so the geometric fit is not stable at the 1e-6 level.
- The geometric decay fit rests on three clearly positive gains plus one of 0.0000081.
- The depth comparison with B0-M is not compute-matched, and this model trained on about 2.4 times the work per step of the Stage 2D reference.
- The same 4,096 validation windows were used in Stage 2D and here, so the two analyses are comparable window for window.

TL:DR. Stage 2D's oracle analysis, repeated on the Stage 2H gain-plus-all-depth model (seed 0, analysed on q-pc), gives the same answers as before. P9 is supported (slope 0.149, R^2 0.993, over a narrow mean-depth range of 1.53 to 2.69). The oracle beats fixed depth and the free-exit rule by about 0.062 nats at equal mean depth (0.0619 and 0.0623, against 0.0631 and 0.0635 before), so the RL gate is still OPEN, though the oracle sees each token's true loss, so this is headroom and not an achievable result. Extra passes no longer hurt: loss is flat after depth 4 (3.3326), so the free-exit rule loses nothing until it exits very early, but plain fixed depth is just as flat and the free-exit rule still has no edge over it. A fixed depth of 3 is within 0.0005 of the full six passes at half the passes. This is a repeat analysis with the registered rules, not a new pre-registration, and it covers one seed and one model, so it is not a verdict on RL. RL stays paused; the next step is the non-RL halting probe (follow-up 5B), each part with its own pre-registration.
