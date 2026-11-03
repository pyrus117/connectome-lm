# Stage 2D result: the effort law holds in the middle of the range, and the RL gate is open (2026-11-04)

Pre-registration `PREREG_stage2d.md` (dated 2026-11-03, commit `b4dcde9`). Code `src/clm/s2loop.py`, `src/clm/s2train.py`, `src/clm/depth_law.py`, `src/clm/depth_eval.py`, `scripts/stage2d_driver.py`, `scripts/stage2d_analyze.py`; config `configs/stage2d.json`; run `runs2/loop-M-0`; results `results/stage2d.json` (committed in `e6001b6`). Deviations: none logged for Stage 2D.

## Verdict: P9 is supported. The RL gate is OPEN. No RL work was started.

Three questions were registered, and each has an answer.

1. **P9 (does the best depth rise in a straight line with effort?): supported.** Over the nine efforts whose mean best depth sits in the registered middle band, the straight-line fit has R^2 0.957 (bar 0.9) and a positive slope of 0.190 passes per effort level.
2. **Frontier (does a per-token optimal depth beat fixed depth and the free-exit rule by at least 0.01 nats at equal mean depth?): yes, for the oracle.** It wins by 0.0635 nats over fixed depth and 0.0631 over free exit, both executed. The oracle uses each token's true loss, so it is an upper bound and nobody could deploy it. The free-exit rule, which is deployable, does not beat plain fixed depth at all.
3. **RL gate: OPEN.** The oracle beats every executable policy by more than the registered threshold of 0.01 nats, so a learned exit policy has something it could try to close.

Under PREREG section 5, an open gate means stop and report to Nate before any RL work. On 2026-11-03 Nate said "We will pause 2D here". No RL work was started, and the RL step is not run.

## What was run

One training run, `loop-M-0` (seed 0), on the laptop RTX 3050. It is B0-M's six blocks rearranged into a looped model: 2 prelude blocks, a 2-block loop that is reused on every pass, then 2 coda blocks. "Depth" is the number of passes through the loop, from 1 to 6, so depth r executes 4 + 2r blocks. Depth 1 is B0's own six-block computation. Each micro-batch of 8 windows trained at one depth drawn uniformly from 1 to 6.

Training was 12,207 steps of 32 x 512 tokens (about 200M tokens, the Stage 2.0 recipe unchanged). It ran from 19:26 to 22:17 on 2026-11-03. Step time rose from 0.67 s to about 1.5 s partway through. The CPU jobs of Stage 2B-0 were running alongside at that point, but that was not confirmed as the cause. It does not affect any loss.

All losses below are validation loss in nats per token (lower is better; a nat is the unit the loss is measured in) on the first 2,097,152 tokens of `data/fineweb/val.bin`, which is 4,096 windows of 512. One forward pass reads out every depth, so each token's loss at every depth is known exactly.

## Fixed depth against B0-M

B0-M's three Stage 2.0 final losses average 3.4041. "Gain" below is B0-M's mean minus the loop's loss at that depth, so a positive number means the loop is better.

| Depth | Blocks run | Val loss | Gain over B0-M mean (3.4041) |
|---|---|---|---|
| 1 | 6 | 3.4221 | -0.0180 |
| 2 | 8 | 3.3798 | 0.0242 |
| 3 | 10 | 3.3718 | 0.0323 |
| **4** | 12 | **3.3705** | **0.0336** |
| 5 | 14 | 3.3715 | 0.0326 |
| 6 | 16 | 3.3737 | 0.0304 |

Depth 4 is best, 0.034 below B0-M. B0-M's seed standard deviation is 0.0054 (PREREG section 7), so this is well outside seed noise, but it is one run. It is also not a like-for-like compute comparison: depth 4 runs 12 blocks per token against B0's 6. Depth 1 is worse than B0-M by 0.018 even though it is the same six-block computation; one likely reason is that the loop saw depth 1 in only about a sixth of its batches, but that was not tested. Past depth 4 the loss creeps back up.

## Marginal gains and the geometric decay fit

The marginal gain of an extra pass is the loss at depth r minus the loss at depth r + 1, on the mean over all tokens.

| Extra pass | 1 to 2 | 2 to 3 | 3 to 4 | 4 to 5 | 5 to 6 |
|---|---|---|---|---|---|
| Gain (nats) | 0.0422 | 0.0081 | 0.0013 | -0.0010 | -0.0022 |

The first pass beyond depth 1 does most of the work. The gain falls by about a factor of five or six each pass, then turns negative at the fourth pass: past depth 4, an extra pass hurts on average.

DeepSeek's straight-line law follows if gains shrink by a constant factor per pass (geometric decay). The diagnostic fits log gain = log g0 - r / sigma over the positive gains:

| Quantity | Value |
|---|---|
| sigma | 0.572 passes |
| g0 | 0.250 nats |
| R^2 | 0.999 |

The fit used the three positive gains only (negative gains have no logarithm). Three points and two fitted numbers make a high R^2 easy to get, so read it as "consistent with geometric decay", not as strong proof. The law then predicts a slope of sigma / tau = 0.572 / 2.171 = 0.264 passes per effort level. The oracle's measured slope, 0.190, is lower. The oracle depth flattens at high effort, as the P9 section shows.

## Oracle depth by price of effort

The price of one extra pass is k(b) = 0.2 x exp(-b / tau) nats, with tau 2.171, at effort b = 0 to 10. The price runs from 0.2 nats per pass (almost nobody goes deeper) down to 0.002 (everybody who benefits goes deeper). The oracle gives each token the depth that minimises its own loss plus price x depth. It can do this because it knows every token's loss at every depth.

- **Oracle loss** is the loss if each token is read at its own best depth, assuming no token's early exit changes what other tokens see.
- **Executed loss** runs the policy for real. A token that exits keeps its state frozen, and later tokens read that frozen state in the passes the exited token skips. This is the loss that counts for the gate.
- **Cross-token effect** is executed loss minus oracle loss.

| Effort | Price (nats/pass) | Mean depth | Oracle loss | Executed loss | Cross-token effect |
|---|---|---|---|---|---|
| 0 | 0.2000 | 1.22 | 3.3335 | 3.3623 | 0.0288 |
| 1 | 0.1262 | 1.38 | 3.3080 | 3.3453 | 0.0372 |
| 2 | 0.0796 | 1.60 | 3.2861 | 3.3310 | 0.0449 |
| 3 | 0.0502 | 1.87 | 3.2690 | 3.3206 | 0.0516 |
| 4 | 0.0317 | 2.16 | 3.2574 | 3.3140 | 0.0566 |
| 5 | 0.0200 | 2.42 | 3.2507 | 3.3105 | 0.0597 |
| 6 | 0.0126 | 2.64 | 3.2472 | 3.3085 | 0.0613 |
| 7 | 0.0080 | 2.81 | 3.2455 | 3.3076 | 0.0621 |
| 8 | 0.0050 | 2.93 | 3.2447 | 3.3071 | 0.0625 |
| 9 | 0.0032 | 3.03 | 3.2443 | 3.3069 | 0.0626 |
| 10 | 0.0020 | 3.10 | 3.2441 | 3.3067 | 0.0626 |

The mean cross-token effect is 0.0536 nats. About half of the oracle's apparent edge over fixed depth is lost once its exits are actually carried out (at effort 10, 0.1275 predicted against 0.0649 executed, against fixed depth at the same mean depth), and the other half survives (next section).

The oracle never gets near the depth cap. Even at the cheapest price its mean depth is 3.10, because past depth 4 extra passes hurt on average. At effort 0, 82% of tokens stay at depth 1; at effort 10, 27% go to depth 6 (the rest are spread over depths 1 to 5, with 37% at depth 1).

## P9: the effort law

The registered test takes the efforts whose mean best depth lies in [1.5, 5.5] and fits a straight line, depth = slope x b + intercept. Effort 0 (1.22) and effort 1 (1.38) fall below the band, so efforts 2 to 10 are used.

| Quantity | Value | Registered bar |
|---|---|---|
| Points used | 9 (efforts 2 to 10) | at least 4 |
| Slope | 0.190 | positive |
| Intercept | 1.365 | none |
| R^2 | 0.957 | at least 0.9 |
| P9 | **supported** | |

Reading, beyond the rule. The fit passes the registered bar, but the "middle" it covers is mean depth 1.6 to 3.1, not the whole 1.5 to 5.5 band, because the oracle never goes deeper. The curve is also visibly not straight: the step in mean depth per effort level is about 0.27 at the cheap end of the fit and about 0.07 at the other. A straight line still explains 96% of the variation, which is why the bar is met, but the law as DeepSeek states it (a local approximation) is only a fair description here. The flattening is what the gain table predicts, since gains go negative after depth 4 and the oracle has nowhere to go.

## Frontier and the RL gate

At each oracle effort, the free-exit rule and fixed depth are compared at the same mean depth by linear interpolation. Gap is the free-exit (or fixed-depth) loss minus the oracle's executed loss, so a positive gap means the oracle is better.

| Effort | Mean depth | Free exit | Fixed depth | Oracle executed | Gap to free exit | Gap to fixed depth |
|---|---|---|---|---|---|---|
| 0 | 1.22 | 3.4116 | 3.4127 | 3.3623 | 0.0493 | 0.0504 |
| 1 | 1.38 | 3.4042 | 3.4059 | 3.3453 | 0.0589 | 0.0606 |
| 2 | 1.60 | 3.3941 | 3.3966 | 3.3310 | 0.0631 | 0.0656 |
| 3 | 1.87 | 3.3866 | 3.3853 | 3.3206 | 0.0661 | 0.0647 |
| 4 | 2.16 | 3.3787 | 3.3786 | 3.3140 | 0.0647 | 0.0645 |
| 5 | 2.42 | 3.3759 | 3.3764 | 3.3105 | 0.0654 | 0.0660 |
| 6 | 2.64 | 3.3742 | 3.3747 | 3.3085 | 0.0657 | 0.0662 |
| 7 | 2.81 | 3.3729 | 3.3733 | 3.3076 | 0.0653 | 0.0657 |
| 8 | 2.93 | 3.3725 | 3.3723 | 3.3071 | 0.0654 | 0.0651 |
| 9 | 3.03 | 3.3722 | 3.3717 | 3.3069 | 0.0654 | 0.0649 |
| 10 | 3.10 | 3.3720 | 3.3716 | 3.3067 | 0.0653 | 0.0650 |
| **Mean** | | | | | **0.0631** | **0.0635** |

The gate rule (PREREG section 3): OPEN if the mean, over oracle efforts whose mean depth lies inside the free-exit frontier's range, of (free-exit executed loss minus oracle executed loss at the same mean depth) is at least 0.01 nats. The free-exit range is mean depth 1.18 to 6.0, and all eleven oracle efforts (1.22 to 3.10) sit inside it. The mean gap is 0.0631, more than six times the threshold. **The gate is OPEN.**

The oracle sees each token's true loss, so this is the most a perfect halting rule could win, not what any real one would. The plan rates RL as a weak lever at this scale (PREREG section 1).

## Free-exit policies

The free-exit rule stops a token at the first pass where its loop state changes by less than a threshold (relative change norm(x_r - x_(r-1)) / norm(x_r)), and runs to depth 6 otherwise. The threshold is set at a quantile of all observed change values; a higher quantile means more tokens exit early.

| Quantile | Mean depth | Executed loss |
|---|---|---|
| 0.02 | 6.00 | 3.3737 |
| 0.05 | 6.00 | 3.3737 |
| 0.1 | 5.99 | 3.3737 |
| 0.2 | 5.72 | 3.3732 |
| 0.3 | 5.19 | 3.3721 |
| 0.4 | 4.60 | 3.3712 |
| 0.5 | 4.00 | 3.3706 |
| 0.6 | 3.40 | 3.3711 |
| 0.7 | 2.80 | 3.3729 |
| 0.8 | 2.20 | 3.3775 |
| 0.9 | 1.60 | 3.3941 |
| 0.97 | 1.18 | 3.4135 |

The best executable free-exit policy (3.3706 at mean depth 4.0) is no better than plain fixed depth 4 (3.3705). The state-change signal buys nothing over a fixed depth. A learned policy would have to find a better signal than that, and the oracle's 0.063 nat lead is the room it would be searching in.

## Limits

- One seed. B0-M's seed standard deviation at 200M tokens is 0.0054 nats, so differences under 0.01 between frontiers are inside the noise. The 0.063 gap is far above that, but it is still one trained model.
- One size (13.8M non-embedding parameters), one corpus.
- The oracle uses the future loss of each token, so it is not executable. The gate measures headroom, not an achievable result.
- The executed oracle losses are worse than the oracle losses (cross-token effect 0.0536 on average) because frozen exited states change what later tokens read. Whether a learned policy could close any of that gap is untested.
- The price schedule is our choice. The law is DeepSeek's "local reward-level approximation", and the depth cap of 6 bends both ends. The oracle's mean depth reached only 3.10, so P9 was tested over a narrower middle than the registered band.
- Training at one random depth per micro-batch gives each depth about a sixth of the batches, so depth losses may be worse than for a model trained on all depths at once.
- The step-time slowdown during training has an unconfirmed cause. It does not touch the losses.
- The depth comparison with B0-M is not compute-matched: depth 4 runs 12 blocks per token against B0's 6.

TL:DR. One looped model, seed 0, 200M tokens. Fixed depth 4 is best (3.3705, 0.034 below B0-M, at twice the blocks). Marginal gains shrink about fivefold per pass and turn negative after depth 4. P9 is supported: over efforts 2 to 10 the best depth rises in a line (slope 0.190, R^2 0.957), though the line flattens at high effort. The oracle beats fixed depth and free exit by about 0.063 nats at equal mean depth, the free-exit rule is no better than fixed depth, and the RL gate is OPEN. Per Nate on 2026-11-03 ("We will pause 2D here"), no RL work was started and the RL step is not run.
