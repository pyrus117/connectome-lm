# Stage 2I result: no deployable exit rule beats fixed depth by 0.01 nats, P18 is not supported (2026-11-09)

Pre-registration `PREREG_stage2i.md` (dated 2026-11-09, commit `8c6422a`). Code `src/clm/halting.py`, `src/clm/depth_eval.py`, `scripts/stage2i_analyze.py`; tests `tests/test_halting.py`, mutants M152 to M172 (the test suite passed in full and `scripts/mutation_audit.py` killed 168 of 168 mutants, with both controls, as required before the run); results `results/stage2i.json` and `results/stage2i_seed0.json`, `results/stage2i_seed1.json`, `results/stage2i_seed2.json` (committed with this note). Deviations logged for Stage 2I: none.

## Verdict: P18 is not supported. No deployable rule beat fixed depth by 0.01 nats.

**P18 (does any exit rule that sees only what a token has already computed beat fixed depth by at least 0.01 nats at equal mean depth, on all three Stage 2H models?): no.**

Plain-language terms, defined once:

- A **nat** is the unit the loss is measured in. Lower loss is better.
- **SESOI** is the smallest effect that matters, registered as 0.01 nats.
- **Depth** is how many times a token goes through the shared loop, 1 to 6. **Mean depth** is the average depth over all tokens.
- An **exit rule** decides, pass by pass, which tokens stop looping. A rule is **deployable** if it uses only what the token has computed by the pass where it stops, not its future loss.
- A **frontier** is the curve of (mean depth, loss) you get by sweeping a rule's threshold from "stop almost nobody" to "stop almost everybody".
- **Executed loss** runs the exits for real: a token that stops keeps its state frozen, and later tokens read that frozen state. This is the loss used throughout.
- The **oracle** is the rule from Stage 2D and the Stage 2H oracle repeat that gives each token the depth that minimises its own loss plus a price per pass. It reads the token's true future loss, so nobody could deploy it. It is the ceiling, not a target.

Each rule is scored on the part of its frontier whose mean depth lies in the **registered region, 1.25 to 2.70**, which is where the oracle has room (PREREG section 3, fixed before any rule's frontier was computed). **Gap** is fixed-depth loss minus rule loss at the same mean depth, with fixed depth interpolated in a straight line between its six points. A positive gap means the rule is better. The mean of the gap over the three seeds must be at least 0.01, with every seed above zero, for X_BETTER.

| Rule | Counts for P18 | Seed 0 | Seed 1 | Seed 2 | Mean gap | Verdict | Share of oracle gap |
|---|---|---|---|---|---|---|---|
| `change` | no | +0.0009 | +0.0019 | +0.0014 | +0.0014 | NO_MEANINGFUL_DIFFERENCE | 2.2% |
| `entropy` | yes | -0.0301 | -0.0322 | -0.0292 | -0.0305 | X_WORSE | -48.8% |
| `margin` | yes | not scored | not scored | not scored | not scored | NOT_SCORED | not scored |
| `probe_small` | yes | +0.0028 | +0.0030 | +0.0028 | +0.0028 | NO_MEANINGFUL_DIFFERENCE | 4.5% |
| `probe_state` | yes | +0.0034 | +0.0037 | +0.0031 | +0.0034 | NO_MEANINGFUL_DIFFERENCE | 5.4% |
| oracle (reference) | no | +0.0627 | +0.0662 | +0.0588 | +0.0626 | | 100% |

The best deployable rule, `probe_state`, closes about 5.4% of the oracle's gap. `probe_small` closes 4.5% and `change` 2.2%. P18 counts `entropy`, `margin`, `probe_small` and `probe_state`. `change` repeats Stage 2D and does not count. Nothing reached the bar, so `winners` in `results/stage2i.json` is empty.

The number of frontier points that fell inside the region, on seeds 0, 1 and 2: `change` 3, 3, 3; `entropy` 3, 3, 3; `margin` 2, 2, 2; `probe_small` 5, 5, 5; `probe_state` 6, 6, 6; oracle 10, 11, 10. The oracle's effort 0 point sits at mean depth 1.2459 on seed 0 and 1.2352 on seed 2, just under the region's lower edge of 1.25, so those two seeds score ten oracle points and seed 1 scores eleven.

## What was run

No training. Three frozen models, the Stage 2H `both` model on seeds 0, 1 and 2 (`runs2/loop_gain_all-M-0`, `-1`, `-2`): a looped transformer with 13.8M non-embedding parameters, a per-pass gain, trained on the mean loss over all six depths. Stage 2H's note (`notes/stage2h_result_2026-11-09.md`) describes it in full. Seed 0 was trained on the Linux laptop and seeds 1 and 2 on q-pc. All analysis ran on q-pc (RTX 4060), started at 17:09 NZ time on 2026-11-09 and took about three hours.

One full-depth run of each model gives every token's signals at every pass:

- **Entropy** is how spread out the model's guess for the next token is, in nats. Low entropy means a confident guess.
- **Margin** is the gap between the model's top guess and its second guess, before they are turned into probabilities. A large margin means a confident guess.
- **Change** is how much the token's state moved on that pass, as a share of its size.

Five rules were tested. `change` stops a token when its change is small, `entropy` when its entropy is low, `margin` when its margin is large. These three set their threshold at one of twelve **quantiles** (the level at which a chosen share of all pass-by-token readings would stop), found on the fitting set and never on validation data.

`probe_small` and `probe_state` are **probes**: small prediction models fitted to guess how much the next pass will help a given token. Each is a **ridge regression**, a straight-line fit with a penalty (alpha) that stops it chasing noise. Alpha was picked from five values, 10 to 100,000, on the last quarter of the fitting set, which counts as **held-out** data because the fit never saw it. There is one ridge per pass. `probe_small` sees three numbers (entropy, margin, change). `probe_state` sees those three plus the token's 384-number state. A probe stops a token when its predicted gain is below the price of one more pass, at eleven prices (efforts 0 to 10, Stage 2D's schedule).

The probes and thresholds were fitted on 262,144 samples per seed from the last 4,096 windows of `data/fineweb/train.bin`, every 8th position. All rules were evaluated on the first 2,097,152 tokens of `data/fineweb/val.bin`, the same tokens as Stages 2D and 2H.

## Fixed depth, and `change` reproducing Stage 2D

Validation loss of each model read out at each fixed depth (nats per token).

| Depth | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 1 | 3.3934 | 3.4017 | 3.3904 |
| 2 | 3.3406 | 3.3458 | 3.3388 |
| 3 | 3.3331 | 3.3372 | 3.3321 |
| 4 | 3.3326 | 3.3367 | 3.3317 |
| 5 | 3.3326 | 3.3366 | 3.3317 |
| 6 | 3.3326 | 3.3366 | 3.3317 |

The curve is steep from depth 1 to 2 and flat from depth 4 on, as in the Stage 2H note. The depth 6 losses (3.3326, 3.3366, 3.3317) are the Stage 2H `both` best-depth losses.

`change` is Stage 2D's free-exit rule, kept as a check. It did what Stage 2D found: it ties with fixed depth. On seed 0 its loss is 3.3326 from mean depth 5.9 down to mean depth 4.0 (3.33263), 3.3331 at mean depth 3.0, and fixed depth reads 3.3326 at depth 4 and 3.3331 at depth 3. Seeds 1 and 2 look the same (3.3367 at depth 4.0 and 3.3372 at depth 3.0 on seed 1; 3.3317 and 3.3320 on seed 2). The mean gap in the region is +0.0014, about one seventh of the bar, and the same sign on all three seeds. Nothing has changed since 2D.

## Entropy and margin: worse than fixed depth

`entropy` loses to fixed depth by 0.0305 nats on average (-0.0301, -0.0322, -0.0292), at the same mean depth, on every seed. X_WORSE needs a mean gap of -0.01 or lower with every seed below zero, and both hold by a wide margin. In the numbers: at mean depth 1.95 on seed 0, `entropy` reads 3.3800 where fixed depth, interpolated, reads about 3.3432. A confident token is not a finished token. The tokens these signals call finished still gain from an extra pass, in a way the signals cannot see. That is a description of the numbers, not a tested mechanism.

**`margin` is NOT_SCORED, and that is a coverage outcome, not a performance result.** The registered rule needs at least three frontier points inside the region on a seed. The registered twelve quantiles put only two of `margin`'s points inside 1.25 to 2.70 on every seed. Seed 0's frontier, with the fixed-depth loss at the same mean depth interpolated here by me from the six fixed-depth losses in the same file (the gap column uses unrounded values, so it can differ from the rounded columns by 0.0001):

| Quantile | Mean depth | `margin` executed loss | Fixed depth at that mean depth | `margin` behind fixed depth by | In region |
|---|---|---|---|---|---|
| 0.02 | 5.90 | 3.3329 | 3.3326 | 0.0002 | no |
| 0.05 | 5.74 | 3.3336 | 3.3326 | 0.0009 | no |
| 0.1 | 5.47 | 3.3356 | 3.3326 | 0.0030 | no |
| 0.2 | 4.92 | 3.3411 | 3.3326 | 0.0085 | no |
| 0.3 | 4.37 | 3.3479 | 3.3326 | 0.0152 | no |
| 0.4 | 3.82 | 3.3550 | 3.3327 | 0.0223 | no |
| 0.5 | 3.26 | 3.3621 | 3.3330 | 0.0291 | no |
| 0.6 | 2.70 | 3.3688 | 3.3353 | 0.0335 | **no** (mean depth 2.7028) |
| 0.7 | 2.15 | 3.3751 | 3.3395 | 0.0357 | yes |
| 0.8 | 1.64 | 3.3814 | 3.3596 | 0.0218 | yes |
| 0.9 | 1.22 | 3.3875 | 3.3818 | 0.0056 | no |
| 0.97 | 1.04 | 3.3916 | 3.3913 | 0.0003 | no |

The 2.70 point is the near miss. Its exact mean depth is 2.7028, which is 0.0028 above the region's upper edge, so it is out. On seed 1 the matching point is at 2.7002 and on seed 2 at 2.7114. The point at 1.22 is below the lower edge of 1.25. That leaves two points inside the region, at 2.15 and 1.64, and the rule needs three.

By inspection of seed 0, `margin` is behind fixed depth at every point, by 0.022 to 0.036 nats from mean depth 1.6 to 3.8 and by less near either end of the frontier. That is the same size as `entropy`'s deficit (-0.0301 on seed 0). Seeds 1 and 2 show the same shape. At the two in-region points (mean depth about 2.15 and 1.64) the same interpolation gives `margin` behind fixed depth by 0.0383 and 0.0232 on seed 1, and by 0.0356 and 0.0224 on seed 2. A looser grid would almost certainly not have changed the conclusion. I did not run one, and the registered label stays NOT_SCORED. No rule, grid, region or threshold was changed afterwards (PREREG section 7).

## The probes: the gains are almost unpredictable

Both probes beat fixed depth on every seed, by about 0.003 (`probe_small` +0.0028, `probe_state` +0.0034). The edge is consistent, but it is a third of the 0.01 bar. The reason shows in how well the probes predict the thing they were built to predict.

For each pass r from 1 to 5, the gain is the loss at depth r minus the loss at depth r + 1 for a token. `probe_fits` in each seed file gives the ridge's held-out error (`holdout_mse`, mean squared error) and the variance of the gain on the same held-out samples (`holdout_var_of_gain`). A predictor that always guesses the average gain has an error about equal to that variance. So the **share of variance explained** is 1 minus holdout_mse over holdout_var_of_gain. Zero means no better than guessing the average. Negative means slightly worse.

Share of gain variance explained, in percent, held-out:

| Pass | `probe_small` seed 0 | seed 1 | seed 2 | `probe_state` seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|---|---|
| 1 | 0.38 | 0.36 | 0.35 | 0.78 | 0.88 | 0.77 |
| 2 | 0.09 | 0.11 | 0.08 | 0.16 | 0.17 | 0.12 |
| 3 | 0.02 | 0.02 | 0.02 | -0.01 | -0.02 | -0.04 |
| 4 | 0.00 | 0.00 | 0.00 | -0.03 | -0.05 | -0.08 |
| 5 | 0.00 | 0.00 | 0.00 | -0.06 | -0.07 | -0.05 |

At pass 1, where almost all the gain is, the 384-number state lifts the explained share from about 0.36% to about 0.8%. Both are under one percent. From pass 3 on both are at zero, and `probe_state` is a hair under it (so, no better than the average). The spread of the gain at pass 1 is large (variance 0.0905, 0.1001 and 0.0850 on seeds 0, 1 and 2, so a standard deviation of about 0.29 to 0.32 nats), well above the average gain of a first extra pass (about 0.052 to 0.056 on validation, from the fixed-depth table; the two figures come from different token sets). Individual tokens gain or lose far more than the average, and these features cannot tell which.

`probe_state` chose alpha 100,000, the top of the grid, in 13 of its 15 fits. The other two (pass 1 on seeds 0 and 1) chose 10,000. The fit wanted to shrink almost everything away, which fits with there being little to learn. `probe_small` picked alpha anywhere from 10 to 100,000. The grid was fixed in advance (PREREG section 8, limit 6), and I did not try larger alphas.

Per-token gains of an extra pass are close to unpredictable from these features. That is why the probes barely beat fixed depth. A small, consistent edge from an explained share of about one percent is what you would expect. At the cheapest price of a pass, the probes still stop at mean depth 2.8 to 2.9 (`probe_small` 2.92, 2.93, 2.92; `probe_state` 2.82, 2.88, 2.85 on seeds 0, 1, 2), so neither one reaches full depth.

## Reading beyond the rule

These points are read from the tables and were not registered as tests.

- **The oracle's headroom is information none of these signals carry.** The oracle's 0.0626 comes from knowing each token's own future loss. Entropy, margin, change, and linear probes of them and of the state do not carry it, to within the one percent above. Whether any signal does is open. A learned (RL) rule would have to find such a signal on its own. These results do not say one exists.
- **Even a perfect learned rule would not reach the oracle's unexecuted curve.** The oracle picks depths from each token's loss assuming nobody else is affected. When the exits are carried out, about 0.05 nats of its apparent edge disappears (mean 0.0489 in the Stage 2H oracle note, the cross-token effect). Any learned rule is scored on executed loss. The 0.0626 used here is already the executed figure.
- **A fixed depth of 3 already matches the full six passes.** On the three seeds it is 0.0005, 0.0006 and 0.0004 above the depth 6 loss (3.3331 against 3.3326, 3.3372 against 3.3366, 3.3321 against 3.3317), at half the passes (10 blocks against 16 per token). That is a compute fact, separate from halting.
- **The probe edge is small but not noise.** It has the same sign on all three seeds and is 0.0028 to 0.0037 on each. B0-M's seed standard deviation is about 0.0027, but that is for a different model, and the three seeds here share one validation set. Treat it as a small real effect that is far below the bar.

## What this means for RL

PREREG section 5: P18 not supported, so the question of RL goes to Nate with the measured number. The best simple rule, `probe_state`, closes about 5.4% of the oracle gap (0.0034 of 0.0626). The next best, `probe_small`, closes about 4.5%. No RL was run in this stage. RL stays paused (Nate, 2026-11-03). The decision is Nate's. No outcome of Stage 2I changes the memory track.

## Limits

These are PREREG section 8 restated, plus what happened during the stage.

- **Signals come from the full-depth run.** A deployed rule would read each token's signals while its neighbours are frozen at their own exit depths. Here every token ran every pass first, so the signals are slightly optimistic. The loss is not: it is executed. The cross-token effect on loss is about 0.05 nats, so the signal mismatch could matter, and it was not measured.- **The probes were fitted on training tokens the model has seen once.** Their losses may be a little lower than unseen text's.
- **Four candidate rules, best of four, no correction.** Nothing won, so this did not matter. If exactly one had won it would have been a screening result.
- **Three models, one validation set.** The seeds are three trained models, not three data splits. The registered SESOI is 0.01 and B0-M's seed standard deviation is about 0.0027.
- **The region and the price schedule are our choice**, fixed in advance and not tuned.
- **Linear probes only**, on at most 387 features, with a fixed alpha grid. A nonlinear probe, or more features, could find something these did not. That was not tested.
- **`margin` was not scored.** Its registered quantile grid left only two points inside the region on every seed. Whether it would have lost to fixed depth by about the same amount as `entropy` is read from seed 0's frontier by inspection, not from a registered test.
- **The oracle uses the future.** Its share is a ceiling, not a target.
- **One size, one corpus.** 13.8M non-embedding parameters, FineWeb-Edu.
- **Cross-machine.** Seed 0 was trained on the Linux laptop and analysed on q-pc. The analysis checks that the re-evaluated depth 6 loss matches `results.json` within 2e-3. Seed 0's fixed-depth depth 6 loss here (3.3326) agrees with the Stage 2H oracle repeat.
- **The stage trained nothing.** It reads three finished models, so it says nothing about what a model trained with a halting rule from the start would do.

## Files

`PREREG_stage2i.md` (commit `8c6422a`), `src/clm/halting.py`, `src/clm/depth_eval.py`, `scripts/stage2i_analyze.py`, `tests/test_halting.py`, mutants M152 to M172, `results/stage2i.json`, `results/stage2i_seed0.json`, `results/stage2i_seed1.json`, `results/stage2i_seed2.json`. The sha256 of `PREREG_stage2i.md` at the commit that preceded every run, `7bca013ac318571682023795a6ec8afdb94896f09b7326dfe03e2c57e98175b2`, was added to `results/stage2i.json` by hand (field `prereg_sha256`) at the result commit, as the pre-registration said; the file on disk matches it.

TL:DR. Five exit rules that read only what a token has already computed were tested against fixed depth on the three frozen Stage 2H looped models, in the registered region of mean depth 1.25 to 2.70. None came close to the 0.01 nat bar, so P18 is not supported. `probe_state` was best at +0.0034 (about 5.4% of the oracle's gap), `probe_small` was next at +0.0028 (4.5%), `change` tied with fixed depth as in Stage 2D (+0.0014), and `entropy` was worse by 0.0305 (X_WORSE). `margin` is NOT_SCORED because the registered quantile grid left only two frontier points in the region on every seed, which is a coverage result and not a performance one, and by inspection it trails fixed depth like `entropy`. The probes barely win because the gain of an extra pass is almost unpredictable from these features (about 1% of its variance explained at pass 1 and about zero later). The oracle's 0.0626 headroom rests on information none of the tested signals carry. Per PREREG section 5, the RL question goes to Nate with the number that the best simple rule closes about 5.4% of the oracle gap. No RL was run and RL stays paused.
