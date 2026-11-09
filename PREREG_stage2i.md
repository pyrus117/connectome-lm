# Stage 2I pre-registration: can a deployable exit rule beat fixed depth in the gained looped model?

Dated 2026-11-09. Committed before any Stage 2I analysis is run. Its sha256 is recorded in `results/stage2i.json` by hand at the result commit (this stage trains nothing, so there is no `results.json` to carry it). Plan: `docs/superpowers/plans/2026-11-02-stage2-kenyon-memory.md` (section 10). Origin: the Stage 2 audit note, follow-up 5B (`neural-vault/research-logs/2026-11-04 connectome-lm Stage 2 synthesis, local minima and follow-up avenues.md`). Code: `src/clm/halting.py`, `src/clm/depth_eval.py` (`collect_signals`, `collect_fit`), `scripts/stage2i_analyze.py`. Tests `tests/test_halting.py`, mutants M152-M172.

## 1. Question

Stage 2D, and its repeat on the Stage 2H model (`notes/stage2h_oracle_result_2026-11-09.md`), found that an oracle which reads each token's true loss at every depth beats fixed depth by about 0.062 nats at equal mean depth, so the RL gate is OPEN. The oracle cannot be deployed. RL halting is paused (Nate, 2026-11-03). The audit asked for cheaper tests first. So:

**P18.** Does any exit rule that sees only what a token has computed by the pass where it stops (not its future loss) beat fixed depth by at least the SESOI (0.01 nats, the smallest effect that matters) at equal mean depth, on every one of the three Stage 2H models?

If a cheap rule closes much of the 0.062 gap, RL has little left to do. If none does, the RL question goes back to Nate with the measured share of the gap that a simple rule leaves open.

## 2. Fixed design

- **Models.** `runs2/loop_gain_all-M-0`, `-1`, `-2`: the Stage 2H `both` model (looped B0-M with a per-pass gain, trained on the mean loss over all six depths; 13.8M non-embedding parameters; depths 1 to 6). Frozen. Nothing is trained or changed. Seed 0 was trained on the Linux laptop and seeds 1 and 2 on q-pc; all analysis runs on q-pc.
- **Per-token signals at every pass r, from one full-depth run:** the loss; the entropy of the readout distribution at depth r (nats); the margin (top-1 minus top-2 logit of the readout at depth r); the relative change of the state, norm(x_r - x_(r-1)) / norm(x_r).
- **Rules.** A rule stops a token after the first pass r in 1..5 where its signal says stop, and at pass 6 otherwise.
  - `change`: stop if the change is at or below a threshold. This is Stage 2D's free-exit rule, kept as a check that nothing has changed since 2D.
  - `entropy`: stop if the entropy is at or below a threshold.
  - `margin`: stop if the margin is at or above a threshold.
  - `probe_small`, `probe_state`: for each pass r in 1..5 a ridge regression predicts the gain of the next pass (loss at depth r minus loss at depth r + 1) from the token's features at pass r. `probe_small` uses three features: entropy, margin, change. `probe_state` uses those three plus the 384-d state after the pass. A token stops if the predicted gain is below the price of a pass k(b) = 0.2 x exp(-b / tau), tau = 10 / ln(100), at each effort b = 0, 1, ..., 10 (Stage 2D's schedule).
- **Fitting set.** The last 4,096 windows of 512 tokens of `data/fineweb/train.bin`, at every 8th position (positions 7, 15, ...), about 262,144 samples per seed. The model saw these tokens once in training. Validation data is never used to fit anything.
- **Ridge.** Features standardised on the fitting set; intercept unpenalised; alpha chosen from {1e1, 1e2, 1e3, 1e4, 1e5} on the last 25% of the fitting samples (whole later windows), then refitted on all of them. One ridge per pass and per probe variant.
- **Thresholds** for `change`, `entropy`, `margin` are set on the fitting set, not on validation. At quantile q in {0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97}, the threshold is the value at which a share q of the pooled (pass, token) signal values for passes 1 to 5 would stop (low q for entropy and change, high for margin; `halting.level_for_share`). This traces each rule's whole frontier.
- **Evaluation.** The exit depths are chosen from the signals of one full-depth run on the first 2,097,152 tokens of `data/fineweb/val.bin` (4,096 windows of 512), then the loss is **executed**: a stopped token keeps its state and later tokens read it frozen (`depth_eval.executed_loss`, Stage 2D's convention). This is faithful for the loss. The signals themselves come from the unexecuted run (limit 1).
- **Fixed-depth frontier.** The points (r, validation loss at depth r), r = 1..6, of the same model.
- **Oracle.** For reference only, each effort's oracle depths executed, as in Stage 2D.

## 3. Decision rule

- **Region.** The mean-depth range where the Stage 2H oracle has room, 1.25 to 2.70 (its mean depth ran 1.25 to 2.69 over efforts 0 to 10, seed 0). Fixed here, from the oracle only, before any rule's frontier was computed.
- **Score of a rule on one seed.** Over the rule's frontier points whose mean depth lies in the region (at least 3, else the rule is NOT_SCORED on that seed): the mean executed loss of those points, and the mean loss of fixed depth at those same mean depths by linear interpolation. gap = fixed minus rule, so positive means the rule is better.
- **Verdict of a rule.** The Stage 1 paired rule (`clm.analyze.verdict`) on rule loss against fixed loss over the three seeds, SESOI 0.01: X_BETTER if the mean gap is at least 0.01 and every seed's gap is above zero; X_WORSE, NO_MEANINGFUL_DIFFERENCE and INCONCLUSIVE as in Stage 1.
- **P18 is supported if and only if at least one of `entropy`, `margin`, `probe_small`, `probe_state` is X_BETTER.** `change` is reported but does not count (it repeats Stage 2D).
- **Also reported:** each rule's mean gap and its share of the oracle's mean gap in the same region; the ridge's alpha and held-out error for each pass against the variance of the gain; every frontier point.

## 4. Stated expectation, before any run

Stage 2D's free exit gained nothing over fixed depth, and the Stage 2H oracle repeat shows the same (free exit against fixed depth: mean edge 0.0004). The expectation is that `change` again ties with fixed depth. Entropy and margin are cheap, standard signals; their chance of a 0.01 win is real but uncertain, and the fixed-depth curve is steep between depths 1 and 2 (0.053 nats) and flat after depth 3, so most of any win has to come from sending a minority of tokens past depth 1. `probe_state` has the most information and the most room to overfit. No outcome is expected with confidence. Every outcome is written up.

## 5. What each outcome does

- **P18 supported:** the winning rule or rules are written up with their gap and share of the oracle gap. They become candidates for the COUNT plus LoopGPT stage (audit follow-up 5B, second half). RL stays paused.
- **P18 not supported:** written up with the best rule's share of the oracle gap. The question of RL goes to Nate. No RL is run in this stage either way.
- No outcome changes the memory track.

## 6. Procedure

`scripts/stage2i_analyze.py` on q-pc, detached, writing `results/stage2i_seed<s>.json` per seed (a finished seed is skipped on rerun) and then `results/stage2i.json`. The pipeline was run once on a tiny fixture model with random tokens, to check that the pieces fit together; no real model's validation loss was read for any rule before this file was committed. Write-up in `notes/stage2i_result_<date>.md`.

## 7. Conduct

No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash. After any change to `src/`, the test suite and `scripts/mutation_audit.py` must pass. A rule that is added after seeing results is reported as exploratory, not as part of P18.

## 8. Limits stated in advance

1. **Signals from the full-depth run.** A deployed rule would read each token's signals while its neighbours are frozen at their own exit depths. Here the signals come from the run where every token ran every pass, so they are slightly optimistic. The loss is not: it is executed. The Stage 2H oracle repeat measured the cross-token effect on loss at about 0.05 nats, so the signal mismatch could matter and is not measured.
2. **Fitting data seen once.** The probes are fitted on training tokens that the model has seen one time, whose losses may be a little lower than unseen text's.
3. **Four candidate rules, best of four.** P18 asks whether any of them wins; no correction for four tries is applied. If exactly one wins, treat it as a screening result.
4. **Three models, one validation set.** Seeds are three trained models, not three data splits. The registered SESOI is 0.01 and B0-M's seed SD is about 0.0027.
5. **Region and price schedule are our choice.** Both are fixed in advance and neither is tuned.
6. **Linear probes only**, on at most 387 features, and a fixed alpha grid.
7. **The oracle uses the future.** Its share is a ceiling, not a target.
8. **One size, one corpus.** 13.8M non-embedding parameters, FineWeb-Edu.
9. **Cross-machine.** Seed 0 was trained on the Linux laptop and analysed on q-pc. The analysis checks that the re-evaluated depth-6 loss matches `results.json` to 2e-3.

TL:DR. We test whether a simple exit rule that reads only what a token has already computed (state change, entropy, margin, or a linear probe of the next pass's gain) beats fixed depth by 0.01 nats at equal mean depth on all three frozen Stage 2H looped models, in the mean-depth range 1.25 to 2.70 where the oracle has room. Probes and thresholds are fitted on the tail of the training data, never on validation; the loss is executed with frozen neighbours. The Stage 2D free-exit rule is kept as a check and is expected to tie with fixed depth. If a rule wins, it becomes a candidate for COUNT plus LoopGPT; if none does, the RL question goes back to Nate with the share of the oracle gap that a simple rule leaves open. No RL is run either way.
