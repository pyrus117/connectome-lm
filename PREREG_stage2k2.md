# Stage 2K2 pre-registration: does a gate, or a scalar, help COUNT when its parameters train on a slower rate?

Dated 2026-11-10. Committed before any Stage 2K2 run. Origin: Stage 2K (`notes/stage2k_result_2026-11-10.md`) found that on separate Muon and AdamW rates the gated COUNT read was worse than plain COUNT by 0.0461 nats and the one-scalar control worse still (0.0770), both on every seed. The gate and the scalar had trained at the AdamW rate of 64e-3, eight times Stage 2E's gate rate and untuned (Stage 2K limit 2). The note listed three options not run: the gate and scalar on a lower rate, a per-order scalar, and seeds 3 and 4. Nate asked for them on 2026-11-10. This stage runs all three as one design. Code: `src/clm/rules2k2.py`, `src/clm/s2model.py` (`param_groups(..., gate_mult)`, per-order `count_scale`), `src/clm/s2train.py` (`--gate-mult`, `--count-scale-orders`), `scripts/stage2k2_driver.py`, `scripts/stage2k2_analyze.py`. Config `configs/stage2k.json` (unchanged). Tests `tests/test_s2k2.py`, mutants M186 to M198.

## 1. Questions

Cross-entropy (validation loss) is the model's average surprise per token in nats; lower is better.

**P20.** With their parameters on a slower learning rate, does any of three COUNT reads (the gated read, one learned scalar, one learned scalar per order) beat plain COUNT by at least 0.01 nats on every one of five seeds?

**Carry-forward.** Which COUNT read do later stages use? Decided by a rule fixed here (section 3), so that the Stage 2L stage (COUNT plus the looped model) needs no further choice.

## 2. Fixed design

- **Recipe for every arm.** The Stage 2K recipe: COUNT-M, Muon at 8e-3 on the block matrices, AdamW at 64e-3 (`adam_mult` 8) on the embeddings, head and value embeddings, weight decay 0.1 on the block matrices only, warmup 200 steps then linear decay to zero, clip 1.0, bf16, GPU memory cap 0.75, context 512, 200M tokens (12,207 steps of 32 x 512), exact n-gram tables only, leave-one-out reads while training. Same data order and name-keyed initialisation per seed.
- **The one change.** The gate parameters (the gate table and absent vectors of the gated read, and the scalar or scalars of the scaled reads) train in their own AdamW group at `gate_mult` x 8e-3 with `gate_mult` = 1.0, so at 8e-3, with no weight decay. That is the Muon rate: eight times slower than in Stage 2K and four times slower than in Stage 2E (32e-3). One setting only, chosen as the Muon rate, not tuned.
- **Arms**, all on q-pc, five seeds (0 to 4):
  - (a) **COUNT**: `runs2/count_decoupled_q-M-s`. Seeds 0 to 2 are Stage 2K's runs, not rerun. Seeds 3 and 4 are new, same recipe and machine.
  - (b) **gated_low**: `runs2/count_gated_low-M-s`. The Stage 2E gated read (a weight per order, source and count bin, plus absent vectors), gate parameters at 8e-3.
  - (c) **scaled_low**: `runs2/count_scaled_low-M-s`. One learned scalar on the whole read, at 8e-3.
  - (d) **oscaled_low**: `runs2/count_oscaled_low-M-s`. One learned scalar per order (four parameters, one for each of orders 2 to 5), at 8e-3. It sits between the single scalar and the 1,568-parameter gate.
  - All three candidates start bit-identical to COUNT (tested).
- **Not extended:** Stage 2K's high-rate gated and scaled arms get no seeds 3 and 4. Each lost to COUNT by 0.03 to 0.11 on every seed, and the seed spread of COUNT is about 0.03, so two more seeds cannot change any decision that rests on them. Seeds 3 and 4 run for every arm that enters a decision here. The high-rate arms' existing three seeds are used only to describe low rate against high rate.
- **Metric.** As Stage 2A: mean cross-entropy on the first 2,097,152 tokens of `data/fineweb/val.bin` at the last step. Recorded also: seconds per step, peak VRAM, host RSS, final gate table, absent norms, final scalars. `scripts/bucket_eval.py` re-evaluates each checkpoint by 4-token history count class; its overall loss must match `results.json` within 0.002. Described, not decided.

## 3. Decision rule

Paired by seed, d_s = loss(candidate) - loss(COUNT), `clm.analyze.verdict` (the Stage 1 rule), SESOI 0.01, on five seeds.

- **X_BETTER:** mean d <= -0.01 and every d_s < 0.
- **X_WORSE:** mean d >= +0.01 and every d_s > 0.
- **NO_MEANINGFUL_DIFFERENCE:** |mean d| < 0.01 and every |d_s| < 0.02.
- **INCONCLUSIVE:** anything else.
- **P20** is supported if and only if at least one candidate's verdict against COUNT is X_BETTER.
- **Carry-forward** (`clm.rules2k2.carry_forward`): among the candidates that are X_BETTER against COUNT, the one with the lowest mean loss; plain COUNT if none is. Stage 2L uses this read, at `gate_mult` 1.0 if it is one of the candidates.
- Described, no decision: the verdicts of the candidates against each other; class gaps against COUNT; the mean gate table and absent norms; the learned scalars; each candidate's loss against its Stage 2K high-rate counterpart (the same seeds 0 to 2); the sample SD of COUNT's five losses; cost.

## 4. Stated expectation, before any run

In Stage 2K the learned scalar ended near 9 and did worse than the fixed value 1, which fits a parameter that overshoots at 64e-3. A slower rate may let the gate and scalars settle where Stage 2E's gate did at 32e-3, and Stage 2E found a thin gain (0.0111). But Stage 2F and 2G showed that separate rates already lift plain COUNT, which may leave nothing for the read to add. The expectation is that the candidates land near COUNT: NO_MEANINGFUL_DIFFERENCE or INCONCLUSIVE, with the gated read and the per-order scalar the likeliest to show a small gain. A win of 0.01 on every one of five seeds is not the likely outcome, given a COUNT seed spread of about 0.03 (Stage 2K), though the paired differences between arms of the same seed may spread less than the raw losses. The rule decides, not this guess.

## 5. What each outcome does

- **A candidate carried forward:** Stage 2L (COUNT plus the looped model) trains with that read. The gate line reopens at this rate only; whether a size-L check follows goes to Nate.
- **Plain COUNT carried forward:** the gate line closes at this size, and Stage 2L uses plain COUNT. Nothing here changes gate 1.
- Every outcome is written up with the per-class table, the gate table, and the low-rate against high-rate comparison.

## 6. Procedure

All training and analysis on q-pc, detached, GPU memory cap 0.75, driver `scripts/stage2k2_driver.py` (resumable). Order: for each seed 0 to 4, the arms COUNT (seeds 3 and 4 only), gated_low, scaled_low, oscaled_low; then `bucket_eval.py` on every run and the analysis. Before the first run: the branch is synced; the test suite and `scripts/mutation_audit.py` have passed on the committed code; the q-pc tree is clean; a 200-step smoke run of each new arm checks that all three start at COUNT's first-step loss and that the gate rate reaches the optimiser (its validation loss is not read). The analysis rejects a dirty run, a run whose pre-registration hash is not the file it should carry (Stage 2K's for COUNT seeds 0 to 2, this file's for the rest), a run at other rates, or a candidate whose `gate_mult` is not 1.0.

Time: 17 runs at about 75 minutes (Stage 2K measured 0.34 s per step on q-pc), about 21 hours. No money.

## 7. Conduct

A diverged run (non-finite loss or gradient) is rerun once from scratch; a second divergence is reported and gives no verdict. No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash. After any change to `src/`, the test suite and `scripts/mutation_audit.py` must pass. Planted-bug mutants guarding this stage: M186 (per-order scale tiled instead of interleaved), M187 to M189 (the gate group: never separate, its rate multiplier lost, weight decay applied), M190 to M191 (the per-order arm's name and size), M192 to M194 (the refusals and the rate reaching the optimiser), M195 to M198 (the section 3 rules). Seven earlier mutants (M21, M108, M173, M175 to M177, M179) were re-anchored to lines this stage rewrote; each still plants the same bug.

## 8. Limits stated in advance

1. **One slower rate, not tuned.** 8e-3 was chosen as the Muon rate. A different rate could do better or worse, and this stage cannot say.
2. **Best of three.** Three candidate reads are tried against COUNT and P20 asks whether any wins; no correction for three tries is applied. If exactly one wins on the registered rule, treat it as a screening result and a candidate for a check at another seed set.
3. **Seed spread.** Stage 2K found plain COUNT's seed SD about 0.03 on this recipe. Five seeds with a rule that needs every seed below zero resolve 0.01 only if the paired spread is much smaller than that; an INCONCLUSIVE verdict is a likely and honest outcome.
4. **COUNT seeds 0 to 2 are from Stage 2K**, trained at an earlier commit than the new runs. The COUNT path through `param_groups` is unchanged when no gate rate is set (tested), but those runs are not rerun.
5. **One machine, one size, one corpus.** q-pc for all runs; 13.8M non-embedding parameters; FineWeb-Edu.
6. **The scalars add 1 or 4 parameters** and the gate 1,568, negligible against 13.8M.
7. **Class evaluation by 4-token history counts only**, as in Stage 2E.

TL:DR. In Stage 2K the gated COUNT read and a one-scalar control both lost to plain COUNT, but their parameters trained at an untuned 64e-3. This stage reruns them with those parameters at 8e-3, adds a per-order scalar (four parameters), and runs five seeds each (COUNT's seeds 3 and 4 are new). P20 asks whether any of the three beats COUNT by 0.01 on every seed. The read that wins by the registered rule, or plain COUNT if none does, is what Stage 2L (COUNT plus the looped model) uses. The expectation is that all three land near COUNT; a clean win is unlikely given a COUNT seed spread of about 0.03. About 21 hours on q-pc, no money.
