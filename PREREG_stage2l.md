# Stage 2L pre-registration: does the COUNT read help the looped model?

Dated 2026-11-10. Committed before any Stage 2L run. Origin: the Stage 2 audit note, follow-up 5B, second half (`neural-vault/research-logs/2026-11-04 connectome-lm Stage 2 synthesis, local minima and follow-up avenues.md`): test COUNT plus LoopGPT. Two tracks have each paid on their own at size M: COUNT beats B0-M by 0.1224 nats on separate learning rates (Stage 2G), and the gained all-depth loop beats Stage 2D's loop by 0.0384 nats (Stage 2H). Do the gains add? Code: `src/clm/s2model.py` (`B0GPT._input`), `src/clm/s2loop.py` (the count read in `LoopGPT`), `src/clm/s2train.py` (looped arch with a memory arm), `scripts/stage2l_driver.py`, `scripts/stage2l_analyze.py`; the decision rules are Stage 2H's, `src/clm/rules2h.py`, reused unchanged. Config `configs/stage2l.json`. Tests `tests/test_s2l.py`.

## 1. Question

Cross-entropy (validation loss) is the model's average surprise per token, in nats; lower is better. "Best-depth loss" is the lowest of a looped model's six per-depth validation losses (Stage 2H).

**P21.** Does the COUNT read lower the looped model's best-depth loss by at least 0.01 nats on every one of three seeds, both models on the separate-rate recipe?

Described, no decision: whether the two gains add up (how much the read is worth in the loop against in the plain model, and how much the loop is worth with the read against without it).

## 2. Fixed design

- **Recipe for every arm.** Size M (13.8M non-embedding parameters), Muon at 8e-3 on the block matrices, AdamW at 64e-3 on everything else, weight decay 0.1 on the block matrices only, warmup 200 steps then linear decay to zero, clip 1.0, bf16, GPU memory cap 0.75, context 512, 200M tokens (12,207 steps of 32 x 512), seeds 0, 1, 2. The looped arms use micro-batch 4 (the all-depth loss needs it). Same data order and name-keyed initialisation per seed.
- **The looped model.** Stage 2H's `both`: B0's six blocks as 2 prelude blocks, a 2-block loop shared across up to 6 passes, 2 coda blocks, a learned per-pass per-channel residual gain (`--loop-gain`), trained on the mean loss over all six depths (`--loop-loss all`).
- **The COUNT read in the loop.** Added once, at the input, exactly where B0 reads it: the count-weighted sum of head rows is added to the token embedding before the first RMS normalisation, and the whole stack then runs on that. Every pass of the loop sees the same input. Nothing else about the loop changes. With no count read the loop is bit-identical to Stage 2H's code path (tested).
- **Which read.** The COUNT read is the one Stage 2K2 carried forward (`results/stage2k2.json`, field `carry_forward`): plain COUNT, or the gated read, the scalar, or the per-order scalar at `gate_mult` 1.0. The driver reads it and records it in `runs2/stage2l_choice.json`. The same read, on the plain model, is the reference in section 3 (`runs2/count_decoupled_q`, `count_gated_low`, `count_scaled_low` or `count_oscaled_low`, seeds 0 to 2, from Stage 2K and 2K2).
- **Arms** (new runs on q-pc):
  - (a) **LOOP**: `runs2/loop_gain_all_sep-M-s`. The looped model with no count read, on the separate-rate recipe. Stage 2H's own runs used one shared rate and are not reused, so that the two arms differ only by the read.
  - (b) **LOOP+COUNT**: `runs2/loop_gain_all_count_sep-M-s`. The same with the read.
  - Described only: the plain model with the read (above), and B0-M on separate rates (`runs2/b0_separate-M-s`, trained on the laptop in Stage 2G).
- **Metric.** The Stage 2H metric: validation loss at each depth 1 to 6 on the first 2,097,152 tokens of `data/fineweb/val.bin`, at the last step.

## 3. Decision rule

Paired by seed, d_s = best-depth loss of LOOP+COUNT minus best-depth loss of LOOP, `clm.analyze.verdict` (the Stage 1 rule), SESOI 0.01.

- **X_BETTER:** mean d <= -0.01 and every d_s < 0. **X_WORSE:** mean d >= +0.01 and every d_s > 0. **NO_MEANINGFUL_DIFFERENCE:** |mean d| < 0.01 and every |d_s| < 0.02. **INCONCLUSIVE:** anything else.
- **P21** is supported if and only if the verdict is X_BETTER (`clm.rules2h.p17`, new arm LOOP+COUNT against reference LOOP).
- **Futility stop**, as in Stage 2H. After seed 0 of both arms: (best-depth loss of LOOP+COUNT seed 0) minus (best-depth loss of LOOP seed 0). If it is zero or more, X_BETTER is impossible (it needs every seed below zero), so seeds 1 and 2 are not run and P21 is reported as NOT_SUPPORTED_FUTILITY with the seed-0 difference. The driver does this and prints only "FUTILITY continue" or "FUTILITY stop". Seed 0 of both arms runs in either case.
- **Also reported, by the same verdict rule, no decision:** LOOP+COUNT against LOOP at depth 1 and at depth 6; LOOP+COUNT against the plain model with the same read (is looping worth anything on top of COUNT); the four effects (read in the plain model, read in the loop, loop in the plain model, loop with the read), each as the mean paired difference in best-depth loss (the plain model's loss is its final loss); mean marginal gain of each extra pass in both loop arms; cost.

## 4. Stated expectation, before any run

The two gains come from different places: COUNT reads the training corpus at the input, the loop spends more compute on each token. They could add. They could also overlap: Stage 2H found the loop's gain concentrated in the first extra pass, and a count read that already hands the model the likely next tokens may reduce what an extra pass can add. The expectation is a gain from the read in the loop that is smaller than its gain in the plain model (0.1224 against B0 at Stage 2G), but still well past 0.01; the read's gain is large enough that P21 is likely to be supported if the loop does not absorb it. No outcome is expected with confidence, and the two arms' seed spreads at this recipe are not known: Stage 2K found plain COUNT's seed SD about 0.03 on the separate-rate recipe. The rule decides.

## 5. What each outcome does

- **P21 supported:** COUNT plus the gained loop is the recipe for the next scaling test. Whether to open gate 3 (a size-L check) with it goes to Nate with the numbers and a cost estimate.
- **P21 not supported (including the futility stop):** written up. The tracks do not add at this size; which one carries forward goes to Nate.
- No outcome changes gate 1 (Stage 2G). No RL is run in this stage; the RL gate stays paused (Nate, 2026-11-03).

Every outcome is written up in `notes/stage2l_result_<date>.md`.

## 6. Procedure

All training and analysis on q-pc, detached, GPU memory cap 0.75, driver `scripts/stage2l_driver.py` (resumable). Order: seed 0 LOOP, seed 0 LOOP+COUNT, the futility check, then seeds 1 and 2 of both arms if the check continues, then the analysis. Before the first run: the branch is synced; the test suite and `scripts/mutation_audit.py` have passed on the committed code; the q-pc tree is clean; Stage 2K2 has finished and `results/stage2k2.json` exists; a 200-step smoke run of LOOP+COUNT with the carried read on q-pc checks that it trains and that its first-step loss matches the same read in the plain model's and the loop's own first-step loss (its validation loss is not read). The analysis rejects a dirty run, a run whose pre-registration hash is not this file's, or a run at other settings.

Time: a looped all-depth run takes about 6.3 hours on q-pc (Stage 2H: 1.8 s per step); LOOP+COUNT adds the lookup, so about 6.5 hours. Two seed-0 runs about 13 hours; with seeds 1 and 2, six runs about 39 hours. No money.

## 7. Conduct

A diverged run (non-finite loss or gradient) is rerun once from scratch; a second divergence is reported and gives no verdict. No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash. After any change to `src/`, the test suite and `scripts/mutation_audit.py` must pass. Planted-bug mutants guarding this stage: M199 to M202 (the count read dropped from the loop's input, from `forward`, from the all-depth loss, from the per-token readouts), M203 to M205 (the read not handed to the all-depth loss in training, not given to the evaluation, the arm name). A wrong `states` flag in the looped evaluation makes a gated model raise, so it cannot be a planted bug that survives; the run test covers it by that failure. The old test that the looped arch refuses a memory arm was replaced (`tests/test_s2loop.py::test_an_unknown_arch_is_refused`), because this stage removes that refusal on purpose.

## 8. Limits stated in advance

1. **One COUNT read, one size, one corpus.** The read is whichever Stage 2K2 carried forward; 13.8M non-embedding parameters; FineWeb-Edu.
2. **Equal tokens, not equal compute.** The looped arms do about 2.4 times the block passes per step of the plain models (Stage 2H limit 2), and LOOP+COUNT adds the table lookups.
3. **Seed spread.** Plain COUNT's seed SD on this recipe is about 0.03 (Stage 2K); the looped arms' spread is unknown. Three seeds resolve about 0.01 only if the paired spread is much smaller.
4. **The loop arms are rerun on the separate-rate recipe**, so they differ from Stage 2H's `both`, which used one shared rate. Statements that compare with Stage 2H's numbers compare across recipes.
5. **One placement of the read** (the input), as in B0. A read at a later pass or only at the coda is not tested.
6. **B0-M on separate rates** (the description of the effects) was trained on the laptop; the other plain-model arms on q-pc.
7. **Futility stop**: if it fires, P21 is reported as not supported on one seed's evidence.

TL:DR. We add the COUNT read to Stage 2H's gained all-depth looped model, at the input where B0 reads it, and ask whether it lowers best-depth loss by 0.01 on every one of three seeds, both on the separate-rate recipe (P21). The read is whichever Stage 2K2 carried forward. The loop is rerun without the read on the same recipe so the arms differ in one thing only. As in Stage 2H, if seed 0 shows no gain, the other seeds are not run. The expectation is a gain smaller than the read's gain in the plain model but past 0.01; the two could also overlap. About 13 hours if the stop fires, 39 if not, on q-pc, no money.
