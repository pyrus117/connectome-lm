# Stage 2G pre-registration: gate 1 again, with B0 on separate learning rates too

Dated 2026-11-05. Committed before any Stage 2G run. Origin: Stage 2F (`notes/stage2f_result_2026-11-05.md`) found that separate Muon and AdamW rates lower COUNT-M's loss by 0.0614 nats, and that one B0-M pilot at the same cell also gained 0.0240. Stage 2A's gate 1 (COUNT-M beats B0-M by -0.0798) compared COUNT on a tuned recipe with B0 on an untuned one. Code: `src/clm/rules2g.py`, `scripts/stage2g_driver.py`, `scripts/stage2g_analyze.py`. Config `configs/stage2g.json`. The `adam_mult` setting is Stage 2F's (`src/clm/s2model.py`, `src/clm/s2train.py`).

## 1. Question

**P13.** With both models on their best Muon and AdamW rates, does COUNT-M still beat B0-M by at least 0.01 nats? That is, does gate 1 of the Stage 2 plan still pass on equally tuned recipes?

Described, no decision: how much B0 itself gains from separate rates (B0 on separate rates against B0 on the shared rate).

Cross-entropy (validation loss) is the model's average surprise per token, in nats; lower is better.

## 2. Fixed design

- **B0-M.** The Stage 2.0 B0-M recipe (no memory, Muon on the block matrices at 8e-3, AdamW on the rest, weight decay 0.1 on the block matrices only, warmup 200 steps then linear decay to zero, clip 1.0, bf16, GPU memory cap 0.75, context 512, 200M tokens, seeds 0, 1, 2), with the AdamW group's rate a multiple of the Muon rate (`adam_mult`).
- **The Muon rate stays 8e-3.** B0's Stage 2.0 pilot found 8e-3 best at the shared rate, and Stage 2F found the Muon rate of the winning COUNT cell is also 8e-3 and matters little once the AdamW rate is high. Only the AdamW rate is tuned, which is a thinner search than COUNT's (limit 2).
- **Pilot.** B0-M, seed 0, 50,000,000 tokens with the schedule compressed to that length, AdamW rate in {8e-3, 32e-3, 64e-3, 128e-3}. 8e-3 is the Stage 2.0 pilot (`runs2/pilot/muon-0.008`, 3.664853) and 64e-3 is Stage 2F's B0 control (`runs2/pilot_2f/b0-m0.008-a0.064`, 3.640863); neither is rerun. 32e-3 and 128e-3 are new. If the winner is 128e-3, one run at 256e-3, once.
- **Rerun gate.** B0-M is rerun at the winning AdamW rate (three seeds, 200M tokens, `runs2/b0_separate-M-s`) if and only if that rate's pilot loss is at least 0.01 below the 8e-3 (shared) pilot (`clm.rules2g.separate`). Otherwise Stage 2.0's `runs2/b0-M-0`, `-1`, `-2` stay the B0 reference.
- **COUNT arm.** Stage 2F's `runs2/count_decoupled-M-0`, `-1`, `-2` (Muon 8e-3, AdamW 64e-3). Not rerun.
- **Metric.** As Stage 2A: mean cross-entropy on the first 2,097,152 tokens of `data/fineweb/val.bin` at the last step. Also recorded: seconds per step, peak VRAM.

## 3. Procedure

Runs go under `systemd-run --user`, GPU memory cap 0.75. `embeddinggemma` stopped for the runs and restarted after. One driver, `scripts/stage2g_driver.py`, resumable: the pilot cells, the choice (`runs2/pilot_2g/choice.json`), the B0 reruns if the gate asks, then the analysis. Every run records `git_dirty`; the analysis rejects a dirty run or one whose pre-registration hash differs. Time on the laptop: about 19 min per pilot (two, up to three), about 78 min per 200M-token B0 run (three). No money.

## 4. Decision rule

Paired by seed, d_s = loss(COUNT separate rates) - loss(B0 reference), computed by `clm.analyze.verdict` (the Stage 1 rule). SESOI 0.01.

- **X_BETTER:** mean d <= -0.01 and every d_s < 0.
- **X_WORSE:** mean d >= +0.01 and every d_s > 0.
- **NO_MEANINGFUL_DIFFERENCE:** |mean d| < 0.01 and every |d_s| < 0.02.
- **INCONCLUSIVE:** anything else.
- **P13 / gate 1:** passes if and only if the verdict is X_BETTER. An INCONCLUSIVE result is reported and the question of seeds 3 and 4 goes to Nate; they are not added automatically in this stage.

## 5. What a verdict does

- **X_BETTER:** gate 1 holds on equally tuned recipes and the memory track stands. The next steps stay as in Stage 2F section 5: the gated read and the scale-only control are repeated on the separate-rate recipe.
- **Anything else:** the Stage 2A conclusion that COUNT beats B0 does not hold on tuned recipes. This is written up before any further Stage 2 work, and the memory track's later results are read with that in mind.

Every outcome is written up with the pilot table, the verdict table and the cost.

## 6. Conduct

As Stage 2F. A diverged run is rerun once from scratch; a second divergence is reported and gives no verdict. A diverged pilot is excluded and listed. No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash. After any change to `src/` the suite and `scripts/mutation_audit.py` must pass. Planted-bug mutants guarding this stage: M123 (choosing the highest loss), M124 (the edge rate in the wrong direction), M125 (an already tried rate run again), M126 (the rerun gate at any gain instead of the SESOI).

## 7. Limits stated in advance

1. **One seed and 50M tokens decide the rate**, as in Stage 2F.
2. **B0 gets a thinner search than COUNT did.** COUNT's grid was 12 cells plus two edge cells over both rates; B0's is four AdamW rates at one Muon rate (two of them already measured). If B0's true best sits at another Muon rate, the gate is slightly harder on COUNT than it should be. The 16e-3 AdamW rate is not tried.
3. **Stage 2F's COUNT runs and Stage 2.0's B0 runs come from earlier commits.** The shared-rate arithmetic is unchanged by the code since, but those runs are not rerun.
4. One model size (13.8M non-embedding parameters), one corpus. Three seeds resolve about 0.01 nats and no finer.

TL:DR. We test whether COUNT-M still beats B0-M by at least 0.01 nats when both use separate Muon and AdamW rates (P13, gate 1 again). A two-cell B0 pilot (plus the two cells already measured) picks B0's AdamW rate; B0 is rerun for three 200M-token seeds only if that rate beats the shared rate by 0.01 at the pilot, otherwise the Stage 2.0 runs stand. COUNT is Stage 2F's three runs. X_BETTER means the memory track stands; anything else is written up before more Stage 2 work. B0 gets a thinner rate search than COUNT did. About 40 min for the pilot and 4 h for the reruns, no money.
