# Stage 2F pre-registration: separate learning rates for COUNT's block matrices and its embeddings

Dated 2026-11-04. Committed before any Stage 2F run. Origin: flagged follow-up 3A of the Stage 2 local-minima audit (neural vault, `research-logs/2026-11-04 connectome-lm Stage 2 synthesis, local minima and follow-up avenues.md`). Code: `src/clm/s2model.py` (`param_groups`, `adam_mult`), `src/clm/s2train.py` (`--adam-mult`, per-group rate), `src/clm/rules2f.py` (the rules of section 4), `scripts/stage2f_driver.py`, `scripts/stage2f_analyze.py`. Config `configs/stage2f.json`.

## 1. Question

**P12.** COUNT-M trains its 2-D block matrices with Muon and its embeddings, head and (in later arms) gates with AdamW, both at one shared rate. Stage 2A's pilot gave COUNT a shared rate of 0.032, four times B0's 0.008, and the pilot was still improving at the top of its grid (loss 3.5653, 3.5249, 3.4756, 3.4640 at 4e-3, 8e-3, 16e-3, 32e-3). COUNT reads its memory through `head.weight`, which AdamW trains, so the audit's guess is that COUNT needed a fast AdamW rate and the shared rate pushed Muon up with it. Does letting the two groups run at different rates lower COUNT-M's validation loss by at least 0.01 nats?

Described, no decision: which cell wins, and whether the best Muon rate returns to B0's 8e-3 while the AdamW rate stays high.

A history is the last few tokens of the text so far. Cross-entropy is the model's average surprise per token, in nats. Muon uses `adjust_lr_fn="match_rms_adamw"`, so a Muon rate and an AdamW rate are on the same scale.

## 2. Fixed design

- **Arm.** COUNT-M exactly as Stage 2A: exact n-gram read, no gate (`--count-gate` off), no buckets, leave-one-out on, Muon on the block matrices, AdamW on everything else, weight decay 0.1 on the block matrices only, warmup 200 steps then linear decay to zero, clip 1.0, bf16, GPU memory cap 0.75, context 512, seeds as stated below. The only change is `adam_mult`: the AdamW group runs at `adam_mult` x the Muon rate, through the whole schedule (the schedule multiplies both). `adam_mult` 1.0 reproduces the shared rate exactly.
- **Gated read not tested.** Stage 2E made the gated read the COUNT read of later stages, but its gates also train under AdamW and its own result is not yet separated into scale and structure (limit 1 of `PREREG_stage2e.md`). Testing the decoupled rate on the ungated read keeps this question separable from that one and pairs against Stage 2A's three existing COUNT seeds at no extra cost. A gain here is a reason to repeat the test on the gated read, not a result about it.
- **Pilot grid.** COUNT-M, seed 0, 50,000,000 tokens with the schedule compressed to that length, Muon rate x AdamW rate in {8e-3, 16e-3, 32e-3} x {8e-3, 16e-3, 32e-3, 64e-3}: 12 cells. The three cells with equal rates (8e-3, 16e-3, 32e-3), and the 4e-3 cell outside the grid, are Stage 2A's pilot runs (`runs2/pilot_count/`) and are not rerun. Nine cells are new. Each rate ratio is a power of two, so the arithmetic of the two rates is exact.
- **Edge rule.** If the winning cell sits on an edge of the grid (Muon 8e-3 or 32e-3, AdamW 8e-3 or 64e-3), one more cell is run past each edge it sits on, once (halving or doubling that rate, the other rate unchanged), and the winner is chosen again over all cells. No further extension.
- **Confirmation (section 4).** Seeds 0, 1, 2 at 200M tokens (12,207 steps of 32 x 512) at the winning cell, run as `runs2/count_decoupled-M-s`, against Stage 2A's `runs2/count-M-0`, `-1`, `-2` (not rerun, shared rate 0.032). Same data order and name-keyed initialisation per seed.
- **Metric.** As Stage 2A: mean cross-entropy on the first 2,097,152 tokens of `data/fineweb/val.bin` at the last step. Also recorded: seconds per step, peak VRAM, peak host RSS.
- **B0 control.** Only if the confirmation runs. One B0-M pilot (no memory, seed 0, 50M tokens) at the winning cell, to be set beside the Stage 2.0 B0-M pilot at the shared 8e-3 (3.664853, `runs2/pilot/muon-0.008`). Described, no decision (limit 3).

## 3. Procedure

Runs go under `systemd-run --user`, GPU memory cap 0.75. `embeddinggemma` stopped for the runs and restarted after. One driver, `scripts/stage2f_driver.py`, resumable. Order: the nine new pilot cells, the edge cells if any, the choice (`runs2/pilot_2f/choice.json`). If the choice asks for a confirmation: the B0 control pilot, then three seeds of `count_decoupled`, then the analysis, then seeds 3 and 4 of both arms only if section 4 asks. Every run records `git_dirty`, and the analysis rejects a dirty run or a run whose pre-registration hash differs.

Time, laptop RTX 3050: about 22 min per pilot, so about 3.3 h for nine, up to 45 min more for edge cells; about 82 min per 200M-token run, about 4.1 h for three, about 5.5 h more for seeds 3 and 4 (four runs). No money.

## 4. Decision rule

Paired by seed, d_s = loss(decoupled) - loss(COUNT shared rate), computed by `clm.analyze.verdict` (the Stage 1 rule). SESOI 0.01 nats.

- **Pilot gate.** The 200M-token runs happen if and only if the lowest-loss cell is one with unequal rates and its pilot loss is at least 0.01 below the lowest loss among the equal-rate cells (`clm.rules2f.confirm`). Otherwise P12 is "not supported at the pilot", the shared rate stays, and nothing else runs. The 50M-token pilot is a short-horizon proxy on one seed; the gate only decides whether the longer runs are worth their 4 h.
- **X_BETTER:** mean d <= -0.01 and every d_s < 0.
- **X_WORSE:** mean d >= +0.01 and every d_s > 0.
- **NO_MEANINGFUL_DIFFERENCE:** |mean d| < 0.01 and every |d_s| < 0.02.
- **INCONCLUSIVE:** anything else.
- **P12:** supported if and only if the verdict is X_BETTER.
- **Five-seed rule:** if the verdict is INCONCLUSIVE on three seeds with |mean d| < 0.02, seeds 3 and 4 are added for both arms, COUNT included (trained with Stage 2A's settings: shared rate 0.032, the Stage 2A pre-registration), and the rule is applied once more on five seeds (`clm.rules2e.needs_seeds`).

## 5. What a verdict does

- **P12 supported:** the winning pair of rates becomes the COUNT recipe for ungated COUNT. The next test repeats the decoupling on the gated read (Stage 2E arm), as a new pre-registration. Stage 2A's COUNT-over-B0 gap (-0.0798) is then described against the B0 control; it is not recomputed.
- **P12 not supported at the pilot, or NO_MEANINGFUL_DIFFERENCE, or X_WORSE:** written up. The shared rate stays and follow-up 3A is closed for ungated COUNT.
- Either way the scale-only control from `PREREG_stage2e.md` section 6 keeps its own recipe (shared rate 0.032, the gated read of Stage 2E) so that it compares against the Stage 2E runs.

Every outcome is written up with the pilot table (all cells), the verdict table and the cost.

## 6. Conduct

As Stage 2E. A diverged run (non-finite loss or gradient) is rerun once from scratch; a second divergence is reported and gives no verdict. A diverged pilot cell is excluded from the choice and listed. No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash. After any change to `src/` the suite and `scripts/mutation_audit.py` must pass. Planted-bug mutants guarding this stage:

- M114-M115, the rates: the AdamW group not scaled; the Muon group scaled instead.
- M116: the per-step rate ignoring the group's multiple.
- M117: the command-line multiple not reaching the config.
- M118-M122, the rules: choosing the highest loss, an edge extension in the wrong direction, an already tried cell run again, the pilot gate at any gain instead of the SESOI, the comparison made against one fixed shared-rate cell instead of the best.

## 7. Limits stated in advance

1. **One seed and 50M tokens decide the cell.** The best pilot cell may not be the best at 200M tokens, and a cell worse at the pilot could win at the long run. The confirmation tests the one chosen cell, not the grid.
2. **Weight decay moves with the Muon rate.** The decay applied to the block matrices is rate x 0.1 per step, so changing the Muon rate also changes its decay strength; it is not tuned separately. The AdamW group has no decay.
3. **COUNT gets more tuning than B0.** B0 was tuned over a coupled grid only. If decoupling also helps B0, a COUNT gain from it is partly a gain from better tuning, not from the memory. The B0 control pilot is one cell and tells only whether B0 at that cell moves in the same direction at 50M tokens.
4. **Ungated COUNT only** (section 2).
5. **Stage 2A's equal-rate pilots come from an earlier commit** (`b4dcde9` for the 3.4640 cell). The ungated read and the rate arithmetic at `adam_mult` 1.0 are unchanged by this code, but those four cells are not rerun.
6. **One model size (13.8M non-embedding parameters), one corpus.** Three seeds resolve about 0.01 nats and no finer.

TL:DR. We test whether giving COUNT-M's embeddings, head and gates their own learning rate, separate from the Muon block matrices, lowers validation loss by at least 0.01 nats against Stage 2A's shared rate. A 12-cell pilot grid (nine new cells, three reused) on one seed at 50M tokens picks the pair of rates; three 200M-token seeds run only if the best separate-rate cell beats the best shared-rate cell by 0.01 at the pilot (P12). Supported only on a clean X_BETTER; otherwise the shared rate stays and follow-up 3A is closed. Ungated COUNT only. The gated read and the scale-only control from Stage 2E are untouched. About 3.3 h for the pilot, 4.1 h more if it triggers, no money.
