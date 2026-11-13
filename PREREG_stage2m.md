# Stage 2M pre-registration: why were the loop passes inert on the separate-rate recipe?

Dated 2026-11-14. Committed before any Stage 2M run (the one smoke run in section 6 excepted). Origin: Nate's request on 2026-11-14, after Stage 2L (`notes/stage2l_result_2026-11-13.md`). Stage 2L's looped models learned nothing from their extra passes. Run a diagnosis: test COUNT on Stage 2H's shared-rate loop, and/or check whether AdamW broke the loop. Code: `src/clm/s2model.py` (`param_groups`, new `gain_mult`), `src/clm/s2train.py` (new `gain_mult` setting and `--gain-mult` flag), `src/clm/rules2m.py`, `scripts/stage2m_driver.py`, `scripts/stage2m_analyze.py`. Config `configs/stage2m.json` (identical to `configs/stage2l.json` except the pre-registration name). Tests `tests/test_s2m.py`. Results `results/stage2m.json`.

## 1. Question

Cross-entropy (validation loss) is the model's average surprise per token, in nats; lower is better. The *looped model* reuses one block pair several times in a row on each token. Each reuse is a *pass*, and its *depth* is how many passes were used (1 to 6). The *loop gain* (`loop_gain`, shape 6 x 384, initialised to 1, no weight decay) scales what each pass adds to the residual stream, per pass and per channel. A gain near zero switches that pass off. The *separate-rate recipe* trains the block matrices with Muon at 8e-3 and everything else with AdamW at 8 times that, 64e-3 (`adam_mult` 8.0). The *shared-rate recipe* of Stage 2H uses 8e-3 for both (`adam_mult` 1.0). The *loop's benefit* inside one run is its depth-1 validation loss minus its best-depth validation loss. *COUNT* is the read that looks up how often the recent words followed each other in the training text and feeds that to the model; the plain COUNT read is the one Stage 2K2 carried forward.

**Background.** Stage 2L trained the gained all-depth looped model without the read (LOOP) and with it (LOOP+COUNT) on the separate-rate recipe. All six depth losses came out equal within 0.00006 nats in every run. Loop benefit was 0.00003, 0.00003, 0.00005 for LOOP and 0.00003, 0.00003, 0.00000 for LOOP+COUNT (seeds 0, 1, 2). Stage 2H (`results/stage2h.json`, `both` arm, one shared rate, `adam_mult` 1.0, no read) had benefits 0.0607, 0.0651, 0.0587. The only difference in settings between Stage 2L's LOOP and Stage 2H's `both` is the AdamW rate on every non-block parameter (64e-3 against 8e-3). Both used micro-batch 4, the all-depth loss and the gain.

**Post-hoc observation.** Read on 2026-11-14 from the seed 0 checkpoints and logs of Stage 2L, before any Stage 2M run. This is evidence for the hypothesis below, not a test of it.

- Mean absolute loop gain per pass (passes 1 to 6) in the final checkpoint:

| Run | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Stage 2L LOOP | 0.371 | 0.064 | 0.014 | 0.013 | 0.012 | 0.012 |
| Stage 2L LOOP+COUNT | 0.318 | 0.055 | 0.016 | 0.013 | 0.012 | 0.012 |
| Stage 2H `both` (laptop, seed 0) | 0.385 | 0.313 | 0.199 | 0.061 | 0.012 | 0.003 |

- In both Stage 2L arms a few pass-1 channels reached gains of 14.8 and 18.5, which the Stage 2L note also reported.
- In Stage 2L LOOP the training loss on each depth was already almost the same by step 200 (5.3518, 5.3480, 5.3480, 5.3488, 5.3507, 5.3541), and at step 50 depth 6 was worse than depth 1 (6.3041 against 6.2851). In Stage 2H at step 1000 the depths still differed (4.3481 at depth 1 against 4.3085 at depth 2).
- AdamW moves a parameter by about its learning rate each step. Summed over the first 100 warmup steps, the 64e-3 rate can move a gain by about 1.6 and the 8e-3 rate by about 0.2, against a starting value of 1.

**H1.** The AdamW rate on `loop_gain` (8 times the shared rate in Stage 2L) switches passes 2 to 6 off during warmup, when extra passes still do not help. Once a pass's gain is near zero its blocks get almost no gradient through that pass, so they never learn to be useful, and the gain has no reason to recover.

**H2 (alternative).** Another parameter trained by AdamW at 64e-3 (embeddings, value embeddings, head, norms) is what kills the loop, or the COUNT read removes the loop's benefit at this size. Stage 2M does not test other AdamW parameters one by one.

**P22.** Is the loop's benefit at least 0.01 nats on every one of three seeds when only `loop_gain` is trained at the lower AdamW rate (GAINLOW)? Supported means the AdamW rate on the loop gain is enough to explain the inert loop.

**P23.** Is the loop's benefit at least 0.01 nats on every one of three seeds on Stage 2H's shared-rate recipe with the COUNT read added (SHARED)? Supported means a loop still earns something on top of COUNT at the shared rate.

Described, no decision: the contrasts listed in section 3.

## 2. Fixed design

- **Recipe for both arms.** Size M (13.8M non-embedding parameters), the gained all-depth looped model (Stage 2H's `both`) with the plain COUNT read at the input (the read Stage 2K2 carried forward, placed as in Stage 2L), 200M tokens, context 512, 12,207 steps of 32 x 512, micro-batch 4, warmup 200 steps then linear decay, clip 1.0, bf16, GPU memory cap 0.75, weight decay 0.1 on block matrices only, seeds 0, 1, 2. Same data order and name-keyed initialisation per seed.
- **Arms** (both new runs on q-pc):
  - **GAINLOW**: `runs2/loop_count_gainlow-M-s`. Stage 2L's LOOP+COUNT recipe (Muon 8e-3, AdamW 64e-3) except that only `loop_gain` trains at AdamW 8e-3 (`--gain-mult 1.0`), in its own parameter group, with no weight decay. This is the direct test of H1.
  - **SHARED**: `runs2/loop_count_shared-M-s`. Stage 2H's recipe (one shared rate, Muon 8e-3 and AdamW 8e-3, `--adam-mult 1.0`) plus the plain COUNT read. This is COUNT on Stage 2H's shared-rate loop.
- **Metric.** Validation loss at each depth 1 to 6 on the first 2,097,152 tokens of `data/fineweb/val.bin`, at the last step. Best depth comes from `clm.rules2h.summary`.

## 3. Decision rule

The decision quantity is the loop's benefit inside one run (section 1), computed per run. It is a within-run difference, so the seed-to-seed spread of plain COUNT (SD 0.03 to 0.05 on the separate-rate recipe) does not enter it. The SESOI, 0.01 nats, is used as the "alive" threshold. The inert loop sat at 0.00005 and Stage 2H's live loop at about 0.06, both far from 0.01. The rules are in `clm.rules2m` (`benefit`, `alive`, `futile`).

- **P22** is supported if and only if GAINLOW's benefit is at least 0.01 on every one of three seeds.
- **P23** is supported if and only if SHARED's benefit is at least 0.01 on every one of three seeds.
- **Futility stop**, as in Stages 2H and 2L. After seed 0 of both arms, if neither arm's seed-0 benefit reaches 0.01, P22 and P23 are both impossible (each needs every seed). Seeds 1 and 2 are not run, and both are reported NOT_SUPPORTED_FUTILITY with the seed-0 benefits. The driver prints only "FUTILITY continue" or "FUTILITY stop". Seed 0 of both arms runs in either case.
- **Reading table.**
  - *P22 and P23 both supported:* the loop works once its gain is not driven at 64e-3, and it adds to COUNT. GAINLOW is then a usable recipe.
  - *P22 supported, P23 not:* the rate explains it on the separate-rate recipe, but the loop earns nothing with COUNT at the shared rate. This would be a surprise, and it is checked against the described contrasts below.
  - *P22 not supported, P23 supported:* the gain rate alone is not enough. Another AdamW-trained parameter at 64e-3 is implicated (H2), and a further stage would split them.
  - *Neither supported:* the COUNT read removes the loop's benefit at this size (Stage 2H's loop without COUNT earned about 0.06). The loop line is closed and plain COUNT carries forward. This is consistent with the overlap expectation in `PREREG_stage2l.md` section 4.
- **Described, no decision.** Stage 1 verdict rule (`clm.analyze.verdict`, SESOI 0.01, paired by seed, d = first minus second, negative means the first is better), on best-depth loss:
  - GAINLOW against Stage 2L LOOP+COUNT (`runs2/loop_gain_all_count_sep-M-s`).
  - SHARED against Stage 2H `both` (`runs2/loop_gain_all-M-s`), which shows what the read adds to a live loop. Seed 0 of Stage 2H ran on the Linux laptop, seeds 1 and 2 on q-pc.
  - GAINLOW against plain COUNT on separate rates (`runs2/count_decoupled_q-M-s`).
  - SHARED against plain COUNT at the shared rate (`runs2/count-M-s`, Stage 2A).
  - SHARED against GAINLOW.
  - GAINLOW against Stage 2H `both`.
- **Also reported:** mean marginal gain of each extra pass per arm; the mean absolute loop gain per pass from each final checkpoint (GAINLOW, SHARED and Stage 2L LOOP+COUNT); seconds per step; peak VRAM; peak host memory.

## 4. Stated expectation, before any run

H1 is the leading hypothesis and the post-hoc rows point at it, so P22 is expected to hold, with a benefit near Stage 2H's. P23 is uncertain. Stage 2L's expectation was that a read that already supplies likely next tokens may reduce what an extra pass adds, and a benefit under 0.01 in SHARED would be that overlap. No outcome is certain. The rule decides.

## 5. What each outcome does

- No outcome changes gate 1 (Stage 2G) or opens gate 3 (a size-L check). That stays Nate's decision with a cost estimate, and none has been made.
- No RL is run in this stage; the RL gate stays paused (Nate, 2026-11-03).
- If P22 holds, the loop recipe for any later test is GAINLOW (or SHARED if it is better on best-depth loss, described only), and Nate decides the next step.
- Every outcome is written up in `notes/stage2m_result_<date>.md`.

## 6. Procedure

All training and analysis on q-pc, detached, GPU memory cap 0.75, driver `scripts/stage2m_driver.py` (resumable, run as `python scripts/stage2m_driver.py`). Order: seed 0 GAINLOW, seed 0 SHARED, the futility check, then seeds 1 and 2 of both arms if it continues, then the analysis.

Before the first run:

- The branch is synced to q-pc.
- The test suite and `scripts/mutation_audit.py` have passed on the committed code.
- The q-pc tree is clean (the untracked `results/stage2l.json` there is removed).
- A 400-step smoke run of GAINLOW on q-pc (`--tokens 6553600`, output under `runs2/smoke_2m`, not under a run name the analysis reads) checks three things. It trains. `loop_gain` is in its own optimizer group at 8e-3 while the other AdamW parameters are at 64e-3. Its first-step loss matches Stage 2L's LOOP+COUNT first step (9.0109 mean over depths at step 0 in the Stage 2L logs). The smoke run's validation loss is not read. Its per-depth training losses and gain rows are recorded in the result note as a pointer only.

The analysis rejects a dirty run, a run whose pre-registration hash is not this file's, or a run at other settings.

Time: Stage 2L measured 1.854 s per step for LOOP and 1.921 s for LOOP+COUNT on q-pc, so about 6.5 hours per run with the read. Two seed-0 runs about 13 hours; all six runs about 39 hours. No money.

## 7. Conduct

A diverged run (non-finite loss or gradient) is rerun once from scratch; a second divergence is reported and gives no verdict. No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash. After any change to `src/`, the test suite and `scripts/mutation_audit.py` must pass. Planted-bug mutants guarding this stage:

- M206 to M210: the gain not routed to its own group, its rate multiplier not applied during training, weight decay put on it, the setting not passed to the optimizer groups, the refusal of a gain rate without a loop gain removed.
- M211 to M214: benefit sign reversed, `alive` using any instead of all, `alive` threshold divided by ten, `futile` using or instead of and.

## 8. Limits stated in advance

1. **GAINLOW moves only the loop gain's rate.** It cannot say which other parameter, if any, matters (H2).
2. **The mechanism is a hypothesis.** A gain that reaches zero starving its blocks of gradient comes from seed 0 checkpoints of Stage 2L. The arms test the rate, not the mechanism.
3. **Equal tokens, not equal compute.** Looped arms do about 2.4 times the block passes per step of the plain models, so a loop benefit is not a compute-matched win.
4. **Three seeds.** The benefit decision is within-run, but the described best-depth contrasts inherit plain COUNT's seed spread (five-seed SD 0.0512 in Stage 2K2), so they are pointers only.
5. **SHARED against Stage 2H `both` crosses hardware and code.** Seed 0 of Stage 2H ran on the laptop, and the code has since changed (the loop code path with no read is tested bit-identical).
6. **One read, one placement of the read (the input), one size, one corpus.**
7. **Futility stop.** If it fires, P22 and P23 are reported not supported on one seed's evidence.
8. **The 0.01 "alive" threshold is a choice,** made because the two earlier cases sit far on either side of it.

TL:DR. Stage 2L's loop passes did nothing, and the one difference from Stage 2H's working loop was the AdamW rate on non-block parameters (64e-3 against 8e-3). The leading guess (H1) is that this rate switched the loop gain off for passes 2 to 6 during warmup. We run two arms with the plain COUNT read on q-pc: GAINLOW (only `loop_gain` at AdamW 8e-3, P22) and SHARED (Stage 2H's shared-rate recipe, P23). Each is supported only if the loop's benefit, depth-1 loss minus best-depth loss, is at least 0.01 nats on all three seeds. If seed 0 of both arms is below 0.01, the other seeds are not run. We expect P22 to hold and are unsure about P23. About 13 hours if the stop fires, 39 if not, no money.
