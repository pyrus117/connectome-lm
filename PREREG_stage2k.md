# Stage 2K pre-registration: does the gated COUNT read survive separate learning rates, and is its gain structure or scale?

Dated 2026-11-09. Committed before any Stage 2K run. Origin: Stage 2E found the gated COUNT read beat plain COUNT by 0.0111 nats at one shared learning rate (`notes/stage2e_result_2026-11-04.md`), with exact gates grown to 13 to 90 against a start of 1, so the gain could be scale alone (PREREG_stage2e.md limit 1). Stage 2F then found separate Muon and AdamW rates lower COUNT-M's loss by 0.0614 nats, and Stage 2G confirmed COUNT-M still beats B0-M on those rates (`notes/stage2g_result_2026-11-05.md`). Stage 2G section 5 left two things open: repeat the gated read and the scale-only control on the separate-rate recipe. This stage does both. Code: `src/clm/rules2k.py`, `src/clm/s2model.py` (`B0Config.count_scale`), `src/clm/s2train.py` (`--count-scale`), `scripts/stage2k_driver.py`, `scripts/stage2k_analyze.py`, with class losses from `scripts/bucket_eval.py`. Config `configs/stage2k.json`. Tests `tests/test_s2k.py`, mutants M173 to M185.

## 1. Questions

Cross-entropy (validation loss) is the model's average surprise per token in nats; lower is better.

**P19.** On separate Muon and AdamW rates, does the gated COUNT read still beat plain COUNT by at least 0.01 nats on every seed?

**Reading (what the gain is).** The scale-only control is COUNT with one learned scalar on the whole memory read (no per-order, per-source or per-count-bin split, no absent vectors). Three paired contrasts tell the story: gated against COUNT, gated against the scalar control, and the scalar control against COUNT. If the gate beats the control, structure matters. If the control beats COUNT as much as the gate does, the gain was scale.

## 2. Fixed design

- **Recipe for every arm.** COUNT-M as in Stage 2F: Muon at 8e-3 on the block matrices, AdamW at 64e-3 (`adam_mult` 8) on the embeddings, head, value embeddings and every gate or scalar, weight decay 0.1 on the block matrices only, warmup 200 steps then linear decay to zero, clip 1.0, bf16, GPU memory cap 0.75, context 512, 200M tokens (12,207 steps of 32 x 512), exact n-gram tables only (`data/count`, orders 2 to 5, 8 continuations per history), leave-one-out reads while training, seeds 0, 1, 2. Same data order and name-keyed initialisation per seed. Nothing is tuned in this stage: the rates are the Stage 2F and 2G cell.
- **Arms** (all new runs, trained on q-pc):
  - (a) **COUNT**: `runs2/count_decoupled_q-M-s`, plain COUNT, rerun on the same machine and code as the other two arms.
  - (b) **COUNT-gated**: `runs2/count_gated_sep-M-s`. The Stage 2E read: a learned scalar per order, source and count bin, plus a learned vector for an order with no stored history (`B0GPT._gate`). Starts bit-identical to COUNT (tested). The 1,568 gate parameters train with AdamW in the group of the other non-matrix parameters: now 64e-3, weight decay 0.
  - (c) **COUNT-scaled**: `runs2/count_scaled_sep-M-s`. One learned scalar, initialised to 1, multiplies every score of the read before it is added to the embedding. One parameter, same AdamW group. Starts bit-identical to COUNT (tested). It is the registered control for Stage 2E's limit 1.
  - Not run: the Kenyon-gated and random-gated arms (the bucket line is closed at this size, Stage 2E P11).
- **Why COUNT is rerun.** Stage 2F's COUNT runs (`runs2/count_decoupled-M-s`) were trained on the Linux laptop. The effect to detect is about 0.01 nats, and cross-machine floating-point differences were never measured. Rerunning plain COUNT on q-pc puts all three arms on one machine, one code version and one tree, so the paired contrasts carry no machine difference. The Stage 2F runs are kept as a described cross-check: their per-seed difference from the q-pc COUNT runs is the first direct measure of the machine effect on this model (no decision uses it).
- **Metric.** As Stage 2A: mean cross-entropy on the first 2,097,152 tokens of `data/fineweb/val.bin` at the last step. Also recorded: seconds per step, peak VRAM, peak host RSS, the final gate table and absent-vector norms (gated), the final scalar (scaled).
- **Class evaluation.** `scripts/bucket_eval.py` re-evaluates each final checkpoint by the stored count of the exact 4-token history, in the five classes of Stage 2B (not stored, 1-9, 10-99, 100-999, >=1000). Its overall loss must match `results.json` within 0.002. Described, not decided.

## 3. Decision rule

Paired by seed, d_s = loss(x) - loss(y), computed by `clm.analyze.verdict` (the Stage 1 rule). SESOI 0.01 nats.

- **X_BETTER:** mean d <= -0.01 and every d_s < 0.
- **X_WORSE:** mean d >= +0.01 and every d_s > 0.
- **NO_MEANINGFUL_DIFFERENCE:** |mean d| < 0.01 and every |d_s| < 0.02.
- **INCONCLUSIVE:** anything else.

Contrasts: `gated_vs_count`, `gated_vs_scaled`, `scaled_vs_count`.

- **P19** is supported if and only if `gated_vs_count` is X_BETTER.
- **Reading** (`clm.rules2k.reading`), one of:
  - **NOT_HOLDING:** P19 not supported.
  - **STRUCTURE:** P19 supported and `gated_vs_scaled` is X_BETTER.
  - **SCALE:** P19 supported, `gated_vs_scaled` is not X_BETTER, and `scaled_vs_count` is X_BETTER.
  - **UNRESOLVED:** P19 supported, `gated_vs_scaled` is not X_BETTER, and `scaled_vs_count` is not X_BETTER.
- **Five-seed rule**, decided once on three seeds: if `gated_vs_count` or `gated_vs_scaled` is INCONCLUSIVE with |mean d| under 0.02, seeds 3 and 4 are run for all three arms and every rule above is applied once more on five seeds.
- Described, no decision: class gaps for each contrast; the mean gate table and absent norms; the learned scalar per seed; cost; the cross-machine check against Stage 2F's COUNT runs.

## 4. Stated expectation, before any run

Two things in the earlier results point to scale. The Stage 2E exact gates grew to 13 to 90, a large overall boost to the memory. And Stage 2F found the AdamW rate is what lifted COUNT, which means plain COUNT can already scale its read by moving the head rows quickly. So the expectation is that the gated read's 0.0111 edge shrinks on separate rates, and that where an edge remains the scalar control takes much of it. The registered rule, not this guess, decides. SCALE or NOT_HOLDING seems more likely than STRUCTURE. No outcome is expected with confidence, and a thin result of the Stage 2E kind (clearing the bar by a tenth of the SESOI) would be INCONCLUSIVE or trigger the five-seed rule.

## 5. What each outcome does

- **STRUCTURE:** the gated read is the COUNT read of later stages, including the COUNT plus LoopGPT stage. Size L with the gated read goes to Nate with the numbers.
- **SCALE:** the one-parameter scalar is the COUNT read of later stages. The gate table is not carried forward.
- **NOT_HOLDING:** COUNT stays ungated at the separate-rate recipe and the gate line closes at this size.
- **UNRESOLVED:** written up. The five-seed rule has already run if it applied. Which read to carry forward goes to Nate.
- No outcome changes gate 1 (Stage 2G).

Every outcome is written up with the per-class table, the gate table and the cost.

## 6. Procedure

All training and analysis run on q-pc (RTX 4060 Laptop 8 GB, Windows), detached, GPU memory cap 0.75, driver `scripts/stage2k_driver.py` (resumable). Order: for each seed, COUNT, gated, scaled; then the analysis; then seeds 3 and 4 only if section 3 asks. Before the first run: the count tables (`data/count`, 5.2 GB) are copied to q-pc and their sha256 checked against the laptop's; the branch is synced; the test suite and `scripts/mutation_audit.py` have passed on the committed code; the q-pc tree is clean. A 200-step smoke run of each new arm on q-pc, run before the first real run, checks that the tables load, the speed, and that the three arms' first-step losses match (they start bit-identical); its validation loss is not read. Every run records `git_dirty`, and the analysis rejects a dirty run, a run whose pre-registration hash is not this file's, or a run at other rates.

Time, from the Stage 2E laptop measurements (COUNT 0.40 s per step, gated 0.46): about 1.4 to 1.6 h per run, nine runs, about 13 h. The five-seed rule would add up to six runs, about 9 h. q-pc's speed will be measured in the smoke run and replaces this estimate in the write-up. No money.

## 7. Conduct

A diverged run (non-finite loss or gradient) is rerun once from scratch; a second divergence is reported and gives no verdict. No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash. After any change to `src/`, the test suite and `scripts/mutation_audit.py` must pass. Planted-bug mutants guarding this stage: M173 to M179 (the scalar: dropped from the read, wrong start value, arm name, results field, config flag, refusal checks), M180 to M185 (the section 3 rules: flipped P19 and reading contrasts, a mistaken scale test, a flipped reading branch, the five-seed rule reduced or widened).

## 8. Limits stated in advance

1. **The scalar control is one global scalar.** The gated arm bundles three things the scalar lacks: a weight per order, a weight per count bin and source, and the absent vectors. A gated win over the scalar says the bundle helps, not which part. A per-order scalar (4 parameters) would narrow it and is not run.
2. **The gate and scalar rates are not tuned.** They train at 64e-3 with AdamW and no weight decay, eight times Stage 2E's gate rate. A gate may move faster than is good for it.
3. **One machine for the new runs, a different one for Stage 2F and 2E.** The three arms are paired on q-pc. Statements that compare with the Stage 2E gap of 0.0111 compare across machines, recipes and code versions.
4. **One model size and corpus.** 13.8M non-embedding parameters, FineWeb-Edu. Three seeds resolve about 0.01 nats and no finer.
5. **The scalar adds one parameter** to the non-embedding count (the gated arm adds 1,568), which is negligible against 13.8M.
6. **Cross-machine check is descriptive.** It has three pairs and is not a test.
7. **The class evaluation uses 4-token history counts only**, as in Stage 2E.

TL:DR. We rerun Stage 2E's gated COUNT read on the separate-rate recipe (Muon 8e-3, AdamW 64e-3) and add the scale-only control it called for: COUNT with one learned scalar on the whole read. Plain COUNT is rerun on q-pc so all three arms share a machine. Three seeds each; P19 asks whether the gate still beats COUNT by 0.01 on every seed; the paired contrasts against the scalar then read the gain as STRUCTURE, SCALE, UNRESOLVED or NOT_HOLDING. The expectation, from the 13 to 90 sized gates and the AdamW-rate effect, is that scale explains much of it. A thin or inconclusive result adds seeds 3 and 4 once. About 13 hours on q-pc, no money.
