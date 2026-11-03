# Stage 2E pre-registration: the gated COUNT read

Dated 2026-11-04. Committed before any Stage 2E run. Design record: `notes/count_gate_design_2026-11-04.md`. Code: `src/clm/countread.py` (states), `src/clm/s2model.py` (`B0GPT._gate`), `src/clm/s2train.py` (`--count-gate`), `src/clm/rules2e.py` (the decision rules of section 5), `scripts/stage2e_driver.py`, `scripts/stage2e_analyze.py`, `scripts/bucket_eval.py`. Config `configs/stage2e.json`. Branch `count-gate`, merged into `stage2` before the first run.

## 1. Question

**P10.** Does the gated read beat plain COUNT? In COUNT every order's slot gets the same weight 1/J, exact and bucket slots look alike, and an order with no stored history adds nothing. The gated read gives each order a learned scalar for each (source, count bin), and a learned vector for an order with no stored history.

- **P10a (the absent vector).** Is the gain in the "not stored" class, without a cost in the stored classes?
- **P10b (the count bins).** Does the gate remove COUNT's loss in the ">=1000" class?

**P11.** Once the network can scale bucket slots down or up on their own, do Kenyon buckets help over the gated read? Stage 2B found they did not under the ungated read.

- **P11s (similarity).** If they help, is it the fly-style similarity and not pooling? Tested against gated random buckets.

Described, no decision: whether bucket gates end below exact gates of the same order (Nate's expectation: bucket weight near 0.1, exact 4-gram near 1).

A history is the last few tokens of the text so far. Cross-entropy is the model's average surprise per token, in nats.

## 2. Fixed design

- **Recipe for every arm.** COUNT-M with the Stage 2A read and recipe: Muon at lr 0.032 (Stage 2A's pilot, `runs2/pilot_count/choice.json`), weight decay 0.1, warmup 200 steps then linear decay to zero, clip 1.0, bf16, GPU memory cap 0.75, context 512, 200M tokens (12,207 steps of 32 x 512), leave-one-out reads while training, seeds 0, 1, 2. Same data order and name-keyed initialisation per seed. The gate parameters start at fixed constants, so every tensor two arms share starts identical.
- **Arms.**
  - (a) **COUNT**: the Stage 2A runs `runs2/count-M-0`, `-1`, `-2`. Not rerun.
  - (b) **COUNT-gated**: `runs2/count_gated-M-s`.
  - (c) **Kenyon-gated**: `runs2/count_kenyon_gated-M-s`, with the Stage 2B Kenyon K31 bucket tables `data/count_kenyon` (fallback rule exactly as `PREREG_stage2b.md` section 2).
  - (d) **Random-gated, conditional** (section 5): `runs2/count_random_gated-M-s`, with the Stage 2B random-bucket tables `data/count_randbucket`.
- **The gated read.** For each position and order (J = 4 orders, histories of 1 to 4 tokens), the reader labels the slot: no stored history, exact history, or bucket fallback. A stored slot also gets the bin of its history count after leave-one-out: 1-9, 10-99, 100-999, >=1000. Parameters: `count_gate`, shape (4, 8) (order x (source, bin)), starts at 1.0; `count_absent`, shape (4, 384), starts at 0. Each slot's scores are multiplied by its order's gate for its (source, bin). An order with no stored history (including window positions too early to hold that history length) adds that order's absent vector to the embedding. That is 1,568 parameters. At the start the gated model gives bit-identical output to plain COUNT (tested). The gates and absent vectors train with AdamW in the group of the other non-matrix parameters: same learning rate, weight decay 0.
  - Why this form: it is Nate's Avenue A (a weight per order and source, and a missing-history vector) with the weight also split by history count, because Stage 2A was worse than B0 at both ends of the count range. Count-bucketed interpolation weights are the standard n-gram smoothing fix (Bahl, Jelinek & Mercer 1983; Chen & Goodman 1999).
- **Metric.** As Stage 2A: mean cross-entropy on the first 2,097,152 tokens of `data/fineweb/val.bin` at the last step. Also recorded: seconds per step, peak VRAM, peak host RSS, mean lookup seconds, each run's final gate table and absent-vector norms.
- **Class evaluation.** `scripts/bucket_eval.py` re-evaluates each final checkpoint by the stored count of the exact 4-token history, in the same five classes as Stage 2B (not stored, 1-9, 10-99, 100-999, >=1000). Its overall loss must match `results.json` within 0.002.

## 3. Procedure

Runs go under `systemd-run --user` with the GPU memory cap 0.75. `embeddinggemma` is stopped for the runs and restarted after. One driver, `scripts/stage2e_driver.py`. Order: seed 0 gated, seed 0 Kenyon-gated, seed 1 gated, seed 1 Kenyon-gated, seed 2 gated, seed 2 Kenyon-gated, then analysis. Then the random-gated arm (seeds 0 to 2) only if section 5 asks for it. Then seeds 3 and 4 only if section 5 asks for them.

Before the first run: the branch is merged into `stage2`, the test suite and `scripts/mutation_audit.py` pass on the merged code, and the working tree is clean. Every run records `git_dirty`, and the analysis rejects a dirty run.

Smoke run, already done before this file was committed: 400 steps of Kenyon-gated, seed 0, 2026-11-04, laptop RTX 3050. Median 0.406 s per step. Median lookup 0.113 s (max 0.292 s), so the prefetch hides it. Peak VRAM 1,594 MiB, peak host RSS 9,501 MiB, largest gradient norm 2.25. Its validation loss was not looked at. Its gate table was looked at: by step 400 most exact gates were near 6 to 7 and the bucket gates near 2 to 4, under the smoke run's own 400-step schedule. That observation is the reason for limit 1 in section 8.

Time: about 1.4 h per run (Stage 2B: six runs in 8 h 30 min). The six main runs take about 8.5 h. The random arm adds about 4.2 h. The five-seed rule adds up to 8 runs, about 11 h. No money.

## 4. Reference numbers (not decision inputs)

- Stage 2A, COUNT minus B0 by class (mean of 3 seeds): not stored (47.3% of positions) +0.028; 1-9 (17.4%) -0.222; 10-99 (19.5%) -0.224; 100-999 (11.5%) -0.106; >=1000 (4.2%) +0.027. Overall -0.0798.
- Removing both penalties fully would gain about 0.028 x 0.473 + 0.027 x 0.042 = 0.014 nats overall. So a gate that only removes the penalties lands near the SESOI, and INCONCLUSIVE is a likely outcome for P10. A clearly larger gain would have to come from the stored classes.
- Stage 2B: Kenyon minus COUNT +0.0046, random minus COUNT +0.0040. In "not stored", +0.0085 and +0.0073.
- Stage 2B-0 offline gain of Kenyon buckets over the exact n-gram model: 0.0151 in "not stored", 0.0116 overall. If the gate lets the network use the buckets fully, a "not stored" gain near 0.015 is the most the offline numbers suggest. So P11's 0.01 bar is reachable but not the likely outcome.
- Real-table check (`notes/count_gate_design_2026-11-04.md`): with Kenyon buckets, 0.5% of 4-token positions have no slot, against 43.9% with exact COUNT. Bucket fills fall mostly in the pooled-count bins 100-999 (0.171 of positions) and >=1000 (0.216).

## 5. Decision rule

Paired by seed, d_s = loss(x) - loss(y), computed by `clm.analyze.verdict` (the Stage 1 rule). SESOI 0.01 nats.

- **X_BETTER:** mean d <= -0.01 and every d_s < 0.
- **X_WORSE:** mean d >= +0.01 and every d_s > 0.
- **NO_MEANINGFUL_DIFFERENCE:** |mean d| < 0.01 and every |d_s| < 0.02.
- **INCONCLUSIVE:** anything else.

Class gaps are means over seeds of the class loss, first arm minus second.

- **P10** (primary contrast, gated against COUNT): supported if and only if the verdict is X_BETTER.
- **P10a:** supported iff (gated - COUNT) in "not stored" <= -0.01, AND (gated - COUNT) < +0.01 in each of "1-9", "10-99" and "100-999".
- **P10b:** supported iff (gated - COUNT) in ">=1000" <= -0.01.
- **P11:** supported iff (Kenyon-gated - gated) in "not stored" <= -0.01. The overall verdict of Kenyon-gated against gated is also computed and reported. Why the class: a bucket fill only reaches positions whose exact history is not stored, so that is where a gain must show.
- **The random-gated arm** is run if and only if, on three seeds, the verdict of Kenyon-gated against gated is X_BETTER or P11 holds. Decided once, on three seeds, and not revisited.
- **P11s** (only if the random arm ran; otherwise reported as "not tested"): supported iff (Kenyon-gated - random-gated) in "not stored" <= -0.01. The verdicts random-gated against gated and Kenyon-gated against random-gated are also reported.
- Also reported by the same verdict rule: Kenyon-gated against COUNT.
- **Five-seed rule:** if gated against COUNT, or Kenyon-gated against gated, is INCONCLUSIVE on three seeds with |mean d| < 0.02, seeds 3 and 4 are added for every arm run so far, COUNT included (trained with Stage 2A's settings), and every rule is applied once more on five seeds.
- Described, no decision: each arm's mean gate table, raw and relative to the mean of the gates the read uses (a gate still exactly at its start value of 1.0 got no gradient and is left out); absent-vector norms; class losses next to B0's (`runs2/b0-M-s/bucket.json`); cost.

## 6. What a verdict does

- **P10 X_BETTER:** the gated read becomes the COUNT read of later stages. The next test is a scale-only control (one global scalar gate, no per-order or per-bin split, no absent vector) to tell structure from scale (limit 1). Whether to open gate 3 (a size-L check) with the gated read goes to Nate with the numbers.
- **P10 NO_MEANINGFUL_DIFFERENCE or X_WORSE:** written up. COUNT stays ungated.
- **P11 and P11s both supported:** multi-probe Kenyon reading (Avenue B, `notes/count_gate_design_2026-11-04.md`) becomes worth building.
- **P11 not supported:** the bucket line is closed at this model size.

Every outcome is written up with the per-class table and the gate tables.

## 7. Conduct

As Stage 2B. A diverged run (non-finite loss or gradient) is rerun once from scratch. A second divergence is reported and gives no verdict. No change to code, config or data after this file is committed except crash fixes, each logged in `notes/deviations.md` with its commit hash.

After any change to `src/`, the suite and `scripts/mutation_audit.py` must pass. Planted-bug mutants guarding this stage:

- M100-M102, the reader's states: an exact slot labelled as a bucket, a bucket slot labelled as exact, shifted count bins.
- M103-M105, the gate: absent vector added where a slot is filled, every slot read through the first gate, gates repeated in the wrong slot order.
- M106-M107: the gated model not starting equal to COUNT.
- M108: the arm not named gated.
- M109-M113, the section 5 rules: a flipped P10a sign, the no-harm clause dropped, P11 compared with COUNT instead of the gated read, the five-seed rule ignoring the 0.02 bound, the random arm triggered by the class rule alone.

## 8. Limits stated in advance

1. **Scale and structure are confounded.** The gates sit before the RMS normalisation, and the head rows start at zero, so the gates can raise or lower the memory as a whole against the token embedding. Plain COUNT can do that only by growing the head rows, which are also the output weights. A gain over COUNT can come from this scale freedom alone. The smoke run's gates (section 3) show the scale moving fast. Relative gate tables are reported to help read the shape, and section 6 names the control.
2. The gates train at COUNT's learning rate (0.032, at the top of Stage 2A's pilot grid) with AdamW and no weight decay, so a gate can move by about 0.032 per step. No separate rate was tuned for them.
3. For bucket slots the count bins say little: pooled counts are large, so most fills land in the top two bins. The source gate carries most of the bucket signal.
4. With Kenyon buckets almost every slot at orders 3 to 5 is filled (0.5% empty at 4 tokens), so in that arm the absent vector rarely acts at those orders.
5. The COUNT baseline comes from Stage 2A runs at an earlier commit. The ungated read is unchanged by this code (the gated model starts bit-identical to it), but the runs are not rerun. If the five-seed rule triggers, COUNT's seeds 3 and 4 are trained at the later commit.
6. One model size (13.8M non-embedding parameters), one corpus. Three seeds resolve about 0.01 nats and no finer.

TL:DR. We test whether a gated COUNT read, with a learned weight per order, source and count bin plus a learned vector for empty slots, lowers validation loss by at least 0.01 nats against plain COUNT over three paired seeds (P10), and then whether Kenyon buckets help once the network can turn them down (P11, against the gated read). Removing COUNT's two end-of-range penalties against B0 is worth about 0.014 nats overall, so INCONCLUSIVE is the likely outcome for P10, and P11's 0.01 bar is reachable but not the likely result. A random-bucket arm runs only if Kenyon-gated shows a gain over gated. P10 X_BETTER makes the gated read the COUNT read and calls for a scale-only control. No difference leaves COUNT ungated. If P11 and P11s both hold, multi-probe reading is worth building. If P11 fails, the bucket line closes at this size. Six main runs take about 8.5 h on the laptop, no money, everything fixed before any run.
