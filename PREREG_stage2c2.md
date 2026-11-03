# Stage 2C-2 pre-registration: two suspected faults in consolidation

Dated 2026-11-04. Written before any Stage 2C-2 run; the driver records this file's sha256 in its results. Follows `PREREG_stage2c.md` (result in `notes/stage2c_result_2026-11-04.md`). Model: the finished COUNT-M seed 0 run (`runs2/count-M-0`, Stage 2A), never retrained. Code: `src/clm/consolidate_fix.py`, `scripts/stage2c2.py` (branch `stage2`). The registered `src/clm/fastwrite_arms.py` is unchanged. GPU time: under an hour. All four arms of one task seed took 337 to 358 seconds in Stage 2C, so each variant run is under 6 minutes and the nine runs are under 54 minutes at the worst.

This is exploratory follow-up to a registered failure, written before any of its runs. It does not change the Stage 2C verdict.

## 1. Questions

Stage 2C tested **consolidation**: train the backbone with the memory off so that it matches backbone plus memory on every pair written so far. It failed its registered rule. The consolidate arm's `final_mean` was 0.061 against the write arm's 0.760, and its backward transfer was -0.251. Its language loss rose only 0.03 nats from the start of 3.362, so that condition and the backward-transfer condition passed. That verdict stands.

The failure was left undiagnosed. The cause could be the theory (complementary learning systems) or the recipe. We suspect the recipe, in two places. This stage tests each.

- **Suspect 1, teacher drift.** In `src/clm/fastwrite_arms.py` line 111, the teacher of every phase is a copy of the student as it stands at the start of that phase. From phase 2 on, that backbone has already been changed by earlier consolidation. If a changed backbone reads the memory's overlay worse, the student is being taught by a worse teacher.
- **Suspect 2, head drift.** The head (`model.head.weight`, an untied unembedding in `src/clm/s2model.py`) does two jobs. It is the output layer. Its rows are also the table the COUNT read and the overlay add into the token embedding (the `count` argument of the forward pass). The overlay writes "the answer's head row times the write weight w" into the embedding. If consolidation moves the head rows, then what the rest of the network learned to read from that channel changes meaning.

## 2. Evidence for the suspects

All from seed 0, consolidate arm, `results/stage2c_lists.json`.

- After phase 1 the student recalls list 1 at 0.742, the same as the write arm. After phase 2 it recalls lists 1 and 2 at 0.612 and 0.247. Phase 1 is the only phase where the teacher is the untouched model. This fits suspect 1.
- The consolidated backbone, read with the memory switched on, scores 0.604 on list 1 after phase 1. The original model with memory scores 0.742. After phase 2 it scores 0.292 and 0.219. It ends at 0.07, 0.039, 0.026 and 0.008. So training the backbone damages its ability to read the overlay. This fits both suspects.

This is one seed and a pattern, not a test. The runs below are the test.

## 3. Design

Same lists as Stage 2C (4 lists of 384 pairs, task seeds 0, 1, 2), the same write weight (4.0, taken from `results/stage2c_lists.json`) and the same settings as the registered consolidate arm. AdamW, lr 3e-4, betas 0.9 and 0.95, no weight decay, warmup 20 steps, 300 steps per list, batches of 16 sequences of 64 pairs, gradient clip 1.0. The KL anchor to the original model on ordinary validation text has weight 1.0 and 4 windows per step, on the same validation windows. Nothing is tuned. Every figure is a mean over the three task seeds unless stated.

The student and the quiz have the memory off in every variant. Three variants:

1. **original_teacher.** The teacher of every phase is the untouched original COUNT-M plus the overlay, with every list written so far. The head trains. Tests suspect 1 alone.
2. **original_teacher_frozen_head.** As variant 1, and `head.weight` is left out of training. Everything else trains. Tests suspect 2 on top of the clean teacher.
3. **rolling_teacher_frozen_head.** The registered rolling teacher, with `head.weight` frozen. Tests suspect 2 alone.

The registered consolidate and write arms are not rerun. They are the comparison, read from `results/stage2c_lists.json`: consolidate `final_mean` 0.061 and `backward_transfer` -0.251; write `final_mean` 0.760; `gradient_count` `backward_transfer` -0.651.

**Diagnostic.** Every variant records `teacher_matrix`: the teacher's own recall, with the memory on, of every list at the start of each phase. This is the ceiling the student can reach. It shows directly whether the rolling teacher gets worse.

## 4. Decision rules

All are fixed here.

- **Consolidation works** is scored for each variant with the three conditions of `PREREG_stage2c.md` section 6:
  - (a) `final_mean` at least the write arm's minus 0.05, so at least 0.710;
  - (b) `backward_transfer` at least `gradient_count`'s plus 0.3, so at least -0.351;
  - (c) language loss after the last phase within 0.05 nats of its starting loss.

  Passes and numbers are reported for all three variants, whether or not they pass.
- **Suspect 1 (teacher drift) is supported** iff `original_teacher`'s `final_mean` exceeds the registered consolidate arm's (0.061) by at least 0.10 absolute, in the mean and in each of the three task seeds.
- **Suspect 2 with a clean teacher (head drift) is supported** iff `original_teacher_frozen_head`'s `final_mean` exceeds `original_teacher`'s by at least 0.10 absolute, in the mean and in each seed.
- **Suspect 2 alone is supported** iff `rolling_teacher_frozen_head`'s `final_mean` exceeds the registered consolidate arm's by at least 0.10 absolute, in the mean and in each seed.
- Anything else is reported as not supported, with the numbers. A variant can fail the "consolidation works" rule and still support a suspect.
- If no variant passes the consolidation rule, that is reported as it is. Further variants need a new pre-registration.

## 5. Conduct

Runs go under `systemd-run --user` on the laptop RTX 3050 (4 GB) with the GPU memory cap 0.75. No money is spent. Three variants times three task seeds, each costing about as much as one registered consolidate run. The driver writes `results/stage2c2_lists.json`. Only the finished COUNT-M model is used, so no validation loss of a primary arm (`real`, `shuf`) is read.

Tests are in `tests/test_consolidate_fix.py`. One test shows that the variant with the rolling teacher and a trainable head reproduces the registered `run_consolidate` bit for bit: the same accuracy matrices, language losses and final weights on a small model. So the only differences between arms are the two named switches. Planted-bug mutants M90 to M95 in `scripts/mutation_audit.py` cover the new code. The suite and `scripts/mutation_audit.py` must pass before the run.

No change to code, config or data after this file is written except crash fixes, each logged in `notes/deviations.md`.

## 6. Limits stated in advance

Not tested here:

- training only the attention and MLP matrices;
- a self-consistency term with the overlay on;
- freezing the embeddings;
- other write rules;
- other model seeds.

Also:

- The teacher is still the backbone plus the memory, so the student can at best match the memory's own recall. This tests retention, not extra learning.
- A null result here does not clear complementary learning systems theory. It only rules out these two faults.
- There is one model seed (0) and three task seeds. The model has 13.8M parameters.

TL:DR. Stage 2C's consolidation failed (final mean 0.061 against 0.760 for plain writes), and that stays on the record. We suspect two recipe faults: the teacher in later phases is a backbone already damaged by earlier consolidation, and the head rows that carry the memory's overlay get moved by training. Three variants on the finished COUNT-M model, with a clean teacher and a frozen head in each combination, test them over three task seeds. A suspect is supported only if it lifts the final mean by at least 0.10 in the mean and in every seed. The settings, thresholds and comparison numbers are fixed before any run, nothing is tuned, and no money is spent.
