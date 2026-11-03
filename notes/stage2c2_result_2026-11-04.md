# Stage 2C-2 result: consolidation failed because the teacher drifted, not the head (2026-11-04)

Pre-registration `PREREG_stage2c2.md` (decision rules in its section 4; sha256 `79e7c51ad4f9727704865a25af8142751a70c4473f096276c4ab48febf46b462`, recorded in the results file). Code `src/clm/consolidate_fix.py`, driver `scripts/stage2c2.py`, tests `tests/test_consolidate_fix.py`, mutants M90 to M95; results `results/stage2c2_lists.json`. Run 2026-11-04 on the laptop RTX 3050 (4 GB) with the GPU memory cap at 0.75, under `systemd-run`. The model is the finished COUNT-M seed 0 from Stage 2A. Nothing was retrained. Task seeds 0, 1, 2; write weight 4.0. Nine runs, about 183 to 208 seconds each. No deviations and no crashes. Before the run the suite passed and the mutation audit passed (95 mutants, all killed by assertion, controls C1 and C2 as required).

## Verdict: Suspect 1 (teacher drift) is SUPPORTED. Suspect 2 (head drift) is NOT SUPPORTED, alone or with a clean teacher.

- **Teacher drift: supported, by a wide margin.** With an untouched teacher, consolidation reaches a `final_mean` of 0.765. The registered recipe reached 0.061.
- **Head drift: not supported.** Freezing the head did not help. With a clean teacher it cost about 0.013 in the mean. With the rolling teacher it changed nothing.
- **Consolidation works with a clean teacher.** Two variants pass all three registered conditions. The one with the rolling teacher fails.
- **The registered Stage 2C verdict stands.** The registered recipe specified the rolling teacher, so consolidation does not work as registered. This note shows why. The recipe was at fault, not the idea.

Plain-word terms used below. A **pair** is a cue token and the answer it should produce. **Recall** is the share of cues that return the right answer. **`final_mean`** is the mean recall over the four lists at the end. **Backward transfer** is the average change in accuracy on earlier lists, from just after each was learned to the end of all four. Zero means nothing was forgotten, negative means forgetting and positive means later learning helped earlier lists. **Language loss** is how badly the model predicts ordinary validation text (nats per token, lower is better). **Consolidation** means training the backbone, with the memory switched off, to match the backbone plus the memory on every pair written so far. The **teacher** is the model the student copies in each phase. The **student** is the backbone being trained, and it is quizzed with the memory off. The **head** is the output layer, `head.weight`. Its rows are also the table the memory's overlay writes into.

## The three variants

The registered consolidate arm gave each phase a teacher that was a copy of the student as it stood at the start of that phase. After phase 1 that copy has already been changed by training. Everything else here is the registered recipe: 300 steps per list, learning rate 3e-4, a language anchor, four lists of 384 pairs. Nothing was tuned.

| Variant | Teacher | Head |
|---|---|---|
| original_teacher | the untouched COUNT-M plus the overlay | trains |
| original_teacher_frozen_head | the untouched COUNT-M plus the overlay | frozen |
| rolling_teacher_frozen_head | the registered rolling teacher | frozen |

The comparison numbers come from `results/stage2c_lists.json` and were not rerun. Registered consolidate `final_mean` was 0.054, 0.054 and 0.074 by seed (mean 0.061). The write arm's `final_mean` mean was 0.760. The `gradient_count` backward transfer mean was -0.651.

## Results

`final_mean` by task seed 0, 1, 2, then the mean. Language loss is shown as the rise over the start of 3.362, by seed.

| Variant | final_mean by seed | mean | backward transfer | language loss rise |
|---|---|---|---|---|
| original_teacher | 0.768, 0.760, 0.768 | 0.765 | +0.026 | +0.034, +0.033, +0.033 |
| original_teacher_frozen_head | 0.734, 0.727, 0.794 | 0.752 | +0.029 | +0.035, +0.033, +0.035 |
| rolling_teacher_frozen_head | 0.052, 0.035, 0.057 | 0.048 | -0.277 | +0.033, +0.033, +0.034 |

The registered rule for consolidation (PREREG section 4, the three conditions of `PREREG_stage2c.md` section 6), scored for every variant.

| Condition | Needed | original_teacher | frozen head, clean teacher | rolling, frozen head |
|---|---|---|---|---|
| (a) `final_mean` at least 0.760 minus 0.05 | 0.710 | 0.765 yes | 0.752 yes | 0.048 FAIL |
| (b) backward transfer at least -0.651 plus 0.3 | -0.351 | +0.026 yes | +0.029 yes | -0.277 yes |
| (c) language loss within 0.05 of its start | +0.05 | all three seeds within | all three seeds within | all three seeds within |
| Result | | PASSES | PASSES | FAILS (a) |

Without the memory, the clean-teacher students recall about 0.73 to 0.81 per list at the end. That matches the write arm (0.76). Backward transfer is slightly positive, so later phases even help earlier lists a little. The language loss rise is about 0.033 in every variant.

## Decision rules (PREREG section 4)

Each suspect is supported only if the gain is at least 0.10 absolute, in the mean and in every seed.

| Suspect | Comparison | Per seed | Mean | Result |
|---|---|---|---|---|
| 1, teacher drift | original_teacher minus registered consolidate | +0.714, +0.706, +0.694 | +0.704 | SUPPORTED |
| 2, head drift, clean teacher | original_teacher_frozen_head minus original_teacher | -0.034, -0.033, +0.026 | -0.013 | NOT SUPPORTED |
| 2, head drift, alone | rolling_teacher_frozen_head minus registered consolidate | -0.002, -0.020, -0.018 | -0.013 | NOT SUPPORTED |

Freezing the head made no useful difference. Two of three seeds got worse with a clean teacher, and the third gained 0.026, a quarter of what the rule asks for.

## Why the rolling teacher fails

Each variant records `teacher_matrix`: the teacher's own recall of the list it must teach, with the memory on, at the start of phases 1 to 4. This is the most the student can reach, because it only copies the teacher.

| Teacher | seed | phase 1 | phase 2 | phase 3 | phase 4 |
|---|---|---|---|---|---|
| clean (original) | 0 | 0.742 | 0.797 | 0.714 | 0.784 |
| rolling, head frozen | 0 | 0.742 | 0.411 | 0.057 | 0.018 |
| rolling, head frozen | 1 | 0.776 | 0.273 | 0.052 | 0.018 |
| rolling, head frozen | 2 | 0.734 | 0.357 | 0.060 | 0.008 |

The clean teacher equals the write arm in every seed. The rolling teacher collapses, and it does so with the head frozen. In seed 0 its own read of list 1 falls 0.742, 0.583, 0.255, 0.112 across the four phases. So the damage to the overlay read comes from other weights moving, not from the head rows. By phase 3 the student is being taught by a teacher that no longer recalls the pairs it is meant to teach.

## Things found while doing it

- The failure in Stage 2C was the recipe. A replay teacher must be the untouched model plus the memory. A teacher that is itself being trained forgets the memory's overlay, and the student inherits that.
- Replaying the memory into the weights works here at the registered budget (300 steps per list, learning rate 3e-4), at a language loss cost of about 0.033 nats. This is complementary-learning-systems style replay, and it matched the write arm.
- It tests retention, not extra learning. The student cannot beat the memory's own recall of about 0.76, and it matched it.
- Backward transfer is slightly positive for both clean-teacher variants (+0.026 and +0.029), against -0.251 for the registered recipe and -0.651 for fine-tuning.
- Frozen head: no gain. The head is not the channel that breaks.
- **Odd, not diagnosed.** Read with the memory also switched on, the clean-teacher student scores only 0.32 to 0.50 per list at the end (seed 0, original_teacher: 0.505, 0.458, 0.401, 0.495). That is below its 0.73 to 0.81 with the memory off. The registered consolidate arm read with the memory on scored 0.003 to 0.081. So the unexplained memory-on score noted in Stage 2C is partly explained, since the registered backbone had collapsed. But a working consolidated backbone also reads worse with the memory on than off. A likely reason is that the backbone and the overlay both supply the answer. This was not tested.

## Limits

- One model seed (COUNT-M seed 0) and three task seeds.
- One write rule.
- Head freezing was tested only as a full freeze of `head.weight` under this recipe. Training only the attention and MLP weights, a self-consistency term with the memory on, and freezing the embeddings were not tried.
- The language loss rise of 0.033 is stable across variants. The anchor holds but costs a little.
- The clean-teacher recipe needs the original weights kept alongside the student.
- No compute cost beyond laptop time.
- The memory-on result above is untested.
- [TK: how the clean-teacher recipe behaves at a larger model size or with more than four lists]

**TL:DR:** The Stage 2C consolidation failure was teacher drift. With an untouched teacher the student recalls 0.765 of pairs against 0.061 for the registered recipe (supported at +0.704, every seed above +0.69). Freezing the head did not help (-0.013 in the mean), and the registered 2C verdict still stands because it specified the rolling teacher.
