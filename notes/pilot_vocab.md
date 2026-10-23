# Vocabulary-list pilot, 2026-10-23

Brain-only fine-tune (everything outside `model.brain` frozen) of `runs/er-0` on random single-token word pairs. Plan: `docs/superpowers/plans/2026-10-23-vocab-list-pilot.md`. Only `er-0` was used; no `real`, `shuf`, `real_randio` or `dense` model was fine-tuned or read.

Hardware: one RTX 3090 on Vast (instance 57328956, offer 54025930, about US$0.22/h), torch 2.14.1+cu130, numpy 2.5.4. Speed 0.215–0.231 s/step (batch 16 x 129 tokens). Laptop RTX 3050 measured 0.72 s/step for the same step (diagnostic run, 128 pairs).

## Results (1,500 steps, quiz every 50 steps)

| Setting | Chance | Steps to 50% | Steps to 90% | Final |
|---|---|---|---|---|
| 512 pairs, lr 3e-4 | 0.0020 | 200 | 250 | 1.00 |
| 512 pairs, lr 1e-3 | 0.0020 | 100 | 150 | 1.00 |
| 512 pairs, lr 3e-3 | 0.0020 | 50 | 100 | 1.00 |
| 128 pairs, lr 3e-4 | 0.0078 | 100 | 150 | 1.00 |
| 1536 pairs, lr 3e-4 | 0.00065 | 400 | 600 | 1.00 |

1536 pairs, lr 3e-4, quiz accuracy by step: 150 0.02, 200 0.06, 250 0.14, 300 0.25, 350 0.38, 400 0.53, 450 0.65, 500 0.77, 550 0.85, 600 0.92, 650 0.96, 700 0.98.

## Choices by the pre-written rules

- Learning rate: every setting tied at 1.00 final accuracy, so the tie-break picked the smallest, 3e-4.
- List size: the rule wants the largest list whose final accuracy lands between 0.30 and 0.90. None did; all reached 1.00. Per the plan, this goes to Nate, not a silent design change.
- 1536 pairs is the largest list the 3,394-word pool allows (cues and answers disjoint), so final accuracy cannot be made harder by list size alone.

## What it means for the main run

The brain stores 1,536 random pairs perfectly. Final accuracy is at its ceiling, so it cannot separate wirings. Learning speed still varies widely (steps to 90% go 150, 250, 600 as the list grows), so a speed measure can.

## Deviation from the plan

`tests/test_finetune.py`: the plan's tiny-model memorisation test failed (8 pairs stayed at chance after 1,000 steps; scaled projections reached only 0.375). The untrained toy model's frozen projections cannot carry the brain's output. It was replaced by the same 50% bar on the real `er-0` checkpoint (128 pairs, 100 steps, GPU only). Commit `aec1f5d`.
