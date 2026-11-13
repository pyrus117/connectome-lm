# Stage 2M before the runs: post-hoc evidence and the smoke run (2026-11-14)

Pre-registration `PREREG_stage2m.md`, committed as `feb582e`. Neither measurement below is a test of a pre-registered claim.

## Post-hoc reading of Stage 2L seed 0 (read from q-pc checkpoints and logs, copied to `/tmp/2l`)

Mean absolute `loop_gain` per pass in the final checkpoint:

| Run | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Stage 2L LOOP (AdamW 64e-3) | 0.371 | 0.064 | 0.014 | 0.013 | 0.012 | 0.012 |
| Stage 2L LOOP+COUNT (AdamW 64e-3) | 0.318 | 0.055 | 0.016 | 0.013 | 0.012 | 0.012 |
| Stage 2H `both` (AdamW 8e-3, laptop) | 0.385 | 0.313 | 0.199 | 0.061 | 0.012 | 0.003 |

Pass 1 in both Stage 2L arms has a few channels at 14.8 and 18.5. Training loss by depth in Stage 2L LOOP: step 50 depth 1 6.2851, depth 6 6.3041; step 200 all six within 0.006; step 12206 all six 3.5068 to 3.5071. Stage 2L LOOP+COUNT is the same at step 12206 (3.274 at every depth). Stage 2H `both` at step 1000: 4.3481, 4.3085, 4.3067, 4.3071, 4.3070, 4.3070.

Stage 2H's final gains also fall to near zero for passes 5 and 6, so a pass switching itself off is not unique to Stage 2L. The difference is how early and how many passes: Stage 2L ended with passes 2 to 6 off, Stage 2H with passes 1 to 3 on.

## Smoke run (GAINLOW, seed 0, 400 steps, q-pc, `runs2/smoke_2m`, commit `feb582e`)

- Trains, 1.913 s per step.
- Optimizer groups in the checkpoint: Muon on 24 block matrices; AdamW `lr_mult` 8.0 on 6 parameters and `lr_mult` 1.0 on 1 parameter (`loop_gain`), weight decay 0.0 on both AdamW groups.
- Step 0 training loss by depth 9.0109 at every depth, the same as Stage 2L.
- Mean absolute gain per pass after 400 steps: 0.605, 0.669, 0.641, 0.621, 0.604, 0.590. No pass collapsed. This run's schedule (warmup 200, decay over 400 steps) is not the production schedule, so it is a pointer only.
- Training loss by depth at step 399: 4.2124, 4.2094, 4.2076, 4.2064, 4.2060, 4.2064. Validation loss not read.
