# Stage 2C result: writing to memory does not forget, consolidation fails, and the twin effect is real but small (2026-11-04)

Pre-registration `PREREG_stage2c.md` (decision rules in its section 6). Code `scripts/stage2c.py`, `src/clm/fastwrite.py`, `src/clm/fastwrite_arms.py`, driver `scripts/stage2c_driver.py`; results `results/stage2c_lists.json` and `results/stage2c_twins.json`. Run 2026-11-03, 22:18 to 22:48, on the laptop RTX 3050 (4 GB) with the GPU memory cap at 0.75. The models are the finished B0-M seed 0 and COUNT-M seed 0 from Stage 2A. Nothing was retrained. Three task seeds (0, 1, 2); every figure is a mean over them unless stated. One deviation (in `notes/deviations.md`): the twins stage first crashed at the random-pairs control with an `UnboundLocalError` (a variable named `weight` where the code needed `weights`). The fix was one word, and the twins stage was rerun from scratch. The lists results were untouched (commit `b58bad5`).

## Verdict: P7 is supported. Consolidation does NOT work as registered. P8 is supported by the rule, but the effect is small.

- **P7 (a write rule forgets nothing, while fine-tuning forgets a lot): supported.** The write arm has backward transfer 0.000. Both fine-tuned arms are at or below -0.3.
- **Consolidation (moving what the memory holds into the model's own weights): fails.** It forgets less than fine-tuning and barely hurts language, but after the memory is removed it recalls only 0.061 of pairs, against 0.760 for the write arm.
- **P8 (similar words help each other through the Kenyon hash): supported, small.** Only 3 in 100 twin cues reach the stored twin's answer.

Plain-word terms used below. A **pair** is a cue token and the answer it should produce. **Recall** is the share of cues that return the right answer. **Backward transfer** is the average change in accuracy on earlier lists, from just after each was learned to the end of all four. Zero means nothing was forgotten and negative means forgetting. **Language loss** is how badly the model predicts ordinary validation text (nats per token, lower is better), so it shows whether the model still speaks English.

## The write rule and its weight

The write rule adds one extra entry to the COUNT read. The cue token adds the answer's head row, multiplied by a weight w, to the embedding. There is no gradient and no weight change. The weight was chosen on a separate calibration list (task seed 100, 384 pairs) as the smallest w within 5% of the best recall. Language loss with the memory switched off is 3.362.

| w | recall | language loss, memory on |
|---|---|---|
| 1 | 0.406 | 3.418 |
| **4** | **0.763** | 3.524 |
| 16 | 0.643 | 3.562 |
| 64 | 0.570 | 3.571 |
| 256 | 0.557 | 3.573 |
| 1024 | 0.555 | 3.574 |

**Chosen w = 4.** Past 4, recall falls and language loss rises, so a bigger write is worse on both counts.

## Sequential lists

Four lists of 384 pairs, learned one after another. `final_mean` is the mean accuracy over the four lists at the end. Backward transfer is shown as the mean, then by seed.

| Arm | final_mean | backward transfer | by seed |
|---|---|---|---|
| gradient_b0 (fine-tune B0-M) | 0.753 | -0.330 | -0.319, -0.340, -0.329 |
| gradient_count (fine-tune COUNT-M) | 0.512 | -0.651 | -0.712, -0.579, -0.661 |
| write | 0.760 | 0.000 | 0, 0, 0 |
| consolidate (memory removed) | 0.061 | -0.251 | -0.266, -0.262, -0.223 |

Write `final_mean` by seed: 0.759, 0.743, 0.779.

Language loss on validation text at the end of the four lists. It started at 3.428 for B0 and 3.362 for COUNT.

| Arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| gradient_b0 | 10.385 | 10.292 | 10.676 |
| gradient_count | 12.865 | 11.505 | 13.502 |
| write, memory on | 4.125 | 4.253 | 4.147 |
| consolidate | 3.393 | 3.392 | 3.392 |

With the memory off, the write arm stays at exactly 3.362. With it on, the loss rises after every list (seed 0: 3.510, 3.656, 3.828, 4.125). Consolidation ends +0.03 above its start of 3.362.

Seed 0 accuracy matrices. Row is the point after that list was learned; column is the list being tested.

| Write | list 1 | list 2 | list 3 | list 4 |
|---|---|---|---|---|
| every row | 0.742 | 0.797 | 0.714 | 0.784 |

Each list keeps the accuracy it had when written, and later writes do not touch it.

| Consolidate (memory off) | list 1 | list 2 | list 3 | list 4 |
|---|---|---|---|---|
| after list 1 | 0.742 | 0 | 0 | 0 |
| after list 2 | 0.612 | 0.247 | 0 | 0 |
| after list 3 | 0.323 | 0.208 | 0.023 | 0 |
| after list 4 | 0.112 | 0.078 | 0.023 | 0.003 |

| gradient_count | list 1 | list 2 | list 3 | list 4 |
|---|---|---|---|---|
| after list 1 | 1.000 | | | |
| after list 2 | 0.576 | 1.000 | | |
| after list 3 | 0.198 | 0.729 | 1.000 | |
| after list 4 | 0.052 | 0.172 | 0.641 | 1.000 |

Fine-tuning learns each new list perfectly and loses the old ones. The consolidated model never learns the lists well to begin with (0.247, 0.023 and 0.003 on lists 2, 3 and 4 when they were new), and what it did learn fades.

## Decision rules (PREREG section 6)

**P7** is supported if the write arm's backward transfer is within 0.05 of zero and both gradient arms are at most -0.3.

| Condition | Measured | Met |
|---|---|---|
| write within +-0.05 of zero | 0.000 | yes |
| gradient_b0 at most -0.3 | -0.330 | yes, only just |
| gradient_count at most -0.3 | -0.651 | yes |

**P7 SUPPORTED.** The PREREG itself says the first condition is true by construction, since a write never changes what was already written. The informative part is the two gradient arms, and gradient_b0 clears its line by only 0.030.

**Usefulness** (reported, not part of P7). The write arm's `final_mean` of 0.760 beats gradient_count (0.512) and is level with gradient_b0 (0.753). Writes do cost language loss while the memory is on: +0.15 after one list, rising to +0.76 after four. Fine-tuning cost far more, with language loss above 10.

**Consolidation works only if all three hold.**

| Condition | Needed | Measured | Result |
|---|---|---|---|
| (a) `final_mean` at least the write arm's minus 0.05 | 0.710 | 0.061 | FAIL |
| (b) backward transfer at least gradient_count's plus 0.3 | -0.351 | -0.251 | pass |
| (c) language loss within 0.05 of its start | 3.412 | 3.392 (+0.03) | pass |

**CONSOLIDATION DOES NOT WORK as registered.** It fails (a), and by a wide margin.

**A limit stated in advance.** PREREG section 2 expected write recall to saturate near a fifth of pairs. That figure came from a smoke test on a 50M-token pilot checkpoint. On the finished COUNT-M the write recalls about three quarters of pairs (0.76), so the registered limit was too pessimistic for the final model.

## Twins

Twins are pairs of words that are very alike. 512 cues were built from the 256 most alike word pairs. An **overlay** is the rule that decides which memory slot a cue lands in. The exact overlay matches only identical cues. The token hash groups cues by a hash of their token. The Kenyon overlay is the selected hash on one-token histories, which can put alike words in the same slot. The random-codes version is the same hash with random codes, so any similarity is removed. Three measures:

- **Discrimination:** with all 512 cues written, how often each cue returns its own answer.
- **Transfer:** with only one of each pair written, how often the unwritten twin returns the stored twin's answer. This is the benefit of similarity.
- **Stored-twin recall:** how often the written twin still returns its own answer. This checks that sharing a slot does not damage ordinary recall.

| Overlay | discrimination | transfer | stored-twin recall | twin pairs sharing a bucket |
|---|---|---|---|---|
| exact | 0.790 | 0.000 | 0.776 | n/a |
| token hash | 0.790 | 0.000 | 0.776 | 0.0 |
| Kenyon | 0.793 | 0.030 | 0.778 | 0.062 |
| Kenyon, random codes | 0.790 | 0.000 | 0.776 | 0.0 |

The control for discrimination is recall on 512 random unrelated pairs with the exact overlay: 0.778 (by seed 0.831, 0.729, 0.773). Kenyon transfer by seed: 0.036, 0.031, 0.021. A twin asked for its own answer when only the other twin is written scored 0.000 for every overlay.

**P8** is supported if (a) exact discrimination is at least 0.9 times the control recall, and (b) Kenyon transfer beats both exact and token hash by at least 0.02 absolute, with Kenyon stored-twin recall at least 0.95 times exact's.

| Condition | Needed | Measured | Met |
|---|---|---|---|
| (a) exact discrimination | 0.700 | 0.790 | yes |
| (b) Kenyon transfer over exact and token hash | 0.020 | 0.030 (every seed above 0.02) | yes |
| (b) Kenyon stored-twin recall | 0.737 | 0.778 | yes |

**P8 SUPPORTED by the registered rule, but small.** Only 3% of twins reach the stored answer, because only 6.2% of twin pairs land in the same bucket. Random codes remove the effect entirely, so it comes from the similarity and not from the hashing itself. PREREG section 8 notes that the one-token hash was never calibrated for one-token histories.

## Things found while doing it

- The write arm cannot forget, and it is as accurate as the best fine-tuned arm (0.760 against 0.753). That is its whole case. Its price is language loss, which grows with every list written while the memory is on.
- gradient_b0 passes P7 by 0.030, with a seed spread of 0.021 (-0.319 to -0.340).
- Fine-tuning does not just forget the old lists. It wrecks ordinary language (loss above 10 from a start near 3.4), which the write arm and consolidation do not.
- The consolidated backbone with the memory also switched on scored a final row of only about 0.01 to 0.08 per list in every seed. This is unexplained and worth a look, since the memory alone holds the lists at about 0.76.
- The registered expectation that writes saturate near a fifth of pairs was wrong for the finished model. Pilot-checkpoint smoke tests understated it by a wide margin.
- The Kenyon twin gain is a small, real effect. It scales with how many twins share a bucket (6.2%), so it is capped by the hash, not by the write rule.

## Limits

- One model seed (B0-M seed 0 and COUNT-M seed 0). The three seeds are task seeds only.
- One write rule. A different rule could behave differently.
- Single-token words only.
- The twin hash was never tested at one-token histories (PREREG section 8).
- The consolidation failure is undiagnosed. It could be the recipe (300 steps at learning rate 3e-4, with an anchor) and not the idea.
- The memory-on score of the consolidated backbone is unexplained.

**TL:DR:** Writing to memory forgets nothing (backward transfer 0.000, against -0.330 and -0.651 for fine-tuning), so P7 is supported, and the Kenyon twin result passes P8 but helps only 3% of twins. Consolidation fails as registered (0.061 against 0.760), though the recipe may be at fault and not the idea.
