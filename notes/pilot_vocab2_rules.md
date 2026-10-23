# Harder vocab-task pilot: design and rules (written before any result)

The first pilot (`notes/pilot_vocab.md`) was too easy: every list size reached 100% accuracy, so final accuracy cannot separate wirings. This pilot tries two harder tasks, each based on what the mushroom body is known to do.

## Tasks

**Twins (pattern separation).** Cues are the most alike word pairs in the pool, each word with its own random answer. "Alike" is the cosine of positive-PMI context vectors (±2 tokens) over `data/fineweb/train.bin` (`src/clm/word_sim.py`). It uses no model, so it is the same for every arm. The top twins are "said / says", "million / billion", "north / south"; the 768th pair still has similarity 0.38, against 0.12 for a random pair. Basis: Kenyon-cell expansion and sparse connectivity support discrimination of similar inputs (Litwin-Kumar et al., Neuron 93:1153, 2017); associative capacity collapses when stored patterns resemble each other.
- 1536 cues = all 768 top twin pairs. 512 cues = the 256 most alike pairs only (harder per item).

**Sequential lists (continual learning).** k disjoint lists, learned one after another, steps per list fixed, then every list quizzed. Basis: the fly's two-layer olfactory associative circuit reduces catastrophic forgetting (Shen, Dasgupta & Navlakha, Neural Computation 2023, doi 10.1162/neco_a_01615). Metrics from Lopez-Paz & Ranzato (2017): `final_mean` (mean accuracy over all lists after the last list) and `backward_transfer` (mean change in each earlier list's accuracy from right after its own phase to the end; negative = forgetting).

Fixed for all settings: brain-only training, batch 16 sequences x 64 pairs, warmup 20, quiz every 50 steps, `task_seed` 0.

## Grid (scripts/pilot_vocab2.sh), on er-0 and then er-2

| Task | Pairs | Learning rate | Steps |
|---|---|---|---|
| twins | 1536, 512 | 3e-4 | 1500 |
| sequential, 4 lists | 384 per list | 3e-4, 1e-3 | 300 per list |
| sequential, 4 lists | 384 per list | 3e-4, 1e-3 | 600 per list |
| sequential, 8 lists | 192 per list | 1e-3 | 300 per list |

Laptop RTX 3050: 0.71–0.75 s/step measured in smoke runs. About 12,600 steps per model, about 5.5–6 h for both models with quizzes.

## Selection rules

Headline metric: `final_acc` for twins, `final_mean` for sequential.
1. A setting is **usable** if both er-0 and er-2 land between 0.30 and 0.90.
2. Among usable settings, pick the one whose er-0/er-2 mean is nearest 0.60. Ties go to the shorter run.
3. Report |er-0 − er-2| for every setting as a first estimate of seed-to-seed noise; it informs the SESOI for the 15-model run.
4. If no setting is usable, report every number to Nate; do not change the tasks without him.

Only er models are used. No real, shuf, real_randio or dense model is fine-tuned or read during this pilot.
