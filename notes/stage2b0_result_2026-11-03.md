# Stage 2B-0 result: the Kenyon backoff helps offline, by less than the gate asked (2026-11-03)

Pre-registration `PREREG_stage2b0.md` (commit `95f0517`, branch `stage2b0`). Code `src/clm/kenyon.py`, `src/clm/kenyon_lm.py`, `scripts/stage2b0.py`; results `results/stage2b0_sweep.json`, `results/stage2b0_selection.json`, `results/stage2b0_final.json`. CPU only, 4-5 threads at low priority beside the GPU runs. Deviations (all in `notes/deviations.md`): a speed-up of the hash with identical keys, an out-of-memory crash fix in the bucket-table builder, and K31 selected in place of winners2 because winners2's tables cannot be built here.

## Verdict: gate 2 FAILS. Stage 2B (the network arm) is replaced by this write-up.

Final run (871M-token universe, held-out slice H not counted, mixing weights fit on H, scored on the 4,096 x 512 validation windows). The Kenyon setting is K31 (PPMI codes, 31 inputs per cell, 2 x 2,048 cells, one winner each, thresholds on). Validation loss (nats/token) by the stored count of the exact 4-token history:

| Class | share | exact n-gram | + Kenyon | + random buckets | + Kenyon, random codes |
|---|---|---|---|---|---|
| not stored | 47.3% | 4.8717 | 4.8565 | 4.8716 | 4.8687 |
| 1-9 | 17.4% | 4.5978 | 4.5847 | 4.5977 | 4.5952 |
| 10-99 | 19.5% | 4.5505 | 4.5410 | 4.5504 | 4.5481 |
| 100-999 | 11.5% | 4.6806 | 4.6774 | 4.6805 | 4.6796 |
| >=1000 | 4.2% | 4.9787 | 4.9775 | 4.9789 | 4.9784 |
| all | | 4.7474 | 4.7358 | 4.7474 | 4.7450 |

Gain over the exact model: Kenyon 0.0151 / 0.0130 / 0.0095 / 0.0031 / 0.0012 (overall 0.0116); random buckets 0.0001 everywhere; Kenyon with random codes 0.0030 / 0.0026 / 0.0024 / 0.0009 / 0.0003 (overall 0.0024).

Gate: (Kenyon gain - random-bucket gain) must be at least 0.02 in both "not stored" and "1-9". Measured **0.0151 and 0.0130**: FAIL. The effect is real (random buckets gain nothing, and similarity carries about four fifths of it: random codes keep a fifth), and it falls where the idea says (largest at histories with no exact counts, shrinking as counts grow), but it is three quarters of the size the gate required and, on these n-gram models, 0.0116 nats over all positions, about the SESOI of 0.01. The network would have to gain about that much for 2B to resolve it with three seeds; that is why the plan's gate sat at 0.02.

What the verdict does (PREREG section 6): 2B is not run; the work goes on to Stage 2C with COUNT alone (and 2D, which never depended on this). Everything needed for 2B is built and registered if Nate wants it run anyway: `PREREG_stage2b.md`, the bucket fallback in `CountReader` (tested against a brute-force reference, mutants M75-M76), the builders, the driver and analysis scripts, and the random-bucket tables (`data/count_randbucket`, built 2026-11-03, 4.19M buckets per order). The Kenyon tables would take about 3 h to build (`scripts/build_bucket_tables.py --kind kenyon`) and the six 200M-token runs about 9 GPU-hours.

## The sweep (100M-token universe, held-out slice only; no validation read)

"Rare loss" is the mixture's cross-entropy on H over positions whose 4-token history is not stored or seen 1-9 times (exact model alone: 5.1216). Lower is better.

| Configuration | rare loss | gain | buckets at 4-token histories |
|---|---|---|---|
| default (dense, one winner) | 5.10116 | 0.0205 | 3.1M |
| K1 | 5.10616 | 0.0155 | 0.4M |
| K3 | 5.10275 | 0.0189 | 2.1M |
| K7 | 5.10119 | 0.0204 | 2.6M |
| K15 | 5.10099 | 0.0206 | 2.8M |
| **K31** | **5.10096** | 0.0207 | 2.9M |
| winners2 | 5.09684 | 0.0248 | 55.5M |
| winners4 | 5.10119 | 0.0204 | 63.1M |
| thresholds off | 5.10242 | 0.0192 | 2.8M |
| random codes | 5.11737 | 0.0043 | 3.9M |
| random buckets | 5.12089 | 0.0007 | 4.2M |

The registered rule picked winners2. Its tables hold about one bucket per distinct history; at 871M tokens the order-4 table alone had 104.6M buckets before the order-5 table was killed by the out-of-memory killer. The network arm needs the whole table, so feasibility (at most 5M buckets) was added to the rule and K31 chosen (`results/stage2b0_selection.json`, `notes/deviations.md`).

Readings of P5 and P6 from the sweep, by the rules of PREREG section 5:
- **P5 (inputs per cell: improves and saturates; dense at least as good as K = 7): supported.** K31 is 0.0052 better than K1 (needed 0.005, barely); dense is 0.00003 better than K7 (it could have been up to 0.002 worse). Quality rises from K1 to K7 and is flat from K7 to dense.
- **P6 (thresholds keep cell use even and quality high): not supported.** Without thresholds the entropy fraction of cell use falls from 0.996 to 0.916 (needed a fall of 0.02: met), but the rare loss is only 0.0013 worse (needed 0.002: not met). Homeostasis balances the cells and buys almost nothing here.
- **P3 (Kenyon beats random buckets in unseen and rare histories): effect present, gate not met** (above).

## Things found while doing it

- A 2-winner hash is close to an exact hash (about one bucket per history) and gives the best offline gain but no table you can hold; one winner pools on the order of a hundred distinct histories per bucket (about 400M histories into 3.6M buckets) and fits.
- The pooled-bucket build needed a disk-sharded pooling step to run at 871M tokens; speed on this machine is dominated by memory traffic (a 16 KB gather per token per history position): 0.5-0.7 s per 100,000 histories after fusing the gather.
- Exact n-gram validation loss on this universe (871M tokens, c_min 1, 1, 2, 2): 4.7474 (Stage 2A's table, built from all 881M tokens: 4.7455).
