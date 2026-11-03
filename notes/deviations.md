# Deviations log (Stage 2)

Crash fixes and any departure from a committed pre-registration, each with its commit hash. Stage 1 recorded none.

## 2026-11-03, Stage 2B-0, commit 9503a45: speed-up of the Kenyon hash (no change of result)

`KenyonHash._drive` and `bucket_keys` (src/clm/kenyon.py) now gather and sum the per-token tables with one fused
`embedding_bag`, process 2,048 histories at a time and subtract the thresholds in place. The committed version took
708 s for one 100M-token order-3 bucket table (the sweep would have taken about 7 h and the 871M final run about 18 h).
Check: bucket keys from the committed version and the new one are identical on 100,000 histories each for four
settings (k = 2 default; k = 3 with 2 winners; k = 4 with in-degree 7; k = 3 without thresholds; real PPMI codes and
real calibrated thresholds), and the new code is 2.5 times faster (0.5-0.7 s per 100,000 histories against 1.2-2.0 s).
tests/test_kenyon.py::test_the_fused_gather_gives_the_drive_and_buckets_of_the_plain_sum keeps the equivalence.
The sweep had run 20 minutes of its first configuration when it was stopped and restarted on the new code; no result
had been written.

## 2026-11-03, Stage 2B-0, commit 74eb3f9: crash fix, memory-bounded bucket tables, and --resume

The sweep was killed by the kernel OOM killer at 15:31 (MemoryMax 12 GB) while building the order-5 bucket table of the
4-winner configuration, after seven configurations had finished (default, K1, K3, K7, K15, K31, winners2; their rows
are in results/stage2b0_sweep.json). Pooling every (bucket, continuation) pair in memory also would not fit at the
871M-token scale of the final run (the estimated pair count is several hundred million).
Fix: `bucket_table` (src/clm/kenyon.py) now routes each partition's entries to 32 spill files on disk (data/spill, not
the RAM-backed /tmp) by bucket-key range and pools one shard at a time; shards are ascending key ranges so the table
comes out in key order. Check: tables identical (all five arrays) to the previous implementation on a 12M-token
universe with 2 winners, and tests/test_kenyon.py compares 1, 2, 7 and 40 shards on skewed and hashed keys.
`scripts/stage2b0.py sweep --resume` keeps the rows already in the result file and runs only the missing
configurations; the seven finished rows were computed by the earlier, unsharded code, whose output is identical by the check above.

## 2026-11-03, Stage 2B-0, commit bc2e941: the final stage uses K31, not winners2 (feasibility added to the selection rule)

PREREG_stage2b0.md section 3 selects the Kenyon configuration with the lowest held-out rare loss. Applied to the whole
sweep it chose winners2 (5.09684). The final stage at 871M tokens then failed: the winners2 bucket tables have about one
bucket per distinct history (10.2M buckets at 2-token histories and 104.6M at 3-token histories before the order-5 table
was OOM-killed at 14 GB, 2026-11-03 18:07; the sweep showed 55M at 4-token histories for only 100M tokens). The network
arm needs the whole table in host memory at every training position, which this machine cannot hold for such a table, so a
winners2 result could never be used by Stage 2B.
Decision, taken before any validation score of the final stage existed (the killed run had scored only the exact model):
the rule is applied to the configurations whose tables fit, defined as at most 5M buckets at the longest history length (the
winners=1 family, 0.4M to 3.1M). K31 (31 inputs per cell, one winner, thresholds on) has the lowest rare loss among them
(5.10096 against K15 5.10099 and default 5.10116). results/stage2b0_selection.json records this; scripts/stage2b0.py
`final`, scripts/build_bucket_tables.py and scripts/stage2c.py read it in place of the sweep's own choice. The sweep file
is unchanged. PREREG_stage2b.md is amended to match before any Stage 2B run. The exact tables and codes of the final stage were
cached by the killed run and are reused.
