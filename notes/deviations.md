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
