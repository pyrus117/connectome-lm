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

## 2026-11-03, Stage 2C, crash fix: twins stage used an unbound name

The `lists` stage of scripts/stage2c.py finished (results/stage2c_lists.json, 22:35). The `twins` stage then crashed at the
random-pairs control with `UnboundLocalError: weight` (line 132): the variable is called `weights` there (it holds the single
chosen write weight read from stage2c_lists.json). One-word fix, `Overlay(vocab, weights)`. No result had been produced by
the twins stage, and the lists results are untouched. The twins stage is rerun from scratch; the driver skips the finished lists.

## 2026-11-03, Stage 2B: run at Nate's request although gate 2 failed

PREREG_stage2b.md says Stage 2B runs only if gate 2 passed. Gate 2 failed (0.0151 and 0.0130 against 0.02, see
notes/stage2b0_result_2026-11-03.md) and the plan's rule was to write up instead. After the Stage 2D report Nate decided
(2026-11-03, evening) to run Stage 2B anyway. This is a decision to run, not a change to the registered design: the arms,
settings, metric and decision rules in PREREG_stage2b.md are unchanged, and the amendment note at its top says so. The result
is reported as "run despite a failed gate".

## 2026-11-04, Stage 2E: forced machine restart, one run resumed from its checkpoint

The laptop was restarted at 18:28 NZDT while `runs2/count_gated-M-2` was at about step 9,257 of 12,207. The driver was
restarted with the same command; the run resumed from its step-9000 checkpoint (data order depends only on seed and step, so the
batches repeat). No code, config or data changed and the working tree stayed clean. About 257 steps were redone. The
resumed run is not bit-identical to an uninterrupted one (bf16 on GPU), which is the usual run-to-run noise. The 257 log
lines written after the checkpoint are kept in `runs2/count_gated-M-2/log.pre-restart-tail.jsonl`; the NUL-filled last line
from the crash was dropped and `log.jsonl` was cut to steps 0-8999 so no step is logged twice (`RESTART.txt` in that folder).
`embeddinggemma.service` had restarted itself on boot and was stopped again for the remaining runs.

## 2026-11-05, Stage 2F, crash fix: analysis read a key Stage 2A's runs do not have

All Stage 2F runs finished (05:43 NZDT). `scripts/stage2f_analyze.py` then crashed with `KeyError: 'adam_mult'` on the check that
Stage 2A's COUNT runs used one shared rate: their `results.json` predates the `adam_mult` setting. Fix (commit 45c17a5): read it
with `.get("adam_mult", 1.0)`. No run, config, pre-registered rule or result was touched; the analysis was rerun by hand and wrote
`results/stage2f.json`. The driver stopped at the crash, so it did not print its final VERDICT line.

## 2026-11-08: Stage 2H moves to the Q-Youth work laptop (q-pc, Windows)

The six unfinished Stage 2H runs (arm `both` seeds 1 and 2, `reference` seeds 1 and 2, ablations `gain` seed 0 and `all` seed 0) run on q-pc
instead of the Linux laptop. The finished runs (`both` seed 0, and `reference` seed 0 reused from Stage 2D) ran on the Linux laptop.

| | Linux laptop (seed 0 runs) | q-pc (the rest) |
|---|---|---|
| GPU | RTX 3050 Laptop, 4 GB | RTX 4060 Laptop, 8 GB |
| OS and Python | Linux, 3.14.9 | Windows, 3.12.10 |
| torch, numpy | 2.15.0 (CUDA 13.4), 2.5.4 | 2.15.0+cu134, 2.5.4 |

Floating-point results are not bit-identical across the two machines, so the seed 0 pair and the seed 1 and 2 pairs are compared across machines.
The registered effect size is 0.01 nats and the Stage 2D seed SD is about 0.003, so the expected effect is larger than the machine difference, but
this was not measured. `train_seconds` is not comparable across the machines.

Code changes (commit on branch `stage2-windows`; no change to the model, loss, optimiser, config, data or any registered rule):

1. `src/clm/s2train.py`: the `resource` module does not exist on Windows. `host_rss_peak_mib` reads the peak working set through `psutil` there.
   The Linux code path is the same expression as before.
2. `scripts/stage2_0_driver.py`, `scripts/stage2h_driver.py`: venv python path `.venv/Scripts/python.exe` on Windows, and a child environment that
   keeps `SYSTEMROOT`, `PATH`, `TEMP` and `TMP` there. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` stays Linux-only.

Git for Windows converts line endings by default (`core.autocrlf=true`). That changed the bytes of `configs/stage2h.json` and `PREREG_stage2h.md`
and so their recorded SHA-256 values. This clone sets `core.autocrlf=false`; both hashes equal the values recorded in the seed 0 `results.json` files.

Test suite on q-pc: 3 failures, none in code Stage 2H uses. `test_countread::test_leave_one_out_removes_a_history_that_was_seen_only_once`
fails on a Windows file lock when a temporary folder holding a memory-mapped file is deleted. `test_prepared_data` (2 tests) needs
`train_ids.txt` and `val_ids.txt`, which were not copied. `scripts/mutation_audit.py` was not rerun; the change touches no model logic.

## 2026-11-09: Stage 2H analysis loader rejected the reused reference seed 0

All six runs finished on q-pc. `scripts/stage2h_analyze.py` then stopped with
`loop-M-0 trained with {... 'adam_mult': None ...}, expected {... 'adam_mult': 1.0 ...}`.
Stage 2D's seed 0 predates the `adam_mult` setting (added in Stage 2F, commit bca00eb), so its `results.json` does not record it.
Stage 2D's code (`param_groups(model, optimizer, lr, wd)`) used one shared rate for every parameter, which is `adam_mult` 1.0.
The loader already defaults the missing `loop_gain` and `loop_loss` of that run. The same default is now applied to `adam_mult` (1.0).

The change is in the loader only. No decision rule, threshold, run, or result changed, and the analysis had not produced any number before this change.
The same one-line fix is needed in the Linux repo, where the unmodified loader would reject Stage 2D's seed 0 in the same way.

## 2026-11-09: Windows port and analysis loader fix merged into `stage2`; oracle script reads `loop_gain`

All training and inference now run on q-pc, so the two q-pc commits above (applied here with `git am`) are on `stage2`. The Linux code path of each is the same expression as before. The q-pc clone is reset to this branch.
`scripts/stage2d_analyze.py` now builds the model with `gain=tcfg.loop_gain` and takes an optional output file name as a second argument, so the Stage 2H repeat of the oracle analysis (PREREG_stage2h.md section 7) does not overwrite `results/stage2d.json`. For Stage 2D's own run both defaults give the old behaviour. Test suite 100% and mutation audit 147/147 killed, both required controls as expected, after the merge.

## 2026-11-12: Stage 2L `loop_gain_all_sep-M-0` resumed from its step-11,000 checkpoint after a forced restart of q-pc

q-pc restarted at 17:47 NZ on 2026-11-11 (a Windows update), while `runs2/loop_gain_all_sep-M-0` was between steps 11,000 and 11,238 of 12,207. The driver process died with the restart. The tree was clean at commit f96d054 (`git status` empty), and the checkpoint (`ckpt.pt`, 17:39) was intact. The driver was started again at 12:10 on 2026-11-12 with the same launch script, and the run resumed from step 11,000.

Effect: the steps from 11,000 to about 11,238 ran twice, and `log.jsonl` holds both passes of them. Resuming is bit-identical to an unbroken run in the tests (`tests/test_s2loop.py::test_interrupted_and_resumed_loop_run_ends_bit_identical`, and the Stage 2H resume tests), but a CUDA run on a different driver state is not guaranteed to be; on the Stage 2E precedent the effect is at the level of ordinary run-to-run noise, well below the 0.01 nat bar. No code, config, data or rule changed. The Stage 2L driver also restarted its log (`runs2/stage2l_driver.log` was overwritten); `runs2/stage2l_choice.json` is unchanged (read: count).

## 2026-11-14: Stage 2M seed 0 GAINLOW validation losses seen before the futility check (conduct slip, no code change)

PREREG_stage2m.md section 3 says the driver's futility check is the only validation read of a registered arm before the analysis. At 20:24 NZ on 2026-11-14 I printed the last line of `runs2/loop_count_gainlow-M-0/log.jsonl` to report status. That line carries `val_loss_by_depth`, so I saw seed 0 GAINLOW's six validation losses (all about 3.2189, equal to within 0.00002) together with its training losses by depth (3.26480, 3.26458, 3.26472, 3.26475, 3.26483, 3.26472). The training losses alone, which the pre-registration allows, show the same flat profile. Nothing was changed because of it: the design, code, config, rules and driver are as committed (`feb582e`), the run and the driver were not touched, and the futility check and the analysis will run mechanically as registered. Status reads from now on take only `step`, `loss`, `train_loss_by_depth` and `seconds`.
