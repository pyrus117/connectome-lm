# Mutation audit — shared-brain design, 2026-10-21

Rerun with `python scripts/mutation_audit.py` (exit 0 = pass). Each mutant is planted in a fresh copy under `/tmp/clm-mut`, never in the real source; the copy's `src/` takes import precedence and `data/fineweb` is a read-only symlink to the real token files. A mutant counts as killed only when an assertion fails; a crash or type error does not count.

| ID | File | Mutation | Killing test | Evidence |
|---|---|---|---|---|
| M1 | graphs.py | shuffle returns the real post ends | test_shuffle_moves_most_edges | `assert (600 / 600) < 0.5` |
| M2 | graphs.py | permutation replaced by uniform random post ends | test_shuffle_keeps_every_neurons_in_and_out_degree | degree arrays differ |
| M3 | block.py | leak memory term dropped | test_brain_matches_the_dense_reference_layer_by_layer | Tensor-likes are not close |
| M4 | model.py | brain state reset at every layer | test_each_layer_receives_the_state_the_previous_layer_left | `layer 1 received no state` |
| M5 | block.py | per-step normalisation removed | test_brain_matches_the_dense_reference_layer_by_layer | Tensor-likes are not close |
| M6 | model.py | causal mask off | test_logits_at_a_position_ignore_later_tokens | Tensor-likes are not close |
| M7 | model.py | init from the global RNG | test_parameters_shared_by_two_arms_start_identical | `('shuf', 'tok.weight')` |
| M8 | data.py | data order unseeded | test_sampler_depends_only_on_seed_and_step | arrays differ (fresh-process check) |
| M9 | train.py | warmup off by one | test_lr_warms_up_then_decays_to_the_floor | `assert 0.0 == 0.0001` |
| M10 | train.py | resume restarts at step 0 | test_interrupted_and_resumed_run_ends_bit_identical | `tok.weight` |
| M11 | analyze.py | X_BETTER ignores sign agreement | test_verdict_rules | `'X_BETTER' == 'INCONCLUSIVE'` |
| M12 | analyze.py | NaN loss not flagged | test_check_run_names_each_defect | `assert (0 == 1)` |
| M13 | prep.py | split rule inverted | test_split_sends_about_one_percent_to_validation_and_is_stable | `assert 19788 <= 260` |
| M14 | graphs.py | dense brain sized from weights only | test_dense_arm_is_fully_connected_and_matches_the_regions_parameters | `assert 144 <= 24` |
| M15 | train.py | results.json records no wiring | test_completed_run_reports_what_the_analysis_checks | `assert [] == ['6243…']` |
| M16 | block.py | first-layer leak dropped | test_brain_matches_the_dense_reference_layer_by_layer | Tensor-likes are not close |
| C1 | block.py | rename local `drive` to `drv` in `step()` (behaviour-preserving) | full suite | survived, as required |
| C2 | graphs.py | `raise ImportError` on import | full suite | collection error, as required |

## Findings during the audit

- **First design (per-layer blocks), M8:** the sampler test compared two calls in one process; the per-process cache hid an unseeded order. Fixed by comparing against a fresh interpreter.
- **Shared-brain design, M4:** first "killed" by a `TypeError` (`torch.equal(None, …)`), not an assertion. The test now asserts the state is present first, and the runner no longer counts crashes as kills.
- **C1 first failed** because the copy lacked the token files used by `test_prepared_data.py`; the runner now symlinks `data/fineweb`.

## Not covered by a test

- A readout normalisation re-added in `Layer.forward` would cancel the last step's global norm and idle some edges at the start. The resulting gradients are only eps-small, not zero, so no exact test can detect it. The rule lives in the `SharedBrain` docstring and in `notes/measurements_2026-10-21.md` (design C).
