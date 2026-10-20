# Mutation audit, 2026-10-21

Each mutant was planted in a fresh copy of the repo under `/tmp/clm-mut` (runner: `/tmp/clm_mutation_runner.py`, not committed), never in the real source. The copy's `src/` was confirmed to take import precedence over the editable install. Command per row: `pytest -x -q <test>` in the copy.

| ID | File | Mutation | Test | Result | Evidence |
|---|---|---|---|---|---|
| M1 | graphs.py | shuffle returns the real post ends | test_shuffle_moves_most_edges | killed | `assert (600 / 600) < 0.5` |
| M2 | graphs.py | permutation replaced by uniform random post ends | test_shuffle_keeps_every_neurons_in_and_out_degree | killed | AssertionError (degree arrays differ) |
| M3 | block.py | leak memory term dropped | test_block_matches_the_dense_reference | killed | Tensor-likes are not close |
| M4 | block.py | one sparse pass fewer | test_first_step_gradient_reaches_exactly_the_edges_edge_usage_counts | killed | AssertionError (gradient sets differ) |
| M5 | block.py | first-step normalisation removed | test_block_matches_the_dense_reference | killed | Tensor-likes are not close |
| M5b | block.py | readout RMS norm added back | test_block_matches_the_dense_reference | killed | Tensor-likes are not close |
| M6 | model.py | causal mask off | test_logits_at_a_position_ignore_later_tokens | killed | Tensor-likes are not close |
| M7 | model.py | init from the global RNG | test_parameters_shared_by_two_arms_start_identical | killed | `('shuf', 'tok.weight')` |
| M8 | data.py | data order unseeded | test_sampler_depends_only_on_seed_and_step | **survived, test fixed, then killed** | see below |
| M9 | train.py | warmup off by one | test_lr_warms_up_then_decays_to_the_floor | killed | `assert 0.0 == 0.0001` |
| M10 | train.py | resume restarts at step 0 | test_interrupted_and_resumed_run_ends_bit_identical | killed | AssertionError `tok.weight` |
| M11 | analyze.py | X_BETTER ignores sign agreement | test_verdict_rules | killed | `'X_BETTER' == 'INCONCLUSIVE'` |
| M12 | analyze.py | NaN loss not flagged | test_check_run_names_each_defect | killed | `assert (0 == 1)` |
| M13 | prep.py | split rule inverted | test_split_sends_about_one_percent_to_validation_and_is_stable | killed | `assert 19788 <= 260` |
| C1 | block.py | rename local `drive` to `drv` in `state()` (behaviour-preserving) | full suite | survived, as required | all passed |
| C2 | graphs.py | `raise ImportError` on import | full suite | collection error, as required | stopped at collection |

## M8: the one survivor

The test called `batch_starts` twice in one process and compared the results. The permutation is cached per process (`lru_cache` on `_order`), so the two calls agreed even with an unseeded generator. In real use, a resumed run and every other arm start in a new process and would see a different data order. That is exactly the defect the test exists to catch. The test now computes the same batch in a fresh interpreter (`subprocess`) and compares. Rerun of M8: killed by assertion.
