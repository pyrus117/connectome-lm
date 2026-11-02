"""Plant each known bug in a throwaway copy of the repo and check a test kills it.

The real source is never edited: each mutant is applied to a fresh copy under
/tmp/clm-mut, and that copy's src/ is put first on PYTHONPATH. Writes the
results to /tmp/clm_mutation_results.json and prints one line per mutant.
Every mutant must be KILLED by an assertion; C1 must survive; C2 must error.

    python scripts/mutation_audit.py
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "bin" / "python"
WORK = Path("/tmp/clm-mut")

MUTANTS = [
    ("M1", "src/clm/graphs.py", "post = degree_preserving_shuffle(n, pre, post, rng)", "post = post",
     "tests/test_graphs.py::test_shuffle_moves_most_edges"),
    ("M2", "src/clm/graphs.py", "post = rng.permutation(post)", "post = rng.integers(0, n, post.size)",
     "tests/test_graphs.py::test_shuffle_keeps_every_neurons_in_and_out_degree"),
    ("M3", "src/clm/block.py", "h = (1 - self.leak) * h + self.leak * F.gelu(", "h = self.leak * F.gelu(",
     "tests/test_brain.py::test_brain_matches_the_dense_reference_layer_by_layer"),
    ("M4", "src/clm/model.py", "x, h = layer(x, h, self.brain)", "x, h = layer(x, None, self.brain)",
     "tests/test_model.py::test_each_layer_receives_the_state_the_previous_layer_left"),
    ("M5", "src/clm/block.py", "return _norm(h) if self.norm else h", "return h",
     "tests/test_brain.py::test_brain_matches_the_dense_reference_layer_by_layer"),
    ("M6", "src/clm/model.py", "is_causal=True", "is_causal=False",
     "tests/test_model.py::test_logits_at_a_position_ignore_later_tokens"),
    ("M7", "src/clm/model.py", "g = torch.Generator().manual_seed(_seed_for(seed, name))", "g = None",
     "tests/test_model.py::test_parameters_shared_by_two_arms_start_identical"),
    ("M8", "src/clm/data.py", "np.random.default_rng(seed).permutation(windows)",
     "np.random.default_rng().permutation(windows)",
     "tests/test_data.py::test_sampler_depends_only_on_seed_and_step"),
    ("M9", "src/clm/train.py", "cfg.lr * (step + 1) / cfg.warmup", "cfg.lr * step / cfg.warmup",
     "tests/test_train.py::test_lr_warms_up_then_decays_to_the_floor"),
    ("M10", "src/clm/train.py", 'start = state["step"]', "start = 0",
     "tests/test_train.py::test_interrupted_and_resumed_run_ends_bit_identical"),
    ("M11", "src/clm/analyze.py", "if mean <= -sesoi and all(v < 0 for v in d):", "if mean <= -sesoi:",
     "tests/test_analyze.py::test_verdict_rules"),
    ("M12", "src/clm/analyze.py", "or not math.isfinite(loss)", "or False",
     "tests/test_analyze.py::test_check_run_names_each_defect"),
    ("M13", "src/clm/prep.py", "% 100 == 0 else", "% 100 != 0 else",
     "tests/test_prep.py::test_split_sends_about_one_percent_to_validation_and_is_stable"),
    ("M14", "src/clm/graphs.py", "return round(math.sqrt(nnz + n))", "return round(math.sqrt(nnz))",
     "tests/test_graphs.py::test_dense_arm_is_fully_connected_and_matches_the_regions_parameters"),
    ("M15", "src/clm/train.py", '"fingerprints": [model.brain.fingerprint],', '"fingerprints": [],',
     "tests/test_train.py::test_completed_run_reports_what_the_analysis_checks"),
    ("M16", "src/clm/block.py", "h = self.leak * F.gelu(drive)", "h = F.gelu(drive)",
     "tests/test_brain.py::test_brain_matches_the_dense_reference_layer_by_layer"),
    # Stage 2 baseline (src/clm/s2model.py, src/clm/s2train.py)
    ("M17", "src/clm/s2model.py", "is_causal=True", "is_causal=False",
     "tests/test_s2.py::test_logits_at_a_position_ignore_later_tokens"),
    ("M18", "src/clm/s2model.py", 'if name.endswith(_ZERO_INIT) or name.endswith("ve_gate"):',
     'if name.endswith("ve_gate"):', "tests/test_s2.py::test_fresh_model_predicts_uniformly"),
    ("M19", "src/clm/s2model.py", "logits = self.cfg.softcap * torch.tanh(logits / self.cfg.softcap)",
     "logits = logits", "tests/test_s2.py::test_softcap_bounds_the_logits"),
    ("M20", "src/clm/s2model.py", "return torch.cat((a * c - b * s, a * s + b * c), dim=-1)", "return x",
     "tests/test_s2.py::test_rotary_makes_attention_scores_depend_on_offset_only"),
    ("M21", "src/clm/s2model.py", '(matrices if name.startswith("blocks.") and p.dim() == 2 else rest)',
     "(matrices if p.dim() == 2 else rest)",
     "tests/test_s2.py::test_muon_gets_only_block_matrices_and_every_parameter_is_optimised_once"),
    ("M22", "src/clm/s2model.py", "g = torch.Generator().manual_seed(_seed_for(seed, name))", "g = None",
     "tests/test_s2.py::test_same_seed_same_start_different_seed_different_start"),
    ("M23", "src/clm/s2train.py", "return cfg.lr * max(0.0, (steps - 1 - step) / max(1, steps - 1 - cfg.warmup))",
     "return cfg.lr", "tests/test_s2.py::test_lr_warms_up_then_falls_linearly_to_zero"),
    ("M24", "src/clm/s2train.py", 'start, train_seconds = state["step"], state["train_seconds"]',
     'start, train_seconds = 0, state["train_seconds"]',
     "tests/test_s2.py::test_interrupted_and_resumed_run_ends_bit_identical"),
    ("M25", "src/clm/s2model.py", "blocks = upd.view(self.split.get(id(p), 1), -1, upd.size(1))",
     "blocks = upd.view(1, -1, upd.size(1))",
     "tests/test_s2.py::test_head_muon_orthogonalises_each_head_block_on_its_own"),
    ("M26", "src/clm/s2model.py", 'g["adjust_lr_fn"], blocks.shape[1:])', 'g["adjust_lr_fn"], p.shape)',
     "tests/test_s2.py::test_head_muon_orthogonalises_each_head_block_on_its_own"),
    # Stage 2A count table (src/clm/counts.py)
    ("M27", "src/clm/counts.py", "ok &= a[j:j + n] != eot", "ok &= a[j:j + n] >= 0",
     "tests/test_counts.py::test_no_history_spans_a_document_boundary_but_a_continuation_may_be_eot"),
    ("M28", "src/clm/counts.py", "(a[:n] < hi)", "(a[:n] <= hi)",
     "tests/test_counts.py::test_partitions_and_chunks_do_not_change_the_table"),
    ("M29", "src/clm/counts.py", "min(start + chunk, n_tokens - k)\n", "min(start + chunk, n_tokens - k - 1)\n",
     "tests/test_counts.py::test_table_matches_a_brute_force_count"),
    ("M30", "src/clm/counts.py", "top = rank < TOP", "top = rank <= TOP",
     "tests/test_counts.py::test_only_the_top_16_continuations_are_kept_but_the_history_count_is_complete"),
    ("M31", "src/clm/counts.py", "np.lexsort((e_tok, -e_cnt, e_hist))",
     "np.lexsort((-e_tok.astype(np.int64), -e_cnt, e_hist))",
     "tests/test_counts.py::test_only_the_top_16_continuations_are_kept_but_the_history_count_is_complete"),
    ("M32", "src/clm/counts.py", "hcount = np.add.reduceat(cnt, starts)", "hcount = np.maximum.reduceat(cnt, starts)",
     "tests/test_counts.py::test_table_matches_a_brute_force_count"),
    ("M33", "src/clm/counts.py", "keep = hcount >= c_min", "keep = hcount > c_min",
     "tests/test_counts.py::test_c_min_drops_rarer_histories_and_keeps_those_at_the_threshold"),
    ("M34", "src/clm/counts.py", "if table_bytes(kept_rows[c], kept_entries[c]) <= budget_bytes:",
     "if table_bytes(kept_rows[c], kept_entries[c]) < budget_bytes:",
     "tests/test_counts.py::test_pick_cmin_takes_the_smallest_threshold_that_fits"),
    ("M35", "src/clm/counts.py", "weights=np.minimum(fanout, TOP)", "weights=fanout",
     "tests/test_counts.py::test_profile_predicts_the_size_of_every_threshold"),
    ("M36", "src/clm/counts.py", 'out["lens"].append(np.minimum(fanout[keep], TOP))',
     'out["lens"].append(fanout[keep])',
     "tests/test_counts.py::test_only_the_top_16_continuations_are_kept_but_the_history_count_is_complete"),
    # Stage 2A COUNT read, n-gram reference, memory arm (src/clm/countread.py, ngram_lm.py, s2model.py, s2train.py)
    ("M37", "src/clm/countread.py", "ecnt = ecnt - (live & (etok == np.asarray(y)[bi, ti][:, None]))", "ecnt = ecnt",
     "tests/test_countread.py::test_read_matches_a_brute_force_count_with_and_without_leave_one_out"),
    ("M38", "src/clm/countread.py", "                hc = hc - 1\n", "                hc = hc\n",
     "tests/test_countread.py::test_read_matches_a_brute_force_count_with_and_without_leave_one_out"),
    ("M39", "src/clm/countread.py", "keep = hc >= c_min", "keep = hc >= 0",
     "tests/test_countread.py::test_read_matches_a_brute_force_count_with_and_without_leave_one_out"),
    ("M40", "src/clm/countread.py", "hit = table.keys[np.minimum(row, table.keys.size - 1)] == key",
     "hit = row < table.keys.size",
     "tests/test_countread.py::test_read_matches_a_brute_force_count_with_and_without_leave_one_out"),
    ("M41", "src/clm/countread.py", ", 0.0) / len(self.tables)", ", 0.0)",
     "tests/test_countread.py::test_read_matches_a_brute_force_count_with_and_without_leave_one_out"),
    ("M42", "src/clm/countread.py", "at = np.where(live, lo[:, None] + np.arange(kk), 0)",
     "at = np.where(live, lo[:, None] + np.arange(kk)[::-1], 0)",
     "tests/test_countread.py::test_read_matches_a_brute_force_count_with_and_without_leave_one_out"),
    ("M43", "src/clm/countread.py", "(nxt, self.pool.submit(self.make, nxt))", "(nxt, self.pool.submit(self.make, step))",
     "tests/test_countread.py::test_prefetcher_hands_over_each_step_it_was_asked_for_and_reads_ahead"),
    ("M44", "src/clm/countread.py", "tok, score = reader.lookup(x, y if loo else None)", "tok, score = reader.lookup(x, y)",
     "tests/test_s2count.py::test_training_reads_leave_one_out_counts_only_when_asked"),
    ("M45", "src/clm/s2model.py", 'per_sample_weights=score.reshape(b * t, s), mode="sum")',
     'per_sample_weights=torch.ones_like(score).reshape(b * t, s), mode="sum")',
     "tests/test_s2count.py::test_count_input_is_the_score_weighted_sum_of_head_rows"),
    ("M46", "src/clm/s2model.py", "F.embedding_bag(tok.reshape(b * t, s), self.head.weight,",
     "F.embedding_bag(tok.reshape(b * t, s), self.embed.weight,",
     "tests/test_s2count.py::test_count_input_is_the_score_weighted_sum_of_head_rows"),
    ("M47", "src/clm/s2train.py", "tok, score = reader.lookup(x.numpy())", "tok, score = reader.lookup(x.numpy(), y.numpy())",
     "tests/test_s2count.py::test_validation_reads_counts_without_leave_one_out_and_training_reads_them_with_it"),
    ("M48", "src/clm/ngram_lm.py", "denom = np.maximum(1.0 - (low_stored * live).sum(1), 1e-12)", "denom = np.ones_like(mass)",
     "tests/test_ngram_lm.py::test_probabilities_sum_to_one_over_the_vocabulary_at_every_position"),
    ("M49", "src/clm/ngram_lm.py", "(table.cnt[at] - self.d)", "table.cnt[at]",
     "tests/test_ngram_lm.py::test_a_stored_history_gives_discounted_ratios_and_a_new_one_gives_the_unigram"),
    ("M50", "src/clm/countread.py", "x[:, last - (k - 1) + j].astype", "x[:, last - k + j].astype",
     "tests/test_countread.py::test_read_matches_a_brute_force_count_with_and_without_leave_one_out"),
    ("M51", "src/clm/countread.py", "BUCKET_EDGES = (1, 10, 100, 1000)", "BUCKET_EDGES = (2, 10, 100, 1000)",
     "tests/test_countread.py::test_buckets_split_at_1_10_100_and_1000_and_skip_short_windows"),
    ("M52", "src/clm/countread.py", "out[bi, ti] = table.hcount[row]", "out[bi, ti] = 1",
     "tests/test_countread.py::test_history_counts_report_the_stored_count_zero_for_unstored_and_minus_one_for_short_windows"),
    # Stage 2D looped model and depth law (src/clm/s2loop.py, depth_law.py, depth_eval.py, s2train.py)
    ("M53", "src/clm/s2loop.py", "            for _ in range(depth):\n", "            for _ in range(1):\n",
     "tests/test_s2loop.py::test_depth_one_is_the_b0_forward_and_depth_r_runs_the_loop_r_times"),
    ("M54", "src/clm/s2loop.py", "x = torch.where((depth >= i).unsqueeze(-1), self._loop(x), x)",
     "x = torch.where((depth >= i).unsqueeze(-1), self._loop(x), self._loop(x))",
     "tests/test_s2loop.py::test_an_exited_token_keeps_its_state_and_later_tokens_read_it_frozen"),
    ("M55", "src/clm/s2loop.py", "np.random.default_rng([_seed_for(seed, \"loop-depth\") % 2**31, step, micro_index])",
     "np.random.default_rng([_seed_for(seed, \"loop-depth\") % 2**31, 0, micro_index])",
     "tests/test_s2loop.py::test_sampled_depths_cover_the_range_and_depend_only_on_seed_step_and_micro"),
    ("M56", "src/clm/s2loop.py", "/ new.float().norm(dim=-1).clamp_min(1e-12))", "/ x.float().norm(dim=-1).clamp_min(1e-12))",
     "tests/test_s2loop.py::test_all_depths_match_separate_forwards_and_report_the_state_change"),
    ("M57", "src/clm/depth_law.py", "return cost.argmin(axis=0) + 1", "return cost.argmin(axis=0)",
     "tests/test_depth_law.py::test_oracle_picks_the_depth_minimising_loss_plus_price_times_depth"),
    ("M58", "src/clm/depth_law.py", "((y - (a + s * x)) ** 2).sum() / ss_tot", "((y - (a + s * x))).sum() / ss_tot",
     "tests/test_depth_law.py::test_affine_fit_recovers_a_line_and_scores_noise_low"),
    ("M59", "src/clm/depth_law.py", "lo <= r[\"mean_depth\"] <= max_depth - margin", "lo <= r[\"mean_depth\"] <= max_depth",
     "tests/test_depth_law.py::test_flat_marginal_gains_do_not_give_one"),
    ("M60", "src/clm/depth_eval.py", "return first + 1", "return first",
     "tests/test_depth_eval.py::test_free_exit_stops_at_the_first_pass_below_the_threshold_or_runs_to_the_end"),
    ("M61", "src/clm/s2train.py", "kw[\"depth\"] = sample_depth(seed, step, k, tcfg.loop_max)",
     "kw[\"depth\"] = sample_depth(seed, 0, k, tcfg.loop_max)",
     "tests/test_s2loop.py::test_loop_run_reports_the_loss_at_every_depth"),
    # Stage 2B-0 Kenyon hash and offline mixture (src/clm/kenyon.py, kenyon_lm.py, counts.py group_weighted)
    ("M62", "src/clm/kenyon.py", "drive = drive / self._norm[t].sum(1).sqrt().clamp_min(1e-6)[:, None]", "drive = drive",
     "tests/test_kenyon.py::test_activation_is_the_projection_of_the_centred_normalised_concatenated_odour"),
    ("M63", "src/clm/kenyon.py", "self.theta += (step * np.sign(freq - target)).reshape(-1).astype(np.float32)",
     "self.theta -= (step * np.sign(freq - target)).reshape(-1).astype(np.float32)",
     "tests/test_kenyon.py::test_calibration_balances_how_often_each_cell_wins"),
    ("M64", "src/clm/kenyon.py", "return (_mix(packed.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)) % np.uint64(n_buckets))",
     "return (_mix(packed.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)))",
     "tests/test_kenyon.py::test_hash_bucket_is_deterministic_in_range_and_spread"),
    ("M65", "src/clm/kenyon.py", "np.arange(k - 1, -1, -1, dtype=np.uint64)", "np.arange(k, dtype=np.uint64)",
     "tests/test_kenyon.py::test_unpack_inverts_pack_history"),
    ("M66", "src/clm/kenyon.py", "part = (buckets[entry_hist], tok, cnt)", "part = (buckets[np.zeros_like(entry_hist)], tok, cnt)",
     "tests/test_kenyon.py::test_bucket_table_pools_continuation_counts_across_the_histories_in_a_bucket"),
    ("M67", "src/clm/kenyon.py", "keys[(tokens == eot).any(axis=1)] = NO_BUCKET", "keys[(tokens == eot).all(axis=1)] = NO_BUCKET",
     "tests/test_kenyon.py::test_a_history_holding_end_of_text_gets_no_bucket"),
    ("M68", "src/clm/kenyon.py", "m = min(chunk, a.size - off)", "m = a.size - off",
     "tests/test_kenyon.py::test_cooccurrence_matches_a_brute_force_count_and_skips_end_of_text"),
    ("M69", "src/clm/kenyon.py", "out[s:s + chunk] = win[:, 0, 0] * np.uint64(self.units) + win[:, 1, 0]",
     "out[s:s + chunk] = win[:, 0, 0] + win[:, 1, 0]",
     "tests/test_kenyon.py::test_distinct_winner_pairs_get_distinct_buckets"),
    ("M70", "src/clm/counts.py", "np.bincount(inverse2, weights=weights).astype(np.int64)", "np.bincount(inverse2).astype(np.int64)",
     "tests/test_kenyon.py::test_bucket_table_pools_continuation_counts_across_the_histories_in_a_bucket"),
    ("M71", "src/clm/kenyon_lm.py", "r = l * b / ((1 - l) * a + l * b)", "r = l * a / ((1 - l) * a + l * b)",
     "tests/test_kenyon_lm.py::test_fit_lambdas_finds_the_likelihood_maximum_and_never_mixes_short_windows"),
    ("M72", "src/clm/kenyon_lm.py", "return (1 - w) * pe + w * pb", "return w * pe + (1 - w) * pb",
     "tests/test_kenyon_lm.py::test_mixture_is_a_proper_blend_and_class_loss_reports_every_class"),
    ("M73", "src/clm/kenyon_lm.py", "(cls == 0) | (cls == 1)", "(cls == 0)",
     "tests/test_kenyon_lm.py::test_mixture_is_a_proper_blend_and_class_loss_reports_every_class"),
    ("M74", "src/clm/kenyon_lm.py", "return np.where(cls < 0, N_CLASSES - 1, cls)", "return np.where(cls < 0, 0, cls)",
     "tests/test_kenyon_lm.py::test_history_class_follows_the_stored_count_and_marks_windows_too_short_for_the_history"),
]


def fresh():
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir()
    for name in ("src", "tests"):
        shutil.copytree(REPO / name, WORK / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO / "pyproject.toml", WORK / "pyproject.toml")
    (WORK / "data").mkdir()
    for f in ("mb_alpn_783.npz", "cx_783.npz", "manifest.json"):
        shutil.copy(REPO / "data" / f, WORK / "data" / f)
    (WORK / "data" / "fineweb").symlink_to(REPO / "data" / "fineweb")    # read-only use


def mutate(path, old, new):
    p = WORK / path
    text = p.read_text()
    if text.count(old) != 1:
        sys.exit(f"{path}: expected exactly one {old!r}, found {text.count(old)}")
    p.write_text(text.replace(old, new))


def pytest(target):
    env = {**os.environ, "PYTHONPATH": str(WORK / "src")}
    r = subprocess.run([str(PY), "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", target],
                       cwd=WORK, env=env, capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout + r.stderr


def evidence(out):
    for line in out.splitlines():
        if line.startswith("E "):
            return line[:200]
    return out.strip().splitlines()[-1][:200] if out.strip() else ""


def verdict(code, out):
    if code == 0:
        return "SURVIVED"
    if "AssertionError" in out or "\nE       assert " in out or "Failed: DID NOT RAISE" in out:
        return "KILLED by assertion"
    return "KILLED by another error (test needs a real assertion)"


def main():
    results = []
    for mid, path, old, new, target in MUTANTS:
        fresh()
        mutate(path, old, new)
        code, out = pytest(target)
        results.append({"id": mid, "file": path, "test": target,
                        "verdict": verdict(code, out), "evidence": evidence(out)})
        print(json.dumps(results[-1]), flush=True)

    fresh()                                   # C1: behaviour-preserving rename must survive
    p = WORK / "src/clm/block.py"
    text = p.read_text()
    start, end = text.index("    def step(self, h, write):"), text.index("    def read(self, h):")
    p.write_text(text[:start] + text[start:end].replace("drive", "drv") + text[end:])
    code, out = pytest("tests")
    results.append({"id": "C1", "verdict": "SURVIVED (required)" if code == 0 else "KILLED (harness broken)",
                    "evidence": out.strip().splitlines()[-1][:200]})
    print(json.dumps(results[-1]), flush=True)

    fresh()                                   # C2: a broken import must error, not assert
    p = WORK / "src/clm/graphs.py"
    p.write_text('raise ImportError("mutant")\n' + p.read_text())
    code, out = pytest("tests")
    results.append({"id": "C2", "verdict": "ERROR at collection (required)"
                    if code != 0 and "ImportError" in out else "UNEXPECTED",
                    "evidence": out.strip().splitlines()[-1][:200]})
    print(json.dumps(results[-1]), flush=True)
    Path("/tmp/clm_mutation_results.json").write_text(json.dumps(results, indent=2))
    bad = [r["id"] for r in results if r["id"].startswith("M") and r["verdict"] != "KILLED by assertion"]
    bad += [r["id"] for r in results if r["id"].startswith("C") and "required" not in r["verdict"]]
    if bad:
        sys.exit(f"audit failed: {bad}")


if __name__ == "__main__":
    main()
