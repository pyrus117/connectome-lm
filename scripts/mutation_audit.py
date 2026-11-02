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
