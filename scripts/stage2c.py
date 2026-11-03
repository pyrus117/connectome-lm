"""Stage 2C (PREREG_stage2c.md): fast count writes against gradient fine-tuning, and consolidation by replay.

    .venv/bin/python scripts/stage2c.py lists --runs runs2     # sequential lists, 3 task seeds
    .venv/bin/python scripts/stage2c.py twins --runs runs2     # twins: discrimination and transfer

Models: the finished B0-M seed 0 and COUNT-M seed 0 runs. Writes results/stage2c_lists.json or stage2c_twins.json.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from clm import kenyon as K
from clm.countread import reader_for
from clm.data import eval_windows, open_tokens
from clm.fastwrite import Overlay, quiz
from clm.fastwrite_arms import ArmConfig, run_consolidate, run_gradient, run_write
from clm.prep import EOT
from clm.s2model import B0Config, build_b0
from clm.s2train import S2TrainConfig
from clm.train import ROOT
from clm.vocab_task import make_lists, make_pairs, make_twin_pairs, quiz_batches, word_pool
from clm.word_sim import cached_similarity

WEIGHTS = (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0)  # write strengths tried on the calibration list (the COUNT read's own scores are at most 1/4)
LISTS, PAIRS, CALIBRATION_SEED = 4, 384, 100
TASK_SEEDS = (0, 1, 2)
PREREG = ROOT / "PREREG_stage2c.md"


def load(run, device):
    res = json.loads((run / "results.json").read_text())
    mcfg, tcfg = B0Config(**res["model"]), S2TrainConfig(**res["train"])
    model = build_b0(mcfg, res["seed"]).to(device)
    model.load_state_dict(torch.load(run / "ckpt.pt", map_location=device)["model"])
    reader = reader_for(ROOT / tcfg.count_dir, tcfg.count_k) if tcfg.memory == "count" else None
    return model, reader, mcfg


def fresh(model):
    return copy.deepcopy(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("lists", "twins"))
    ap.add_argument("--runs", type=Path, default=ROOT / "runs2")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--mem-fraction", type=float, default=0.75)
    ap.add_argument("--steps", type=int, default=ArmConfig.steps)
    ap.add_argument("--pairs", type=int, default=PAIRS)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(TASK_SEEDS))
    args = ap.parse_args()
    if not PREREG.exists():
        sys.exit("PREREG_stage2c.md is missing: register the stage before running it")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    tok = Tokenizer.from_file(str(ROOT / "data/fineweb/tokenizer.json"))
    pool, eot, vocab = word_pool(tok), tok.token_to_id(EOT), tok.get_vocab_size()
    val = open_tokens(ROOT / "data/fineweb/val.bin")
    starts = eval_windows(len(val), 512, 2_097_152)
    lm_windows, anchor_windows = starts[:2048:32], starts[2048::32]
    count_model, reader, _ = load(args.runs / "count-M-0", device)
    prereg = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    cfg = ArmConfig(steps=args.steps)

    if args.stage == "lists":
        b0_model, _, _ = load(args.runs / "b0-M-0", device)
        calib = {}
        cal_lists = make_lists(pool, 1, args.pairs, CALIBRATION_SEED)
        for w in WEIGHTS:
            r = run_write(fresh(count_model), reader, cal_lists, w, vocab, eot, cfg, device, val, lm_windows)
            calib[str(w)] = {"recall": r["acc_matrix"][0][0], "lm_loss_memory_on": r["lm_loss_memory_on"][0],
                             "lm_loss_memory_off": r["lm_loss_memory_off"]}
            print(f"write weight {w}: recall {calib[str(w)]['recall']:.3f}, language loss "
                  f"{calib[str(w)]['lm_loss_memory_on']:.3f} (off {calib[str(w)]['lm_loss_memory_off']:.3f})", flush=True)
        best = max(calib[str(w)]["recall"] for w in WEIGHTS)         # smallest weight within 5% of the best recall
        weight = min(w for w in WEIGHTS if calib[str(w)]["recall"] >= 0.95 * best)
        print("CHOSEN write weight", weight, flush=True)
        runs = {}
        for seed in args.seeds:
            lists = make_lists(pool, LISTS, args.pairs, seed)
            c = ArmConfig(steps=args.steps, task_seed=seed)
            t0 = time.time()
            runs[str(seed)] = {
                "gradient_b0": run_gradient(fresh(b0_model), None, lists, eot, c, device, val, lm_windows),
                "gradient_count": run_gradient(fresh(count_model), reader, lists, eot, c, device, val, lm_windows),
                "write": run_write(fresh(count_model), reader, lists, weight, vocab, eot, c, device, val, lm_windows),
                "consolidate": run_consolidate(fresh(count_model), reader, lists, weight, vocab, eot, c, device, val,
                                               lm_windows, anchor_windows)}
            for arm, r in runs[str(seed)].items():
                print(f"seed {seed} {arm}: final_mean {r['final_mean']:.3f} backward_transfer "
                      f"{r['backward_transfer']:.3f}", flush=True)
            runs[str(seed)]["seconds"] = round(time.time() - t0)
        out = {"prereg_sha256": prereg, "lists": LISTS, "pairs_per_list": args.pairs, "steps_per_list": args.steps,
               "lr": cfg.lr, "write_weight": weight, "calibration": calib, "runs": runs}
        (args.out / "stage2c_lists.json").write_text(json.dumps(out, indent=2) + "\n")
        return

    sim = cached_similarity(ROOT / "data/fineweb/train.bin", pool, vocab, ROOT / "data/fineweb/pool_sim.npz")
    weights = json.loads((args.out / "stage2c_lists.json").read_text())["write_weight"]
    keyfns = bucket_keyfns()
    runs = {}
    for seed in args.seeds:
        cues, answers = make_twin_pairs(pool, sim, args.pairs, seed)
        a, b = np.arange(0, len(cues), 2), np.arange(1, len(cues), 2)           # twins sit side by side
        entry = {}
        for name, keyfn in keyfns.items():
            def recall(overlay, q_cues, q_answers):
                qx, qy = quiz_batches(q_cues, q_answers, 64 if len(q_cues) % 64 == 0 else 32, seed, eot)
                return quiz(count_model, reader, overlay, qx, qy, 16, device)
            both = Overlay(vocab, weights, keyfn)
            both.write(cues, answers)
            one = Overlay(vocab, weights, keyfn)
            one.write(cues[a], answers[a])
            entry[name] = {
                "discrimination_both_written": recall(both, cues, answers),
                "twin_gets_stored_twins_answer_when_only_one_is_written": recall(one, cues[b], answers[a]),
                "stored_twin_recall_when_only_one_is_written": recall(one, cues[a], answers[a]),
                "twin_gets_its_own_answer_when_only_one_is_written": recall(one, cues[b], answers[b]),
                "twin_bucket_collision_rate": None if keyfn is None else float(np.mean(keyfn(cues[a]) == keyfn(cues[b])))}
            if name == "exact":                                  # twins against the same number of unrelated pairs
                c_cues, c_answers = make_pairs(pool, args.pairs, seed + 1000)
                control = Overlay(vocab, weight)
                control.write(c_cues, c_answers)
                entry[name]["random_pairs_control_recall"] = recall(control, c_cues, c_answers)
            print(f"seed {seed} {name}: {entry[name]}", flush=True)
        runs[str(seed)] = entry
    out = {"prereg_sha256": prereg, "write_weight": weights, "pairs": args.pairs, "runs": runs}
    (args.out / "stage2c_twins.json").write_text(json.dumps(out, indent=2) + "\n")


def bucket_keyfns():
    """Cue-token bucket functions for the twin test: exact only, the selected Kenyon hash for a one-token history,
    the same hash on random codes, and a plain hash of the token id (which gives no similarity at all)."""
    fns = {"exact": None, "token_hash": lambda t: K.hash_bucket(np.asarray(t, dtype=np.uint64), 2048 * 2048)}
    sweep = ROOT / "results" / "stage2b0_sweep.json"
    caches = sorted((ROOT / "data" / "count_2b0").glob("final-*/codes.npz")) if (ROOT / "data" / "count_2b0").exists() else []
    if not sweep.exists() or not caches:
        print("no Stage 2B-0 final-stage codes: the Kenyon bucket overlay is not run", flush=True)
        return fns
    cfg = json.loads(sweep.read_text())["selected_config"]
    z = np.load(caches[-1])
    corpus = open_tokens(ROOT / "data/fineweb/count_corpus.bin")
    sample = np.asarray(corpus[np.random.default_rng(0).integers(0, len(corpus), 80_000)]).astype(np.int64)[:, None]
    for name, codes in (("kenyon", z[cfg["codes"]]), ("kenyon_random_codes", z["random"])):
        h = K.KenyonHash(codes, 1, 2048, cfg["winners"], cfg["in_degree"], cfg["thresholds"], z["freq"], seed=0)
        h.calibrate(sample[:40_000])
        fns[name] = lambda t, h=h: h.bucket_keys(np.asarray(t, dtype=np.int64)[:, None])
    return fns


if __name__ == "__main__":
    main()
