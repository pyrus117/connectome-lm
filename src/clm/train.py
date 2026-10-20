"""Train one (arm, seed) run of the Stage 1 experiment.

    python -m clm.train --arm real --seed 0 --config configs/stage1.json --out runs/real-0

Resumes from <out>/ckpt.pt when present. Writes log.jsonl as it goes and
results.json on completion. A non-finite loss or gradient stops the run with
status.json {"status": "diverged"} and exit code 2.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import torch

from clm.block import ConnectomeFFN
from clm.data import batch_starts, eval_windows, get_batch, open_tokens
from clm.model import ModelConfig, RUN_ARMS, build_model

ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    region: str = "data/mb_alpn_783.npz"
    train_bin: str = "data/fineweb/train.bin"
    val_bin: str = "data/fineweb/val.bin"
    tokens: int = 200_000_000
    batch: int = 64
    micro: int = 16
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup: int = 500
    wd: float = 0.1
    clip: float = 1.0
    eval_every: int = 1000
    eval_tokens: int = 2_097_152
    ckpt_every: int = 250


def lr_at(step, steps, cfg):
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    frac = min(1.0, (step - cfg.warmup) / max(1, steps - 1 - cfg.warmup))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * frac))


def load_config(path):
    raw = json.loads(Path(path).read_text())
    model = {**raw["model"], "conn_layers": tuple(raw["model"]["conn_layers"])}
    return ModelConfig(**model), TrainConfig(**raw["train"])


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).exists() else None


def _git():
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        return None, None
    return head.stdout.strip(), bool(git("status", "--porcelain").stdout.strip())


def _autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def param_groups(model, wd):
    decay, rest = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 or name.endswith(".values") else rest).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}]


@torch.no_grad()
def evaluate(model, val, ctx, n_eval, micro, device):
    model.eval()
    starts = eval_windows(len(val), ctx, n_eval)
    total = 0.0
    for i in range(0, len(starts), micro):
        x, y = get_batch(val, starts[i:i + micro], ctx, device)
        with _autocast(device):
            _, loss = model(x, y)
        total += loss.item() * len(x)
    model.train()
    return total / len(starts)


def _write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def run(arm, seed, mcfg, tcfg, out, device, max_steps=None, config_path=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    region = ROOT / tcfg.region
    model = build_model(mcfg, arm, seed, region).to(device)
    opt = torch.optim.AdamW(param_groups(model, tcfg.wd), lr=tcfg.lr, betas=(0.9, 0.95))
    steps = tcfg.tokens // (tcfg.batch * mcfg.ctx)
    ckpt, start = out / "ckpt.pt", 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        if (state["arm"], state["seed"]) != (arm, seed):
            raise RuntimeError(f"checkpoint in {out} is {state['arm']}-{state['seed']}, not {arm}-{seed}")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start = state["step"]
    train, val = open_tokens(ROOT / tcfg.train_bin), open_tokens(ROOT / tcfg.val_bin)
    accum, end = tcfg.batch // tcfg.micro, steps if max_steps is None else min(steps, max_steps)
    val_loss = None
    with open(out / "log.jsonl", "a") as log:
        for step in range(start, end):
            t0 = time.time()
            lr = lr_at(step, steps, tcfg)
            for g in opt.param_groups:
                g["lr"] = lr
            starts = batch_starts(len(train), mcfg.ctx, tcfg.batch, seed, step)
            loss_sum = 0.0
            for k in range(accum):
                x, y = get_batch(train, starts[k * tcfg.micro:(k + 1) * tcfg.micro], mcfg.ctx, device)
                with _autocast(device):
                    _, loss = model(x, y)
                (loss / accum).backward()
                loss_sum += loss.item() / accum
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.clip).item()
            if not (math.isfinite(loss_sum) and math.isfinite(gnorm)):
                _write(out / "status.json", {"status": "diverged", "step": step,
                                             "loss": loss_sum, "grad_norm": gnorm})
                raise SystemExit(2)
            opt.step()
            opt.zero_grad(set_to_none=True)
            rec = {"step": step, "loss": loss_sum, "lr": lr, "grad_norm": gnorm,
                   "seconds": round(time.time() - t0, 3)}
            if (step + 1) % tcfg.eval_every == 0 or step + 1 == steps:
                val_loss = rec["val_loss"] = evaluate(model, val, mcfg.ctx, tcfg.eval_tokens,
                                                       tcfg.micro, device)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if (step + 1) % tcfg.ckpt_every == 0 or step + 1 == end:
                torch.save({"arm": arm, "seed": seed, "step": step + 1,
                            "model": model.state_dict(), "opt": opt.state_dict()}, out / "ckpt.tmp")
                (out / "ckpt.tmp").replace(ckpt)
    if end < steps:
        return
    commit, dirty = _git()
    _write(out / "results.json", {
        "status": "complete", "arm": arm, "seed": seed,
        "fingerprints": sorted({m.fingerprint for m in model.modules() if isinstance(m, ConnectomeFFN)}),
        "params_total": sum(p.numel() for p in model.parameters()),
        "tokens_seen": steps * tcfg.batch * mcfg.ctx, "steps": steps,
        "final_val_loss": val_loss,
        "config_sha256": _sha256(config_path), "prereg_sha256": _sha256(ROOT / "PREREG.md"),
        "git_commit": commit, "git_dirty": dirty,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=RUN_ARMS)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int)
    args = ap.parse_args()
    mcfg, tcfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(args.arm, args.seed, mcfg, tcfg, args.out, device, args.max_steps, args.config)


if __name__ == "__main__":
    main()
