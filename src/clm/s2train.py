"""Train one Stage 2 baseline run (B0).

    python -m clm.s2train --config configs/stage2_0.json --size M --seed 0 --out runs2/b0-M-0

Resumes from <out>/ckpt.pt. Writes log.jsonl as it goes and results.json at the
end, including training wall-clock (the Stage 2 headline axis) and peak VRAM.
A non-finite loss or gradient writes status.json {"status": "diverged"} and exits 2.
"""
import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch

from clm.data import batch_starts, eval_windows, get_batch, open_tokens
from clm.s2model import B0Config, build_b0, param_groups
from clm.train import ROOT, _autocast, _git, _sha256, _write

PREREG = ROOT / "PREREG_stage2_0.md"


@dataclasses.dataclass(frozen=True)
class S2TrainConfig:
    train_bin: str = "data/fineweb/train.bin"
    val_bin: str = "data/fineweb/val.bin"
    tokens: int = 200_000_000
    batch: int = 32               # sequences per optimizer step (32 x 512 = 16,384 tokens)
    micro: int = 8
    optimizer: str = "adamw"
    lr: float = 2e-3
    wd: float = 0.1
    warmup: int = 200
    clip: float = 1.0
    eval_every: int = 2000
    eval_tokens: int = 2_097_152
    ckpt_every: int = 500


def lr_at(step, steps, cfg):
    """Linear warmup, then linear decay to zero at the last step."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    return cfg.lr * max(0.0, (steps - 1 - step) / max(1, steps - 1 - cfg.warmup))


def load_config(path, size):
    raw = json.loads(Path(path).read_text())
    if size not in raw["sizes"]:
        raise SystemExit(f"{path} has no size {size!r}; sizes: {sorted(raw['sizes'])}")
    return B0Config(**raw["sizes"][size]), S2TrainConfig(**raw["train"])


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


def run(size, seed, mcfg, tcfg, out, device, max_steps=None, config_path=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = build_b0(mcfg, seed).to(device)
    opts = param_groups(model, tcfg.optimizer, tcfg.lr, tcfg.wd)
    steps = tcfg.tokens // (tcfg.batch * mcfg.ctx)
    if tcfg.batch % tcfg.micro:
        raise ValueError("batch must be a multiple of micro")
    ckpt, start, train_seconds = out / "ckpt.pt", 0, 0.0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        if (state["size"], state["seed"]) != (size, seed):
            raise RuntimeError(f"checkpoint in {out} is {state['size']}-{state['seed']}, not {size}-{seed}")
        model.load_state_dict(state["model"])
        for o, s in zip(opts, state["opts"]):
            o.load_state_dict(s)
        start, train_seconds = state["step"], state["train_seconds"]
    train, val = open_tokens(ROOT / tcfg.train_bin), open_tokens(ROOT / tcfg.val_bin)
    accum, end = tcfg.batch // tcfg.micro, steps if max_steps is None else min(steps, max_steps)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    val_loss = None
    with open(out / "log.jsonl", "a") as log:
        for step in range(start, end):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.time()
            lr = lr_at(step, steps, tcfg)
            for o in opts:
                for g in o.param_groups:
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
            for o in opts:
                o.step()
                o.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = time.time() - t0
            train_seconds += dt
            rec = {"step": step, "loss": loss_sum, "lr": lr, "grad_norm": gnorm, "seconds": round(dt, 4)}
            if (step + 1) % tcfg.eval_every == 0 or step + 1 == steps:
                val_loss = rec["val_loss"] = evaluate(model, val, mcfg.ctx, tcfg.eval_tokens, tcfg.micro, device)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if (step + 1) % tcfg.ckpt_every == 0 or step + 1 == end:
                torch.save({"size": size, "seed": seed, "step": step + 1, "train_seconds": train_seconds,
                            "model": model.state_dict(), "opts": [o.state_dict() for o in opts]},
                           out / "ckpt.tmp")
                (out / "ckpt.tmp").replace(ckpt)
    if end < steps:
        return
    commit, dirty = _git()
    _write(out / "results.json", {
        "status": "complete", "arm": "b0", "size": size, "seed": seed,
        "model": dataclasses.asdict(mcfg), "train": dataclasses.asdict(tcfg),
        "non_embedding_params": model.non_embedding_params(),
        "params_total": sum(p.numel() for p in model.parameters()),
        "tokens_seen": steps * tcfg.batch * mcfg.ctx, "steps": steps,
        "final_val_loss": val_loss, "train_seconds": round(train_seconds, 3),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None,
        "config_sha256": _sha256(config_path), "prereg_sha256": _sha256(PREREG),
        "git_commit": commit, "git_dirty": dirty,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-fraction", type=float, default=0.75)
    ap.add_argument("--optimizer", help="pilot override")
    ap.add_argument("--lr", type=float, help="pilot override")
    ap.add_argument("--tokens", type=int, help="pilot override")
    args = ap.parse_args()
    mcfg, tcfg = load_config(args.config, args.size)
    overrides = {k: v for k, v in (("optimizer", args.optimizer), ("lr", args.lr), ("tokens", args.tokens))
                 if v is not None}
    tcfg = dataclasses.replace(tcfg, **overrides)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    run(args.size, args.seed, mcfg, tcfg, args.out, device, config_path=args.config)


if __name__ == "__main__":
    main()
