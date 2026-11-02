"""Planning probe: training throughput of a modern small GPT on the laptop GPU,
sparse gather/scatter speed of a large table in host RAM, and the cost of
fly-hash addresses (fixed sparse expansion + top-k) on CPU and GPU.

Usage: .venv/bin/python scripts/throughput_probe.py gpu|cpu|addr
Results and conditions: notes/laptop_throughput_2026-11-02.md
"""
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 8192


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.up, self.down = nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d))
        return x + self.down(F.relu(self.up(self.n2(x))).square())


class GPT(nn.Module):
    def __init__(self, d, layers, heads):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.RMSNorm(d)

    def forward(self, idx):
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x)
        return F.linear(self.norm(x), self.emb.weight)


def gpu(d, layers, heads, batch, ctx, steps=15, warmup=5):
    model = GPT(d, layers, heads).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    n = sum(p.numel() for p in model.parameters()) - model.emb.weight.numel()
    x = torch.randint(0, VOCAB, (batch, ctx + 1), device="cuda")
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.float().view(-1, VOCAB), x[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    tok = batch * ctx / dt
    print(f"GPU d={d} L={layers} nonemb={n / 1e6:.1f}M micro={batch}x{ctx}: {dt * 1000:.0f} ms/step, "
          f"{tok / 1e3:.1f}k tok/s, {6 * n * tok / 1e12:.2f} TFLOP/s (6N), "
          f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, 200M tokens = {2e8 / tok / 3600:.1f} h",
          flush=True)
    del model, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def cpu_table(rows, dim, tokens, k, threads, reps=3):
    torch.set_num_threads(threads)
    table = torch.zeros(rows, dim, dtype=torch.float16)
    idx = torch.randint(0, rows, (tokens * k,))
    offsets = torch.arange(0, tokens * k, k)
    t0 = time.time()
    for _ in range(reps):
        F.embedding_bag(idx, table, offsets, mode="sum")
    gather = (time.time() - t0) / reps
    grad = torch.randn(tokens, dim, dtype=torch.float16).repeat_interleave(k, 0)
    t0 = time.time()
    for _ in range(reps):
        table.index_add_(0, idx, grad)
    scatter = (time.time() - t0) / reps
    print(f"CPU table {rows / 1e6:.1f}M x {dim} fp16 ({rows * dim * 2 / 2**30:.1f} GiB), {tokens} tokens x k={k}, "
          f"{threads} threads: gather {gather * 1000:.0f} ms ({tokens * k / gather / 1e6:.0f}M rows/s), "
          f"scatter-add {scatter * 1000:.0f} ms ({tokens * k / scatter / 1e6:.0f}M rows/s)", flush=True)


def fly_addresses(tokens=16384, odour=192, units_a=4096, units_b=2048, in_degree=7, winners=16, reps=3):
    """One n-gram order: drives = odour @ J (J binary, in_degree ones per unit), then top-k per sub-codebook."""
    gen = torch.Generator().manual_seed(0)
    o = torch.randn(tokens, odour, generator=gen)
    wiring = torch.zeros(odour, units_a + units_b)
    for u in range(units_a + units_b):
        wiring[torch.randperm(odour, generator=gen)[:in_degree], u] = 1.0

    def addresses(x, w):
        d = x @ w
        return d[:, :units_a].topk(winners, dim=1).indices, d[:, units_a:].topk(winners, dim=1).indices

    print(f"fly addresses, one order: {tokens} tokens, odour {odour}, units {units_a}+{units_b}, "
          f"K={in_degree}, top-{winners}", flush=True)
    for threads in (8, 16):
        torch.set_num_threads(threads)
        t0 = time.time()
        for _ in range(reps):
            addresses(o, wiring)
        print(f"  CPU fp32, {threads} threads: {(time.time() - t0) / reps * 1000:.0f} ms", flush=True)
    if torch.cuda.is_available():
        og, wg = o.cuda().half(), wiring.cuda().half()
        for i in range(reps + 3):
            if i == 3:
                torch.cuda.synchronize()
                t0 = time.time()
            ia, ib = addresses(og, wg)
            torch.cat([ia, ib], 1).cpu()
        print(f"  GPU fp16, indices copied to host: {(time.time() - t0) / reps * 1000:.1f} ms", flush=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("gpu", "cpu", "addr"):
        raise SystemExit("usage: throughput_probe.py gpu|cpu|addr")
    if sys.argv[1] == "gpu":
        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device")
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
        for d, layers, heads, batch, ctx in [(256, 6, 4, 12, 512), (384, 6, 6, 8, 512), (512, 8, 8, 8, 512)]:
            gpu(d, layers, heads, batch, ctx)
    elif sys.argv[1] == "cpu":
        for threads in (8, 16):
            cpu_table(16_777_216, 64, 16384, 32, threads)
    else:
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.75, 0)
        fly_addresses()


if __name__ == "__main__":
    main()
