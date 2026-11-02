"""Stage 2D: B0's six blocks run as 2 prelude blocks, a shared 2-block loop and 2
coda blocks (plan section 7.5).

The module and parameter names are B0GPT's, so a seed starts from exactly B0's
weights and `param_groups` applies unchanged. Depth r runs the loop r times; r = 1
is B0's own computation (six blocks). Training draws one depth per micro-batch;
analysis reads out every depth from one forward (`all_depths`) and can execute
a per-token exit policy (`depth` as a tensor): an exited token keeps its state,
and later tokens read that frozen state in the passes it skips, so the cost of
exiting early on what other tokens see is measured, not assumed.
"""
import numpy as np
import torch
import torch.nn.functional as F

from clm.model import _seed_for
from clm.s2model import B0Config, B0GPT, init_b0, rms


def sample_depth(seed, step, micro_index, max_depth):
    """The depth of one micro-batch: uniform on 1..max_depth, a function of (seed, step, micro) only,
    so a resumed run draws what an unbroken run would."""
    rng = np.random.default_rng([_seed_for(seed, "loop-depth") % 2**31, step, micro_index])
    return int(rng.integers(1, max_depth + 1))


class LoopGPT(B0GPT):
    def __init__(self, cfg, max_depth=6):
        if cfg.n_layer != 6:
            raise ValueError("the looped layout needs six blocks: 2 prelude, 2 shared loop, 2 coda")
        super().__init__(cfg)
        self.max_depth = max_depth

    def _stem(self, idx):
        x = rms(self.embed(idx))
        ves = {layer: emb(idx) for layer, emb in zip(self.ve_layers, self.value_embed)}
        for i in (0, 1):
            x = self.blocks[i](x, self.cos, self.sin, ves.get(i))
        return x, ves

    def _loop(self, x):
        return self.blocks[3](self.blocks[2](x, self.cos, self.sin), self.cos, self.sin)

    def _read(self, x, ves):
        for i in (4, 5):
            x = self.blocks[i](x, self.cos, self.sin, ves.get(i))
        logits = self.head(rms(x)).float()
        return self.cfg.softcap * torch.tanh(logits / self.cfg.softcap)

    def forward(self, idx, targets=None, depth=1):
        """depth: an int (every token runs that many passes) or a (batch, time) integer tensor
        of per-token exit depths, each in 1..max_depth."""
        x, ves = self._stem(idx)
        if isinstance(depth, int):
            if not 1 <= depth <= self.max_depth:
                raise ValueError(f"depth {depth} outside 1..{self.max_depth}")
            for _ in range(depth):
                x = self._loop(x)
        else:
            if int(depth.min()) < 1 or int(depth.max()) > self.max_depth:
                raise ValueError(f"per-token depths outside 1..{self.max_depth}")
            for i in range(1, self.max_depth + 1):
                x = torch.where((depth >= i).unsqueeze(-1), self._loop(x), x)
        logits = self._read(x, ves)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    @torch.no_grad()
    def all_depths(self, idx, targets):
        """Per-token loss at every depth 1..max_depth, (max_depth, batch, time), and the relative change
        ||x_r - x_(r-1)|| / ||x_r|| of the loop state at each pass, also (max_depth, batch, time)."""
        x, ves = self._stem(idx)
        losses, change = [], []
        for _ in range(self.max_depth):
            new = self._loop(x)
            change.append((new - x).float().norm(dim=-1) / new.float().norm(dim=-1).clamp_min(1e-12))
            x = new
            logits = self._read(x, ves)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                          reduction="none").view(targets.shape))
        return torch.stack(losses), torch.stack(change)


def build_loop(cfg, seed, max_depth=6):
    model = LoopGPT(cfg, max_depth)
    init_b0(model, seed)
    return model
