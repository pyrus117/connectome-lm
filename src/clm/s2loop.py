"""Stage 2D: B0's six blocks run as 2 prelude blocks, a shared 2-block loop and 2
coda blocks (plan section 7.5).

The module and parameter names are B0GPT's, so a seed starts from exactly B0's
weights and `param_groups` applies unchanged. Depth r runs the loop r times; r = 1
is B0's own computation (six blocks). Training draws one depth per micro-batch;
analysis reads out every depth from one forward (`all_depths`) and can execute
a per-token exit policy (`depth` as a tensor): an exited token keeps its state,
and later tokens read that frozen state in the passes it skips, so the cost of
exiting early on what other tokens see is measured, not assumed.

Stage 2H (follow-up 5A) adds two optional changes, both off by default so Stage 2D's
model is unchanged. `gain` gives each pass its own per-channel residual gain
(max_depth x d_model parameters, `loop_gain`, initialised to 1, so a fresh model is
the ungained one): pass r moves the stream by gain_r * (loop(x) - x). `forward_all`
trains every depth in one forward, the mean of the loss at each readout.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from clm.model import _seed_for
from clm.s2model import B0Config, B0GPT, init_b0, rms


def sample_depth(seed, step, micro_index, max_depth):
    """The depth of one micro-batch: uniform on 1..max_depth, a function of (seed, step, micro) only,
    so a resumed run draws what an unbroken run would."""
    rng = np.random.default_rng([_seed_for(seed, "loop-depth") % 2**31, step, micro_index])
    return int(rng.integers(1, max_depth + 1))


class LoopGPT(B0GPT):
    def __init__(self, cfg, max_depth=6, gain=False):
        if cfg.n_layer != 6:
            raise ValueError("the looped layout needs six blocks: 2 prelude, 2 shared loop, 2 coda")
        super().__init__(cfg)
        self.max_depth = max_depth
        self.loop_gain = nn.Parameter(torch.ones(max_depth, cfg.d_model)) if gain else None

    def _stem(self, idx):
        x = rms(self.embed(idx))
        ves = {layer: emb(idx) for layer, emb in zip(self.ve_layers, self.value_embed)}
        for i in (0, 1):
            x = self.blocks[i](x, self.cos, self.sin, ves.get(i))
        return x, ves

    def _loop(self, x):
        return self.blocks[3](self.blocks[2](x, self.cos, self.sin), self.cos, self.sin)

    def _pass(self, x, r):
        """Pass r (1-based) of the loop: the shared blocks, scaled per channel by that pass's gain if the model has one."""
        new = self._loop(x)
        return new if self.loop_gain is None else x + self.loop_gain[r - 1] * (new - x)

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
            for r in range(1, depth + 1):
                x = self._pass(x, r)
        else:
            if int(depth.min()) < 1 or int(depth.max()) > self.max_depth:
                raise ValueError(f"per-token depths outside 1..{self.max_depth}")
            for i in range(1, self.max_depth + 1):
                x = torch.where((depth >= i).unsqueeze(-1), self._pass(x, i), x)
        logits = self._read(x, ves)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    def forward_all(self, idx, targets):
        """Loss at every depth 1..max_depth from one forward, shape (max_depth,); training minimises its mean."""
        x, ves = self._stem(idx)
        losses = []
        for r in range(1, self.max_depth + 1):
            x = self._pass(x, r)
            logits = self._read(x, ves)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)))
        return torch.stack(losses)

    @torch.no_grad()
    def all_depths(self, idx, targets):
        """Per-token loss at every depth 1..max_depth, (max_depth, batch, time), and the relative change
        ||x_r - x_(r-1)|| / ||x_r|| of the loop state at each pass, also (max_depth, batch, time)."""
        x, ves = self._stem(idx)
        losses, change = [], []
        for r in range(1, self.max_depth + 1):
            new = self._pass(x, r)
            change.append((new - x).float().norm(dim=-1) / new.float().norm(dim=-1).clamp_min(1e-12))
            x = new
            logits = self._read(x, ves)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                          reduction="none").view(targets.shape))
        return torch.stack(losses), torch.stack(change)


def build_loop(cfg, seed, max_depth=6, gain=False):
    model = LoopGPT(cfg, max_depth, gain)
    init_b0(model, seed)
    return model
