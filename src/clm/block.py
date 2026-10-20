"""Blocks that fill the FFN slot of a transformer layer.

Both run in float32 even under autocast: torch's sparse matmul has no bfloat16
kernel, and running the dense control at a different precision would add a
difference between arms that is not wiring.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseFFN(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.fc = nn.Linear(d_model, hidden)
        self.proj = nn.Linear(hidden, d_model)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return self.proj(F.gelu(self.fc(x.float())))


class ConnectomeFFN(nn.Module):
    """A leaky recurrent layer whose only connections are a wiring's edges.

    Each token drives the input neurons, activity spreads over the wiring for
    steps-1 matrix applications with a fixed leak, and the readout neurons are
    projected back to d_model. Tokens never mix: this fills a position-wise
    FFN slot, so all sequence mixing stays in attention. The state is
    RMS-normalised over all neurons after every step, with no parameters, so
    activity scale cannot differ between wirings. The readout is deliberately
    not normalised again: that would cancel the last step's norm and leave
    many edges without a gradient at the start.
    """

    def __init__(self, d_model, wiring, steps, leak):
        super().__init__()
        if steps < 2:
            raise ValueError("steps must be >= 2, or the wiring is never used")
        self.n, self.steps, self.leak = wiring.n, steps, leak
        self.fingerprint = wiring.fingerprint()
        self.register_buffer("idx", torch.from_numpy(np.stack([wiring.post, wiring.pre])).long())
        self.register_buffer("inp", torch.from_numpy(wiring.inp).long())
        self.register_buffer("out", torch.from_numpy(wiring.out).long())
        indeg = np.bincount(wiring.post, minlength=wiring.n)
        # fan-in init, one rule for every arm: each neuron's summed input
        # starts at unit variance whatever its in-degree
        self.register_buffer("init_std", torch.from_numpy(1.0 / np.sqrt(indeg[wiring.post])).float())
        self.values = nn.Parameter(torch.zeros(wiring.nnz))
        self.bias = nn.Parameter(torch.zeros(wiring.n))
        self.w_in = nn.Linear(d_model, wiring.inp.size, bias=False)
        self.w_out = nn.Linear(wiring.out.size, d_model, bias=False)

    def state(self, x):
        """Neuron activity after the last step, shape (n, tokens), for x of shape (tokens, d)."""
        drive = x.new_zeros(self.n, x.shape[0]).index_add(0, self.inp, self.w_in(x).t())
        drive = drive + self.bias[:, None]
        # sorted, duplicate-free indices are guaranteed by make_wiring and tested
        a = torch.sparse_coo_tensor(self.idx, self.values, (self.n, self.n),
                                    is_coalesced=True, check_invariants=False)
        h = _norm(self.leak * F.gelu(drive))
        for _ in range(self.steps - 1):
            h = _norm((1 - self.leak) * h + self.leak * F.gelu(torch.sparse.mm(a, h) + drive))
        return h

    def forward(self, x):
        shape = x.shape
        with torch.autocast(x.device.type, enabled=False):
            h = self.state(x.reshape(-1, shape[-1]).float())
            y = self.w_out(h.index_select(0, self.out).t())
        return y.reshape(*shape[:-1], y.shape[-1])


def _norm(h):
    """RMS over the neuron axis (dim 0 of an (n, tokens) state), per token."""
    return h * torch.rsqrt(h.pow(2).mean(dim=0, keepdim=True) + 1e-6)


def connectome_param_count(d_model, wiring):
    return wiring.nnz + wiring.n + d_model * (wiring.inp.size + wiring.out.size)


def matched_hidden(d_model, n_params):
    """Hidden width at which a DenseFFN has n_params parameters, rounded."""
    return round((n_params - d_model) / (2 * d_model + 1))
