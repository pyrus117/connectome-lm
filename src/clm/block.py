"""The shared brain: one recurrent network that every transformer layer uses.

For each token the brain's state starts empty. Each layer writes into the
input neurons, the brain takes one leaky step over its wiring, and the layer
reads the readout neurons back. The state carries from layer to layer within
a token; tokens never share state, so all mixing between tokens stays in
attention. The synapse strengths are shared by all layers; each layer has its
own write and read projections (they live in the Layer, see model.py).

Runs in float32 even under autocast: torch's sparse matmul has no bfloat16
kernel.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedBrain(nn.Module):
    def __init__(self, wiring, leak, norm=True):
        super().__init__()
        self.n, self.leak, self.norm = wiring.n, leak, norm
        self.k_in, self.k_out = int(wiring.inp.size), int(wiring.out.size)
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

    def step(self, h, write):
        """One leaky step. h: (n, tokens) state, or None before the first layer;
        write: (tokens, k_in) from a layer's write projection."""
        drive = write.new_zeros(self.n, write.shape[0]).index_add(0, self.inp, write.t())
        drive = drive + self.bias[:, None]
        if h is None:
            h = self.leak * F.gelu(drive)
        else:
            # sorted, duplicate-free indices are guaranteed by make_wiring and tested
            a = torch.sparse_coo_tensor(self.idx, self.values, (self.n, self.n),
                                        is_coalesced=True, check_invariants=False)
            h = (1 - self.leak) * h + self.leak * F.gelu(torch.sparse.mm(a, h) + drive)
        return _norm(h) if self.norm else h

    def read(self, h):
        """Readout neurons, shape (tokens, k_out)."""
        return h.index_select(0, self.out).t()


def _norm(h):
    """RMS over the neuron axis (dim 0 of an (n, tokens) state), per token."""
    return h * torch.rsqrt(h.pow(2).mean(dim=0, keepdim=True) + 1e-6)


def brain_param_count(wiring):
    return wiring.nnz + wiring.n
