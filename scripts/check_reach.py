"""Gate: every arm must start with its wiring in use (PREREG.md section 3).

The shared brain takes one step per layer, so `steps` here is the layer count.
For every arm and seeds 0-2, at least 90% of edges must get a gradient on the
first step (edge_usage) and at least 90% of readout neurons must be reached by
input activity by the last layer (readout_reach). Uses the graph only.
"""
import json
import sys
from pathlib import Path

from clm.graphs import ARMS, edge_usage, make_wiring, readout_reach
from clm.model import ModelConfig

ROOT = Path(__file__).resolve().parent.parent
REGION = ROOT / "data" / "mb_alpn_783.npz"
STEPS, GATE = ModelConfig().n_layer, 0.9


def main():
    rows = {f"{a}-{s}": {"edge_usage": edge_usage(w, STEPS), "readout_reach": readout_reach(w, STEPS)}
            for a in ARMS for s in (0, 1, 2) for w in [make_wiring(REGION, a, s)]}
    ok = all(r["edge_usage"] >= GATE and r["readout_reach"] >= GATE for r in rows.values())
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "reach.json").write_text(
        json.dumps({"steps": STEPS, "gate": GATE, "pass": ok, "table": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    if not ok:
        sys.exit("an arm starts with under 90% of its wiring in use; stop and bring this table to Nate")


if __name__ == "__main__":
    main()
