"""Write configs/stage1.json from the reach gate and the speed test."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    reach_file, speed_file = ROOT / "results" / "reach.json", ROOT / "results" / "speed.json"
    reach = json.loads(reach_file.read_text())
    if not reach["pass"]:
        raise SystemExit("reach gate failed; nothing to freeze")
    speed = json.loads(speed_file.read_text())
    if not speed["fits"]:
        raise SystemExit(f"speed.json projects {speed['total_hours']:.0f} h, over the "
                         f"{speed['budget_hours']:.0f} h budget; nothing to freeze")
    # one micro-batch for every arm, the smaller of the two that fit, so data
    # order and accumulation are identical across arms
    micro = min(speed["arms"][a]["micro"] for a in ("real", "dense"))
    config = {
        "model": {"vocab": 8192, "ctx": 256, "d_model": 256, "n_layer": 6, "n_head": 4,
                  "leak": 0.5, "k_io": 256, "norm": True},
        "train": {"micro": micro},
        "provenance": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (reach_file, speed_file)},
    }
    (ROOT / "configs" / "stage1.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
