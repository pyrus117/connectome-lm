"""Write configs/stage1.json from the reach gate and the layout rule's outputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    reach_file, speed_file = ROOT / "results" / "reach.json", ROOT / "results" / "speed.json"
    reach = json.loads(reach_file.read_text())
    if not reach["pass"]:
        raise SystemExit("reach gate failed; nothing to freeze")
    steps = reach["steps"]
    speed = json.loads(speed_file.read_text())
    layout = speed["chosen_layout"]
    if layout is None:
        raise SystemExit("speed.json has no layout within budget; nothing to freeze")
    micro = speed["layouts"][str(tuple(layout))]["real"]["micro"]
    config = {
        "model": {"vocab": 8192, "ctx": 256, "d_model": 256, "n_layer": 6, "n_head": 4,
                  "conn_layers": layout, "steps": steps, "leak": 0.5, "k_io": 256},
        "train": {"micro": micro},
        "provenance": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (reach_file, speed_file)},
    }
    (ROOT / "configs" / "stage1.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
