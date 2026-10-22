# FINISHED Fri 23 Oct 2026 08:53 NZDT. Instance 56974855 destroyed 08:56. Everything below is history.
All 15 runs complete; results, logs and checkpoints copied to `runs/` and verified by sha256 (46 files). Analysis: `runs/analysis.json`. Pod commit 62e39e0 merged here. Run cost about US$23.0 over 43 h; credit left US$6.79.

# Live production run (started Wed 21 Oct 2026, 13:53 NZDT)

Vast instance **56974855** (2x RTX 3090, $0.5385/h incl. 40 GB disk, on-demand), label `clm-stage1`.
Credit at launch: US$24.82. Expected finish about 37 h later (Fri 23 Oct, about 03:00 NZDT).
Hard stop at 43 h (Fri about 09:00 NZDT, about US$23.2): stop and ask Nate before spending more.
**Updated Thu 22 Oct 18:39 NZDT:** Nate topped up US$5 (credit US$14.48). Forecast finish Fri 23 Oct about 08:20 NZDT. New ask-first line: Fri 14:00 NZDT (about US$4 credit left). Credit runs out about Fri 21:30 NZDT.

- SSH: `ssh -i ~/.config/vastai/id_ed25519 -p 29310 root@ssh4.vast.ai`
- Run: tmux session `stage1`, `scripts/run_matrix.py --gpus 2`, log `runs/matrix.log` (status only; train.py prints no losses).
- Pod commit `62e39e0` holds PREREG.md, configs/stage1.json, results/speed.json. PREREG sha256 `8a0ad163…296d`. Laptop copy of PREREG: `/tmp/prereg/PREREG.md` (copy into the repo when pulling results back).
- Measured on this pod with both GPUs busy: 1.57 s/step connectome, 0.88 s/step dense (about 11% slower than 1x3090).
- No-peeking rule: do not read `loss` fields in `runs/*/log.jsonl` before all 15 runs finish. `wc -l` is fine (one line per step, 12,207 per run).

## When all 15 runs are complete
1. `.venv/bin/python -m clm.analyze --runs runs --config configs/stage1.json` on the pod.
2. Copy back `runs/*/results.json`, `runs/*/log.jsonl`, `runs/analysis.json`, `PREREG.md`, `configs/stage1.json`, `results/speed.json`.
3. `vastai destroy instance 56974855` (needs `yes |` for the prompt). Confirm `vastai show instances` is empty.
