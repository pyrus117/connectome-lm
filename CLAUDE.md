# connectome-lm

Personal research project (not Q-Youth). Stage 1 tests whether a transformer whose layers share one recurrent brain wired like the fly mushroom body beats the same model with degree-preserving shuffled wiring.

**Read `HANDOVER.md` first.** It holds the design as built, the decisions and their evidence, what waits on Nate, the rented-GPU runbook and the PREREG draft.

## Rules for this repo

- Python: `/home/nate/Projects/connectome-lm/.venv/bin/python` (system 3.14 with CUDA torch). Tests: `.venv/bin/python -m pytest tests`.
- Pre-registration discipline: no validation loss of a primary arm (`real`, `shuf`) is read before all 15 production runs finish. Design pilots run on the `er` arm only.
- Never change code, config or data after `PREREG.md` is committed except to fix a crash, logged in `notes/deviations.md`.
- After any change to `src/`, run the suite and `scripts/mutation_audit.py`; both must pass. A test that cannot fail on a planted bug is rewritten, not kept.
- Long jobs: `systemd-run --user` on the laptop (the harness kills background jobs after 30 min). On the laptop GPU, cap memory (`--mem-fraction 0.75`) and start at micro 4: the internal panel link failed once under a full-VRAM test.
- GPU rental spends Nate's money: confirm the budget with him first.
- `notes/` records every measurement with its conditions. Add to it; do not overwrite earlier numbers. Mark corrections as corrections.
