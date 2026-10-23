#!/usr/bin/env bash
# Overnight pilot of the harder vocab tasks (notes/pilot_vocab2_rules.md).
# er-0 first, then er-2 as a second seed for a noise estimate. er-1 is skipped:
# it had a gradient spike in pretraining. Settings already done are skipped by
# pilot_vocab.py, so a rerun resumes.
set -euo pipefail
cd /home/nate/Projects/connectome-lm
P=".venv/bin/python scripts/pilot_vocab.py"
for RUN in runs/er-0 runs/er-2; do
    $P --run $RUN --task twins --pairs 1536 512 --lrs 3e-4 --steps 1500
    $P --run $RUN --task sequential --lists 4 --pairs 384 --lrs 3e-4 1e-3 --steps 300
    $P --run $RUN --task sequential --lists 4 --pairs 384 --lrs 3e-4 1e-3 --steps 600
    $P --run $RUN --task sequential --lists 8 --pairs 192 --lrs 1e-3 --steps 300
done
echo "pilot_vocab2 finished"
