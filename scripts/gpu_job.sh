#!/usr/bin/env bash
# Whole-GPU job wrapper: stops the GPU services, runs one driver, restarts them on exit.
#   systemd-run --user --unit=<name> --working-directory=/home/nate/Projects/connectome-lm \
#     /home/nate/Projects/connectome-lm/scripts/gpu_job.sh scripts/<driver>.py runs2/<log>
set -euo pipefail
cd /home/nate/Projects/connectome-lm
DRIVER=$1 LOG=$2
mkdir -p "$(dirname "$LOG")"
SERVICES="embeddinggemma.service heretic-harness.service"
restart() { systemctl --user start $SERVICES || echo "WARNING: could not restart $SERVICES" >&2; }
trap restart EXIT
systemctl --user stop $SERVICES
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader
.venv/bin/python "$DRIVER" 2>&1 | tee -a "$LOG"
