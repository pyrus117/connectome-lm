#!/usr/bin/env bash
# Overnight wrapper for Stage 2.0 (Nate approved 2026-11-02: whole GPU, services
# stopped for the night). Launch with:
#   systemd-run --user --unit=stage2-0 --working-directory=/home/nate/Projects/connectome-lm \
#     /home/nate/Projects/connectome-lm/scripts/stage2_0_night.sh
set -euo pipefail
cd /home/nate/Projects/connectome-lm
mkdir -p runs2
SERVICES="embeddinggemma.service heretic-harness.service"
restart() { systemctl --user start $SERVICES || echo "WARNING: could not restart $SERVICES" >&2; }
trap restart EXIT
systemctl --user stop $SERVICES
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader
.venv/bin/python scripts/stage2_0_driver.py 2>&1 | tee -a runs2/driver.log
