#!/usr/bin/env bash
# Stage 2A wrapper (Nate approved 2026-11-02: whole GPU, services stopped, restarted after;
# 2026-11-03: proceed through the series without stopping between tests). Launch with:
#   systemd-run --user --unit=stage2a --working-directory=/home/nate/Projects/connectome-lm \
#     -p MemoryMax=16G /home/nate/Projects/connectome-lm/scripts/stage2a_night.sh
# Only services that were running when the job started are stopped and restarted.
set -euo pipefail
cd /home/nate/Projects/connectome-lm
mkdir -p runs2
RUNNING=""
for s in embeddinggemma.service heretic-harness.service; do
  systemctl --user is-active --quiet "$s" && RUNNING="$RUNNING $s"
done
restart() { [ -z "$RUNNING" ] || systemctl --user start $RUNNING || echo "WARNING: could not restart$RUNNING" >&2; }
trap restart EXIT
[ -z "$RUNNING" ] || systemctl --user stop $RUNNING
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader
.venv/bin/python scripts/stage2a_driver.py 2>&1 | tee -a runs2/driver_2a.log
