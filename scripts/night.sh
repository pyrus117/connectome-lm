#!/usr/bin/env bash
# Run one Stage 2 driver with the GPU to itself (Nate approved 2026-11-02: whole GPU, services
# stopped, restarted after; 2026-11-03: proceed through the series without stopping between tests).
#   systemd-run --user --unit=NAME -p MemoryMax=16G /path/to/scripts/night.sh stage2c_driver.py [wait-for-unit]
# Works from whichever checkout it lives in. Optional second argument: a systemd user unit to wait for first
# (the GPU is then free for this job). Only services that were running when the job started are stopped
# and restarted.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
DRIVER="$1"
mkdir -p runs2
if [ -n "${2:-}" ]; then
  while systemctl --user is-active --quiet "$2"; do sleep 30; done
fi
RUNNING=""
for s in embeddinggemma.service heretic-harness.service; do
  systemctl --user is-active --quiet "$s" && RUNNING="$RUNNING $s"
done
restart() { [ -z "$RUNNING" ] || systemctl --user start $RUNNING || echo "WARNING: could not restart$RUNNING" >&2; }
trap restart EXIT
[ -z "$RUNNING" ] || systemctl --user stop $RUNNING
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader
.venv/bin/python "scripts/$DRIVER" 2>&1 | tee -a "runs2/driver_${DRIVER%.py}.log"
