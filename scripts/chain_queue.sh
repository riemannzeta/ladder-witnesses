#!/usr/bin/env bash
# Run a follow-on queue on the instance after a running queue drains, without touching it:
#   scripts/chain_queue.sh <running-queue-name> <next-queue.yaml>
# The running queue (started with --shutdown) schedules `shutdown -h +2` right after writing its
# _DONE/_FAILED marker; this cancels that and runs the next queue, itself with --shutdown.
set -uo pipefail
cd "$(dirname "$0")/.."
prev="runs/queues/$1"
until [ -e "$prev/_DONE" ] || [ -e "$prev/_FAILED" ]; do sleep 30; done
echo "[$(date -u +%FT%TZ)] $1 drained; cancelling its shutdown"
for i in 1 2 3; do sleep 15; sudo shutdown -c 2>/dev/null; done
echo "[$(date -u +%FT%TZ)] starting $2"
exec ~/.local/bin/uv run python scripts/run_queue.py "$2" --shutdown
