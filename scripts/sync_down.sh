#!/usr/bin/env bash
# Pull results (metrics, preds, meta) and logs back from the instance.
set -euo pipefail
source "$(dirname "$0")/aws_env.sh"
H=$(remote_host)
rsync -az -e "ssh $SSH_OPTS" "$REMOTE_USER@$H:$REMOTE_DIR/results/" results/
rsync -az -e "ssh $SSH_OPTS" --include '*/' --include '*.log' --include 'benchmark.json' \
  --include '_DONE' --include '_FAILED' --exclude '*' "$REMOTE_USER@$H:$REMOTE_DIR/runs/" runs/
echo "pulled results from $H"
