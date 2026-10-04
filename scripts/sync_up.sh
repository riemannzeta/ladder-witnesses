#!/usr/bin/env bash
# Push code + configs (not results/runs) to the instance.
set -euo pipefail
source "$(dirname "$0")/aws_env.sh"
H=$(remote_host)
rsync -az --delete -e "ssh $SSH_OPTS" --exclude .venv --exclude runs --exclude results \
  --exclude 'blog/node_modules' --exclude .git --exclude .env ./ "$REMOTE_USER@$H:$REMOTE_DIR/"
echo "synced to $H:$REMOTE_DIR"
