#!/usr/bin/env bash
# Run a command on the instance:  scripts/remote.sh 'cd ladder-witnesses && nvidia-smi'
set -euo pipefail
source "$(dirname "$0")/aws_env.sh"
ssh $SSH_OPTS "$REMOTE_USER@$(remote_host)" "$@"
