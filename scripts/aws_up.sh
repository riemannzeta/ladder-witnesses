#!/usr/bin/env bash
# Start the GPU instance and wait until SSH answers.
set -euo pipefail
source "$(dirname "$0")/aws_env.sh"
aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
for i in $(seq 1 60); do
  H=$(remote_host); ssh $SSH_OPTS -o ConnectTimeout=5 "$REMOTE_USER@$H" true 2>/dev/null && break; sleep 5
done
echo "instance up at $(remote_host)"
