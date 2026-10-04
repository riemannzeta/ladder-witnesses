#!/usr/bin/env bash
# Stop (not terminate) the GPU instance.
set -euo pipefail
source "$(dirname "$0")/aws_env.sh"
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "instance stopped"
