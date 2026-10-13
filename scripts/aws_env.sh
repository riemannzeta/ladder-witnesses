# shared settings for the AWS helper scripts
# Machine-specific values live in the untracked scripts/aws_env.local.sh, e.g.
#   INSTANCE_ID=i-0123456789abcdef0
#   TAILSCALE_IP=100.x.y.z        # optional; falls back to the instance's public IP
_aws_env_local="$(dirname "${BASH_SOURCE[0]:-$0}")/aws_env.local.sh"
[ -f "$_aws_env_local" ] && source "$_aws_env_local"
: "${INSTANCE_ID:?set INSTANCE_ID in scripts/aws_env.local.sh}"
REGION=${REGION:-us-west-2}
KEY=${KEY:-$HOME/ml-dev.pem}
REMOTE_USER=${REMOTE_USER:-ubuntu}
REMOTE_DIR=${REMOTE_DIR:-ladder-witnesses}
TAILSCALE_IP=${TAILSCALE_IP:-}
remote_host() {
  # prefer Tailscale if reachable, else the current public IP
  if [ -n "$TAILSCALE_IP" ] && nc -z -G 3 "$TAILSCALE_IP" 22 2>/dev/null; then echo "$TAILSCALE_IP"; return; fi
  aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
