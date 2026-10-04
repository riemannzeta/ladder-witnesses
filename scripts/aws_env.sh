# shared settings for the AWS helper scripts
INSTANCE_ID=i-REDACTED
REGION=us-west-2
KEY=~/ml-dev.pem
REMOTE_USER=ubuntu
REMOTE_DIR=ladder-witnesses
TAILSCALE_IP=REDACTED-TAILSCALE-IP
remote_host() {
  # prefer Tailscale if reachable, else the current public IP
  if nc -z -G 3 "$TAILSCALE_IP" 22 2>/dev/null; then echo "$TAILSCALE_IP"; return; fi
  aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
