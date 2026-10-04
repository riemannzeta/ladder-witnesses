#!/usr/bin/env bash
# Poll a remote queue every INTERVAL seconds, pulling results down; exit (and print why) when the
# queue finishes, a job fails, or the instance stays unreachable.
#   scripts/watch_queue.sh wave1 [interval_s]
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/aws_env.sh
Q=$1; INTERVAL=${2:-900}; miss=0; seen_fail=0
while true; do
  H=$(remote_host 2>/dev/null)
  if ssh $SSH_OPTS -o ConnectTimeout=10 "$REMOTE_USER@$H" true 2>/dev/null; then
    miss=0
    ./scripts/sync_down.sh >/dev/null 2>&1
    st=$(ssh $SSH_OPTS "$REMOTE_USER@$H" "cd $REMOTE_DIR/runs/queues/$Q 2>/dev/null && { [ -f _DONE ] && echo DONE; [ -f _FAILED ] && echo FAILED; grep -c ' rc=[1-9]' queue.log; grep -c ' end ' queue.log; } " 2>/dev/null)
    done_flag=$(echo "$st" | grep -E '^(DONE|FAILED)$' | head -1)
    nfail=$(echo "$st" | grep -E '^[0-9]+$' | sed -n 1p); nend=$(echo "$st" | grep -E '^[0-9]+$' | sed -n 2p)
    echo "[$(date -u +%H:%M)] $Q: finished-jobs=${nend:-?} failed-attempts=${nfail:-?} ${done_flag}"
    if [ -n "$done_flag" ]; then echo "QUEUE $Q $done_flag"; exit 0; fi
    if [ "${nfail:-0}" -gt "$seen_fail" ]; then echo "QUEUE $Q: a job attempt failed (see runs/queues/$Q/queue.log)"; exit 1; fi
  else
    miss=$((miss+1)); echo "[$(date -u +%H:%M)] instance unreachable ($miss)"
    if [ $miss -ge 3 ]; then echo "QUEUE $Q: instance unreachable (stopped?)"; exit 2; fi
  fi
  sleep "$INTERVAL"
done
