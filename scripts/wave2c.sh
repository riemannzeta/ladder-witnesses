#!/usr/bin/env bash
# Addendum 4: run queue wave2c (arm 3 folds 0-2 trained to the 240k cap), re-score the new
# checkpoints on the repaired frozen sets and on the balanced sets, then power the instance off
# (whatever the outcome, so billing always ends).
#   tmux new -d -s wave2c scripts/wave2c.sh
set -uo pipefail
cd "$(dirname "$0")/.."
UV=~/.local/bin/uv
mkdir -p runs/logs
$UV run python scripts/run_queue.py configs/queues/wave2c.yaml
$UV run python scripts/rescore_e2_samples.py --folds 0 1 2 --only comp25full \
  > runs/logs/rescore_comp25full.log 2>&1
$UV run python scripts/reeval_e2.py --mode samples --n-per-type 300 --folds 0 1 2 --only comp25full \
  > runs/logs/reeval_comp25full.log 2>&1
touch runs/logs/wave2c_post_DONE
sudo shutdown -h +2
