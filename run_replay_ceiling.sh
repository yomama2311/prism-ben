#!/bin/bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
R="$SCRIPT_DIR/logs/rebuttal_replay_ceiling.tsv"
[[ -f "$R" ]] || echo -e "task\tseed\tsr" > "$R"
for task in pusht cube; do
  for seed in 0 1 42; do
    out="$SCRIPT_DIR/logs/replay_${task}_s${seed}.stdout"
    timeout 1800 python "$SCRIPT_DIR/eval_replay_oracle.py" --config-name=$task \
      policy=$task/lewm solver=mppi eval.num_eval=50 seed=$seed > "$out" 2>&1
    sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
    echo -e "$task\t$seed\t${sr:-NA}" >> "$R"
    echo "[$(date '+%F %T')] $task seed=$seed sr=${sr:-NA}"
  done
done
touch "$SCRIPT_DIR/logs/replay_ceiling.DONE"
