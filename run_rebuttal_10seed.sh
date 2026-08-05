#!/bin/bash
# Rebuttal 10-seed re-run (CoRL 2026 sub #1157, rebuttal §9).
#
# Re-runs the paper's three headline MPPI rows with seeds {0..9} (vs. {0,1,42}
# in the paper), reproducing each row's EXACT original config:
#   vanilla_mppi    : eval.py,            topk=K   (v3 convention, canonical MPPI)
#   mppi_warm_start : eval_prism_head.py, topk=K, injection=warm_start, s=2.0 (v3)
#   prism_mppi      : eval_prism_head.py, topk=30 (yaml default), injection=pog, s=1.0 (s1 rerun)
#
# Seeds 0,1 duplicate existing runs on purpose — deterministic eval means they
# must reproduce v3/s1 numbers exactly (canary for config drift).
#
# 2 tasks x 3 K x 3 methods x 10 seeds = 180 cells x ~68 s ~= 3.4 h.
# Output: logs/rebuttal_10seed_results.tsv
# Watcher: completion marker logs/rebuttal_10seed.DONE (PID-independent).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$SCRIPT_DIR/logs/rebuttal_10seed_results.tsv"
LOG="$SCRIPT_DIR/logs/rebuttal_10seed.log"
mkdir -p "$SCRIPT_DIR/logs"
[[ -f "$RESULTS" ]] || echo -e "task\tmethod\tK\tsr\tseed\twall_s" > "$RESULTS"
export STABLEWM_HOME="${STABLEWM_HOME:-$SCRIPT_DIR/.stable-wm}"
source "$SCRIPT_DIR/.venv/bin/activate"

K_VALUES=(32 64 128)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
NUM_EVAL=50
TIMEOUT=1800

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_cell(){
  local task="$1" method="$2" K="$3" seed="$4"
  # resume support: skip cells already in the tsv
  if grep -qP "^${task}\t${method}\t${K}\t[0-9.]+\t${seed}\t" "$RESULTS"; then
    log "  skip (done): $task $method K=$K seed=$seed"; return 0
  fi
  local out="$SCRIPT_DIR/logs/r10_${method}_${task}_K${K}_s${seed}.stdout"
  local t0=$(date +%s)
  log "  $method | $task | K=$K | seed=$seed ..."
  case "$method" in
    vanilla_mppi)
      timeout $TIMEOUT python "$SCRIPT_DIR/eval.py" --config-name="$task" \
        policy="$task/lewm" solver=mppi \
        solver.num_samples=$K solver.topk=$K \
        eval.num_eval=$NUM_EVAL seed=$seed > "$out" 2>&1 ;;
    mppi_warm_start)
      timeout $TIMEOUT python "$SCRIPT_DIR/eval_prism_head.py" --config-name="$task" \
        policy="$task/lewm" solver=mppi \
        solver.num_samples=$K solver.topk=$K \
        +head.injection_mode=warm_start +head.sigma_scale=2.0 \
        +head.ckpt="$SCRIPT_DIR/prior_head_$task.pt" \
        eval.num_eval=$NUM_EVAL seed=$seed > "$out" 2>&1 ;;
    prism_mppi)
      timeout $TIMEOUT python "$SCRIPT_DIR/eval_prism_head.py" --config-name="$task" \
        policy="$task/lewm" solver=mppi \
        solver.num_samples=$K \
        +head.injection_mode=pog +head.sigma_scale=1.0 \
        +head.ckpt="$SCRIPT_DIR/prior_head_$task.pt" \
        eval.num_eval=$NUM_EVAL seed=$seed > "$out" 2>&1 ;;
  esac
  local sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
  local t1=$(date +%s)
  [[ -z "$sr" ]] && { sr="NA"; log "    !! NA — check $out"; }
  echo -e "$task\t$method\t$K\t$sr\t$seed\t$((t1-t0))" >> "$RESULTS"
  log "    sr=$sr  wall=$((t1-t0))s"
}

log "REBUTTAL-10SEED START  K=${K_VALUES[*]}  seeds=${SEEDS[*]}"
for task in pusht cube; do
  for method in vanilla_mppi mppi_warm_start prism_mppi; do
    for K in "${K_VALUES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run_cell "$task" "$method" "$K" "$seed"
      done
    done
  done
done
log "REBUTTAL-10SEED COMPLETE -> $RESULTS"
touch "$SCRIPT_DIR/logs/rebuttal_10seed.DONE"
