#!/bin/bash
# Rebuttal §5: suboptimal-data corruption study.
# Trains prior heads on ε-corrupted data, evals PRISM-MPPI at the paper's
# exact PRISM config (K=128, topk=30 default, pog, s=1.0, seeds {0,1,42}).
# Usage: bash run_rebuttal_corruption.sh <task>       (pusht | cube)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:?usage: run_rebuttal_corruption.sh <task>}"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
RESULTS="$SCRIPT_DIR/logs/rebuttal_corruption_results.tsv"
LOG="$SCRIPT_DIR/logs/rebuttal_corruption_${TASK}.log"
[[ -f "$RESULTS" ]] || echo -e "task\teps\tseed\tsr\twall_s" > "$RESULTS"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for EPS in 0.25 0.50 0.75 1.00; do
  TAG=$(python3 -c "print(int(float('$EPS')*100))")
  HEAD="$SCRIPT_DIR/prior_head_${TASK}_corrupt${TAG}.pt"
  if [[ ! -f "$HEAD" ]]; then
    log "training $TASK corrupt=$EPS -> $HEAD"
    python "$SCRIPT_DIR/train_prior_head.py" --task "$TASK" --seed 42 \
      --corrupt-ep-frac "$EPS" --force-save --out "$HEAD" >> "$LOG" 2>&1 \
      || { log "TRAIN FAIL eps=$EPS"; continue; }
  fi
  for seed in 0 1 42; do
    if grep -qP "^${TASK}\t${TAG}\t${seed}\t[0-9.]+\t" "$RESULTS"; then
      log "  skip (done): eps=$EPS seed=$seed"; continue
    fi
    out="$SCRIPT_DIR/logs/corrupt_${TASK}_e${TAG}_s${seed}.stdout"
    t0=$(date +%s)
    timeout 1800 python "$SCRIPT_DIR/eval_prism_head.py" --config-name="$TASK" \
      policy="$TASK/lewm" solver=mppi solver.num_samples=128 \
      +head.injection_mode=pog +head.sigma_scale=1.0 \
      +head.ckpt="$HEAD" \
      eval.num_eval=50 seed=$seed > "$out" 2>&1
    sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
    t1=$(date +%s); [[ -z "$sr" ]] && sr="NA"
    echo -e "$TASK\t$TAG\t$seed\t$sr\t$((t1-t0))" >> "$RESULTS"
    log "  eps=$EPS seed=$seed sr=$sr wall=$((t1-t0))s"
  done
done
log "CORRUPTION STUDY ($TASK) COMPLETE"
touch "$SCRIPT_DIR/logs/rebuttal_corruption_${TASK}.DONE"
