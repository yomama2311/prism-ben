#!/bin/bash
# Rebuttal §3 / Table R1: DP-guided planning hybrids.
# 3 modes x 2 tasks x seeds {0,1,42}, K=128, N_dp=16, s=1.0.
# Solver config matches the paper's PRISM row (topk=30 yaml default).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
RESULTS="$SCRIPT_DIR/logs/rebuttal_hybrid_results.tsv"
LOG="$SCRIPT_DIR/logs/rebuttal_hybrid.log"
[[ -f "$RESULTS" ]] || echo -e "task\tmode\tseed\tsr\tdp_ms_mean\twall_s" > "$RESULTS"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for task in pusht cube; do
  for mode in best_of_n replace pog; do
    for seed in 0 1 42; do
      if grep -qP "^${task}\t${mode}\t${seed}\t[0-9.]+\t" "$RESULTS"; then
        log "skip (done): $task $mode seed=$seed"; continue
      fi
      out="$SCRIPT_DIR/logs/hybrid_${task}_${mode}_s${seed}.stdout"
      t0=$(date +%s)
      log "$task | $mode | seed=$seed ..."
      timeout 2400 python "$SCRIPT_DIR/eval_dp_prior.py" --config-name="$task" \
        policy="$task/lewm" solver=mppi solver.num_samples=128 \
        +dpprior.injection_mode=$mode \
        +dpprior.ckpt="$SCRIPT_DIR/dp_baseline_${task}.pt" \
        +dpprior.n_dp_samples=16 +dpprior.sigma_scale=1.0 \
        eval.num_eval=50 seed=$seed > "$out" 2>&1
      sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
      dpms=$(grep -oE "dp_sample_ms_per_replan: mean=[0-9.]+" "$out" | grep -oE "[0-9.]+$")
      t1=$(date +%s); [[ -z "$sr" ]] && { sr="NA"; log "  !! NA — check $out"; }
      echo -e "$task\t$mode\t$seed\t$sr\t${dpms:-NA}\t$((t1-t0))" >> "$RESULTS"
      log "  sr=$sr dp_ms=${dpms:-NA} wall=$((t1-t0))s"
    done
  done
done
log "HYBRID SWEEP COMPLETE -> $RESULTS"
touch "$SCRIPT_DIR/logs/rebuttal_hybrid.DONE"
