#!/bin/bash
# Phase 2: Compute-quality trade-off curves.
# Sweep num_samples ∈ {32, 64, 128, 256, 512} for baseline MPPI vs Oracle PRISM.
# Plot SR vs K to show PRISM achieves baseline-quality SR at lower compute.
#
# Usage:  bash run_phase2.sh <task1> [task2 ...]
#         bash run_phase2.sh pusht tworooms

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/logs/phase2.log"
RESULTS_FILE="${SCRIPT_DIR}/logs/phase2_results.tsv"
mkdir -p "$(dirname "$LOG_FILE")"

# Initialize results TSV with header
if [[ ! -f "$RESULTS_FILE" ]]; then
    echo -e "task\tmethod\tnum_samples\tsr\twall_s" > "$RESULTS_FILE"
fi

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

if [[ -z "${STABLEWM_HOME:-}" ]]; then
    export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
fi

source "$SCRIPT_DIR/.venv/bin/activate"

K_VALUES=(32 64 128 256 512)

run_one() {
    local task="$1" method="$2" K="$3"
    local cfg_name="$task"
    if [[ "$task" == "tworooms" ]]; then cfg_name="tworoom"; fi

    local stdout_log="$SCRIPT_DIR/logs/phase2_${task}_${method}_K${K}.stdout"
    log "  $task | $method | K=$K …"
    local t_start=$(date +%s)

    if [[ "$method" == "baseline" ]]; then
        timeout 1800 python "$SCRIPT_DIR/eval.py" \
            --config-name="$cfg_name" \
            policy="$task/lewm" solver=mppi \
            solver.num_samples=$K solver.topk=$K \
            > "$stdout_log" 2>&1
    elif [[ "$method" == "oracle" ]]; then
        timeout 1800 python "$SCRIPT_DIR/eval_oracle.py" \
            --config-name="$cfg_name" \
            policy="$task/lewm" solver=mppi \
            +oracle.sigma=0.1 \
            solver.num_samples=$K solver.topk=$K \
            > "$stdout_log" 2>&1
    fi
    local rc=$?
    local t_end=$(date +%s)
    local sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$stdout_log" | head -1 | grep -oE "[0-9]+\.[0-9]+")

    if [[ -z "$sr" ]]; then
        log "    ✗ no SR (rc=$rc, see $stdout_log)"
        echo -e "$task\t$method\t$K\tNA\t$((t_end - t_start))" >> "$RESULTS_FILE"
    else
        log "    SR=$sr  wall=$((t_end - t_start))s"
        echo -e "$task\t$method\t$K\t$sr\t$((t_end - t_start))" >> "$RESULTS_FILE"
    fi
}

# ── Main ──────────────────────────────────────────────────────────
TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "Usage: $0 <task1> [task2 ...]"
    exit 1
fi

log "##################################################"
log "PHASE 2 START — tasks: ${TASKS[*]} | K values: ${K_VALUES[*]}"
log "##################################################"

for task in "${TASKS[@]}"; do
    for K in "${K_VALUES[@]}"; do
        run_one "$task" "baseline" "$K"
        run_one "$task" "oracle" "$K"
    done
done

log "##################################################"
log "PHASE 2 COMPLETE → results: $RESULTS_FILE"
log "##################################################"
