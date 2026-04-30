#!/bin/bash
# Phase 1: Validate generality of PRISM-LeWM across cube/reacher/tworooms.
#
# Steps:
#   1. Decompress dataset .h5.zst → .h5 if needed
#   2. Convert HF checkpoint → _object.ckpt
#   3. Run baseline MPPI (50 ep) per task
#   4. Run Oracle PRISM-MPPI σ=0.1 (50 ep) per task
#
# Usage:  bash run_phase1.sh <task1> [task2 ...]
#         bash run_phase1.sh cube reacher tworooms

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/logs/phase1.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

if [[ -z "${STABLEWM_HOME:-}" ]]; then
    export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
fi

source "$SCRIPT_DIR/.venv/bin/activate"

run_task() {
    local task="$1"
    local cfg_name="$task"
    # tworooms uses "tworoom" config name
    if [[ "$task" == "tworooms" ]]; then
        cfg_name="tworoom"
    fi

    log "================================="
    log "TASK: $task (config: $cfg_name)"
    log "================================="

    # ── Step 1: decompress dataset if needed ──────────────────────
    # Each task has its own dataset_name path expected by eval config:
    #   pusht    → pusht_expert_train.h5
    #   cube     → ogbench/cube_single_expert.h5  (extracted from cube_single_expert.tar.zst)
    #   reacher  → dmc/reacher_random.h5  (extracted from reacher.tar.zst → reacher.h5 → rename)
    #   tworooms → tworoom.h5  (extracted from tworoom.tar.zst)
    local data_dir="$STABLEWM_HOME/hf_${task}_data"
    local target_h5
    case "$task" in
        pusht)    target_h5="$STABLEWM_HOME/pusht_expert_train.h5" ;;
        cube)     target_h5="$STABLEWM_HOME/ogbench/cube_single_expert.h5" ;;
        reacher)  target_h5="$STABLEWM_HOME/dmc/reacher_random.h5" ;;
        tworooms) target_h5="$STABLEWM_HOME/tworoom.h5" ;;
        *) log "  ✗ Unknown task '$task'"; return 1 ;;
    esac

    if [[ -f "$target_h5" ]]; then
        log "  Dataset already extracted: $target_h5"
    else
        # Find the tar.zst archive
        local archive
        archive=$(ls "$data_dir"/*.tar.zst 2>/dev/null | head -1)
        if [[ -z "$archive" ]]; then
            # Maybe single-file zst (pusht style)
            archive=$(ls "$data_dir"/*.h5.zst 2>/dev/null | head -1)
            if [[ -n "$archive" ]]; then
                log "  Decompressing single-file zst $archive …"
                mkdir -p "$(dirname "$target_h5")"
                zstd -d "$archive" -o "$target_h5" 2>&1 | tail -1 | tee -a "$LOG_FILE"
            else
                log "  ✗ No tar.zst or h5.zst found in $data_dir"
                return 1
            fi
        else
            log "  Extracting $archive …"
            mkdir -p "$(dirname "$target_h5")"
            local stage_dir="$STABLEWM_HOME/_stage_$task"
            mkdir -p "$stage_dir"
            tar --zstd -xf "$archive" -C "$stage_dir" 2>&1 | tail -3 | tee -a "$LOG_FILE"
            # Move/rename the extracted h5 to the target path
            local extracted_h5
            extracted_h5=$(find "$stage_dir" -maxdepth 2 -name "*.h5" | head -1)
            if [[ -z "$extracted_h5" ]]; then
                log "  ✗ No .h5 found in extracted tar"
                rm -rf "$stage_dir"
                return 1
            fi
            mv "$extracted_h5" "$target_h5"
            rm -rf "$stage_dir"
            log "  ✓ Dataset placed at $target_h5"
        fi
    fi

    # ── Step 2: convert checkpoint if needed ──────────────────────
    local ckpt="$STABLEWM_HOME/$task/lewm_object.ckpt"
    if [[ ! -f "$ckpt" ]]; then
        log "  Converting HF checkpoint …"
        python "$SCRIPT_DIR/convert_hf_checkpoint.py" "$task" 2>&1 | tail -5 | tee -a "$LOG_FILE"
    else
        log "  Checkpoint already exists: $ckpt"
    fi

    # ── Step 3: baseline MPPI ─────────────────────────────────────
    log "  Running baseline MPPI (50 ep) …"
    local t_start=$(date +%s)
    local stdout_log="$SCRIPT_DIR/logs/${task}_baseline.stdout"
    timeout 1800 python "$SCRIPT_DIR/eval.py" \
        --config-name="$cfg_name" \
        policy="$task/lewm" solver=mppi > "$stdout_log" 2>&1
    local rc=$?
    local t_end=$(date +%s)
    local sr_baseline=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$stdout_log" | tail -1)
    log "  Baseline: ${sr_baseline:-(no SR found, rc=$rc, see $stdout_log)} (wall: $((t_end - t_start))s)"

    # ── Step 4: oracle PRISM-MPPI ─────────────────────────────────
    log "  Running Oracle PRISM-MPPI σ=0.1 (50 ep) …"
    t_start=$(date +%s)
    stdout_log="$SCRIPT_DIR/logs/${task}_oracle.stdout"
    timeout 1800 python "$SCRIPT_DIR/eval_oracle.py" \
        --config-name="$cfg_name" \
        policy="$task/lewm" solver=mppi \
        +oracle.sigma=0.1 > "$stdout_log" 2>&1
    rc=$?
    t_end=$(date +%s)
    local sr_oracle=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$stdout_log" | tail -1)
    log "  Oracle PRISM: ${sr_oracle:-(no SR found, rc=$rc, see $stdout_log)} (wall: $((t_end - t_start))s)"

    log ""
}

# ── Main ──────────────────────────────────────────────────────────
TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "Usage: $0 <task1> [task2 ...]   e.g. $0 cube reacher tworooms"
    exit 1
fi

log "##################################################"
log "PHASE 1 START — tasks: ${TASKS[*]}"
log "##################################################"

for task in "${TASKS[@]}"; do
    run_task "$task" || log "  ✗ TASK $task FAILED"
done

log "##################################################"
log "PHASE 1 COMPLETE"
log "##################################################"
