#!/usr/bin/env bash
# Run from root of prism-jepa. All outputs are isolated under STABLEWM_HOME.
# Example: STABLEWM_HOME=/workspace/stable-wm bash prism_libero/run_libero_pipeline.sh \
#          libero_spatial /workspace/FA26_RE/datasets
set -euo pipefail

SUITE="${1:-libero_spatial}"
DATA_ROOT="${2:-/workspace/FA26_RE/datasets}"
STABLEWM_HOME="${STABLEWM_HOME:-/workspace/stable-wm}"
export STABLEWM_HOME
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

SKIP="${SKIP:-}"                    # e.g. "1 2 3 4" to run only eval
IMG_SIZE="${IMG_SIZE:-224}"
EPOCHS_JEPA="${EPOCHS_JEPA:-50}"
EPOCHS_HEAD="${EPOCHS_HEAD:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
LEWM_PATIENCE="${LEWM_PATIENCE:-15}"
HEAD_PATIENCE="${HEAD_PATIENCE:-8}"
CHUNK_STEPS="${CHUNK_STEPS:-5}"
MAX_CHUNKS="${MAX_CHUNKS:-4}"
TRAIN_CAMERA="${TRAIN_CAMERA:-agentview_rgb}"
EVAL_TASKS="${EVAL_TASKS:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-3}"
EVAL_SAMPLES="${EVAL_SAMPLES:-32}"
EVAL_ITERATIONS="${EVAL_ITERATIONS:-3}"
EVAL_VIDEO="${EVAL_VIDEO:-1}"
SEED="${SEED:-42}"

H5="$STABLEWM_HOME/libero/${SUITE}_train.h5"
VAL_H5="$STABLEWM_HOME/libero/${SUITE}_val.h5"
RUN="$STABLEWM_HOME/libero_${SUITE}"
LEWM_CKPT="${LEWM_CKPT:-$RUN/best_object.ckpt}"
CACHE_NAME="$SUITE"
HEAD_DIR="$STABLEWM_HOME/prior_checkpoints/${SUITE}"
CHUNK_HEAD="$HEAD_DIR/chunked_best.pt"
FLAT_HEAD="$HEAD_DIR/flat_best.pt"
OUTPUT_DIR="$STABLEWM_HOME/evaluations/${SUITE}/$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$STABLEWM_HOME/logs/${SUITE}"
mkdir -p "$HEAD_DIR" "$OUTPUT_DIR" "$LOG_DIR" "$(dirname "$H5")"

step() { [[ " $SKIP " != *" $1 "* ]]; }
run_logged() {
  local label="$1"; shift
  echo "[PIPELINE] $label $*"
  "$@" 2>&1 | tee "$LOG_DIR/$label.log"
}

# 1: Convert and split by whole episode, avoiding train/validation clip leakage.
if step 1; then
  run_logged convert python -u prism_libero/convert_libero_to_swm.py \
    --libero-dir "$DATA_ROOT/$SUITE" --out "$H5" \
    --val-out "$VAL_H5" --val-frac 0.10 --seed "$SEED" \
    --camera "$TRAIN_CAMERA"
fi

# 2: Train same JEPA LeWM objective, with best/last and 10-epoch checkpoints.
if step 2; then
  export LIBERO_TRAIN_H5="$H5"
  export LIBERO_VAL_H5="$VAL_H5"
  export LIBERO_LEWM_RUN_DIR="$RUN"
  mkdir -p config/train/data
  cp prism_libero/config/train/data/libero.yaml config/train/data/libero_managed.yaml
  run_logged lewm python -u prism_libero/train_lewm_managed.py \
    data=libero_managed "trainer.max_epochs=$EPOCHS_JEPA" \
    "img_size=$IMG_SIZE" "subdir=libero_${SUITE}" \
    "output_model_name=lewm_${SUITE}" \
    "+checkpoint_every=$CHECKPOINT_EVERY" \
    "+early_stop_patience=$LEWM_PATIENCE" wandb.enabled=False
fi

# 3: Rebuild features using EXACTLY the chosen LeWM best checkpoint.
if step 3; then
  [[ -f "$LEWM_CKPT" ]] || { echo "Missing LeWM checkpoint: $LEWM_CKPT"; exit 1; }
  run_logged cache python -u prism_libero/cache_features_libero.py \
    --h5 "$H5" --jepa-ckpt "$LEWM_CKPT" --name "$CACHE_NAME" --img-size "$IMG_SIZE"
fi

# 4: Train both priors, always saving best.pt, last.pt, epoch_010.pt, ...
if step 4; then
  for arch in chunked flat; do
    out="$HEAD_DIR/${arch}_best.pt"
    run_logged "prior_$arch" python -u prism_libero/train_chunk_prior_head.py \
      --h5 "$H5" --name "$CACHE_NAME" --arch "$arch" \
      --chunk-steps "$CHUNK_STEPS" --max-chunks "$MAX_CHUNKS" \
      --epochs "$EPOCHS_HEAD" --checkpoint-every "$CHECKPOINT_EVERY" \
      --patience "$HEAD_PATIENCE" --window-policy full \
      --checkpoint-dir "$HEAD_DIR/$arch" --out "$out" --seed "$SEED" \
      --goal-swap-prob 0.0
  done
fi

# 5: Automatically TEST after training and record JSON, CSV, JSONL, optional videos.
if step 5; then
  [[ -f "$LEWM_CKPT" ]] || { echo "Missing LeWM checkpoint: $LEWM_CKPT"; exit 1; }
  [[ -f "$CHUNK_HEAD" && -f "$FLAT_HEAD" ]] || { echo "Missing prior heads under $HEAD_DIR"; exit 1; }
  common=(--suite "$SUITE" --h5 "$H5" --cache-name "$CACHE_NAME" \
          --jepa-ckpt "$LEWM_CKPT" --img-size "$IMG_SIZE" \
          --tasks "$EVAL_TASKS" --episodes "$EVAL_EPISODES" \
          --num-samples "$EVAL_SAMPLES" --n-steps "$EVAL_ITERATIONS" \
          --seed "$SEED")
  video=()
  [[ "$EVAL_VIDEO" == "1" ]] && video=(--video)
  run_logged eval_chunked_bc python -u prism_libero/eval_libero_prism.py \
     "${common[@]}" --head "$CHUNK_HEAD" --mode bc_only --goal-mode final \
     "${video[@]}" --out "$OUTPUT_DIR/chunked_bc.json"
  run_logged eval_chunked_final python -u prism_libero/eval_libero_prism.py \
     "${common[@]}" --head "$CHUNK_HEAD" --mode pog --goal-mode final \
     "${video[@]}" --out "$OUTPUT_DIR/chunked_pog_final.json"
  run_logged eval_chunked_subgoal python -u prism_libero/eval_libero_prism.py \
     "${common[@]}" --head "$CHUNK_HEAD" --mode pog --goal-mode subgoal \
     "${video[@]}" --out "$OUTPUT_DIR/chunked_pog_subgoal.json"
  run_logged eval_flat_bc python -u prism_libero/eval_libero_prism.py \
     "${common[@]}" --head "$FLAT_HEAD" --mode bc_only --goal-mode final \
     "${video[@]}" --out "$OUTPUT_DIR/flat_bc.json"
  run_logged eval_flat_final python -u prism_libero/eval_libero_prism.py \
     "${common[@]}" --head "$FLAT_HEAD" --mode pog --goal-mode final \
     "${video[@]}" --out "$OUTPUT_DIR/flat_pog_final.json"
  python -u prism_libero/summarize_eval.py --input-dir "$OUTPUT_DIR" \
    --out "$OUTPUT_DIR/compare.csv" | tee "$OUTPUT_DIR/compare.txt"
  echo "[PIPELINE] evaluation outputs: $OUTPUT_DIR"
fi

echo "[PIPELINE] FINISHED. LeWM=$LEWM_CKPT heads=$HEAD_DIR logs=$LOG_DIR"
