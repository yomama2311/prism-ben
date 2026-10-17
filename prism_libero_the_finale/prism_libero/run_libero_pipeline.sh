#!/usr/bin/env bash
# Full PRISM-JEPA pipeline on LIBERO.  Run from the prism-jepa repo root:
#     bash prism_libero/run_libero_pipeline.sh libero_spatial /path/to/LIBERO/libero/datasets
# Steps can be skipped with  SKIP="1 2"  (e.g. SKIP="1 2 3 4" to only run the eval).
set -euo pipefail

SUITE="${1:-libero_spatial}"                 # libero_spatial | libero_object | libero_goal | libero_10 | libero_90
DATA_ROOT="${2:?path to LIBERO/libero/datasets}"
IMG_SIZE="${IMG_SIZE:-224}"                  # 126 = 9x9 patches, ~3x faster; must be a multiple of 14
EPOCHS_JEPA="${EPOCHS_JEPA:-100}"
CHUNK_STEPS="${CHUNK_STEPS:-5}"              # C: plan-steps per chunk (5 x 5 env steps = 25 env steps)
MAX_CHUNKS="${MAX_CHUNKS:-4}"                # K: chunks per training window (4 x 25 = 100 env steps)
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
SKIP="${SKIP:-}"
RUN="libero_${SUITE}"
H5="$STABLEWM_HOME/libero/${SUITE}.h5"
JEPA_CKPT="$STABLEWM_HOME/${RUN}/lewm_epoch_${EPOCHS_JEPA}_object.ckpt"
HEAD="prior_head_${SUITE}_chunked.pt"
step() { [[ " $SKIP " != *" $1 "* ]]; }

# 1. LIBERO hdf5 -> stable-worldmodel hdf5 -------------------------------------------------
if step 1; then
  python prism_libero/convert_libero_to_swm.py --libero-dir "$DATA_ROOT/$SUITE" --out "$H5"
fi

# 2. train the LeWM JEPA world model (frozen afterwards) --------------------------------------
if step 2; then
  cp prism_libero/config/train/data/libero.yaml config/train/data/libero.yaml
  python train.py data=libero data.dataset.name="libero/${SUITE}" \
      img_size="$IMG_SIZE" trainer.max_epochs="$EPOCHS_JEPA" \
      output_model_name=lewm subdir="$RUN" wandb.enabled=False
fi

# 3. cache latents (features, goals, per-task goal bank, action stats) ------------------------
if step 3; then
  python prism_libero/cache_features_libero.py --h5 "$H5" --jepa-ckpt "$JEPA_CKPT" \
      --name "$SUITE" --img-size "$IMG_SIZE"
fi

# 4. train the chunked prior head (+ flat ablation head) ---------------------------------------
if step 4; then
  python prism_libero/train_chunk_prior_head.py --h5 "$H5" --name "$SUITE" --arch chunked \
      --chunk-steps "$CHUNK_STEPS" --max-chunks "$MAX_CHUNKS" --out "$HEAD"
  python prism_libero/train_chunk_prior_head.py --h5 "$H5" --name "$SUITE" --arch flat \
      --chunk-steps "$CHUNK_STEPS" --max-chunks "$MAX_CHUNKS" --out "prior_head_${SUITE}_flat.pt"
fi

# 5. closed-loop LIBERO evaluation (needs the LIBERO package + GPU) ----------------------------
if step 5; then
  COMMON=(--suite "$SUITE" --h5 "$H5" --cache-name "$SUITE" --jepa-ckpt "$JEPA_CKPT" --img-size "$IMG_SIZE" --episodes 10)
  python prism_libero/eval_libero_prism.py "${COMMON[@]}" --head "$HEAD" --mode pog  --goal-mode subgoal --out res_pog_subgoal.json
  python prism_libero/eval_libero_prism.py "${COMMON[@]}" --head "$HEAD" --mode none --goal-mode subgoal --out res_vanilla_subgoal.json
  python prism_libero/eval_libero_prism.py "${COMMON[@]}" --head "$HEAD" --mode pog  --goal-mode final   --out res_pog_final.json
  python prism_libero/eval_libero_prism.py "${COMMON[@]}" --head "$HEAD" --mode bc_only                    --out res_bc_only.json
  python prism_libero/eval_libero_prism.py "${COMMON[@]}" --head "prior_head_${SUITE}_flat.pt" --mode pog --goal-mode final --out res_flat_pog.json
fi
