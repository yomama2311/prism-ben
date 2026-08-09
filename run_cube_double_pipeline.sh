#!/bin/bash
# cube-double end-to-end: collect (states + full-length episodes) -> train WM.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"; export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
CL="$SCRIPT_DIR/logs/collect_cube_double.log"; TL="$SCRIPT_DIR/logs/train_cube_double.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$TL"; }

echo "=== collection start $(date) ===" > "$CL"
python collect_cube_double.py --episodes 3000 --num-envs 50 --name cube_double_expert >> "$CL" 2>&1 \
  || { log "COLLECT FAILED"; exit 1; }
touch "$SCRIPT_DIR/logs/collect_cube_double.DONE"
log "collection done -> WM training (20 epochs)"
python -u train.py data=ogb_double wandb.enabled=False \
  trainer.max_epochs=20 output_model_name=lewm_cube_double >> "$TL" 2>&1 \
  || { log "TRAIN FAILED"; exit 1; }
log "WM TRAINING COMPLETE"
touch "$SCRIPT_DIR/logs/train_cube_double.DONE"
