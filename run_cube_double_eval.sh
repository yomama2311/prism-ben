#!/bin/bash
# cube-double: cache features -> train prior head -> eval vanilla vs PRISM.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"; export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
LOG="$SCRIPT_DIR/logs/cube_double_eval.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[[ -d "$STABLEWM_HOME/jepa_cache/cube_double" ]] || {
  log "caching JEPA features ..."
  python jepa_cache.py --task cube_double --batch 128 >> "$LOG" 2>&1 || { log "CACHE FAILED"; exit 1; }
}
[[ -f "$SCRIPT_DIR/prior_head_cube_double.pt" ]] || {
  log "training prior head ..."
  python train_prior_head.py --task cube_double >> "$LOG" 2>&1 || { log "HEAD FAILED"; exit 1; }
}
R="$SCRIPT_DIR/logs/rebuttal_cube_double.tsv"
[[ -f "$R" ]] || echo -e "task\tmethod\tK\tseed\tsr" > "$R"
for K in 128 32; do
  for seed in 0 1 42; do
    for m in vanilla prism; do
      grep -qP "^cube_double\t${m}\t${K}\t${seed}\t" "$R" && continue
      out="$SCRIPT_DIR/logs/cd_${m}_K${K}_s${seed}.stdout"
      if [[ "$m" == vanilla ]]; then
        timeout 2400 python eval.py --config-name=cube_double policy=cube_double/lewm \
          solver=mppi solver.num_samples=$K solver.topk=$K eval.num_eval=50 seed=$seed > "$out" 2>&1
      else
        timeout 2400 python eval_prism_head.py --config-name=cube_double policy=cube_double/lewm \
          solver=mppi solver.num_samples=$K +head.injection_mode=pog +head.sigma_scale=1.0 \
          +head.ckpt="$SCRIPT_DIR/prior_head_cube_double.pt" eval.num_eval=50 seed=$seed > "$out" 2>&1
      fi
      sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
      echo -e "cube_double\t$m\t$K\t$seed\t${sr:-NA}" >> "$R"
      log "  $m K=$K seed=$seed sr=${sr:-NA}"
    done
  done
done
log "CUBE-DOUBLE EVAL COMPLETE"
touch "$SCRIPT_DIR/logs/cube_double_eval.DONE"
