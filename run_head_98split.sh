#!/bin/bash
# Control: train the prior head on DP's 98% split (vs our 90%) to confirm the
# 8-point data asymmetry in DP's favour changes nothing.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STABLEWM_HOME="$SCRIPT_DIR/.stable-wm"
source "$SCRIPT_DIR/.venv/bin/activate"
LOG="$SCRIPT_DIR/logs/head_98split.log"
R="$SCRIPT_DIR/logs/rebuttal_head_98split.tsv"
[[ -f "$R" ]] || echo -e "task\tsplit\tseed\tsr" > "$R"
python - <<'PY' >> "$LOG" 2>&1
import re, pathlib
p = pathlib.Path('/home/yuhai/project/prism-jepa/train_prior_head.py')
s = p.read_text()
# pusht: 18685 episodes; DP keeps 98% for train -> 18311
assert 'train_ep_range=(0, 16817)' in s
s2 = s.replace('train_ep_range=(0, 16817),\n        val_ep_range=(16817, 18685),',
               'train_ep_range=(0, 18311),\n        val_ep_range=(18311, 18685),', 1)
pathlib.Path('/tmp/train_prior_head_98.py').write_text(s2)
print("[patch] wrote /tmp/train_prior_head_98.py with 98% pusht split")
PY
cp /tmp/train_prior_head_98.py "$SCRIPT_DIR/train_prior_head_98.py"
HEAD="$SCRIPT_DIR/prior_head_pusht_split98.pt"
[[ -f "$HEAD" ]] || python "$SCRIPT_DIR/train_prior_head_98.py" --task pusht --out "$HEAD" >> "$LOG" 2>&1
for seed in 0 1 42; do
  out="$SCRIPT_DIR/logs/head98_pusht_s${seed}.stdout"
  timeout 1800 python "$SCRIPT_DIR/eval_prism_head.py" --config-name=pusht \
    policy=pusht/lewm solver=mppi solver.num_samples=128 \
    +head.injection_mode=pog +head.sigma_scale=1.0 +head.ckpt="$HEAD" \
    eval.num_eval=50 seed=$seed > "$out" 2>&1
  sr=$(grep -oE "success_rate[^a-z]*[0-9]+\.[0-9]+" "$out" | head -1 | grep -oE "[0-9]+\.[0-9]+")
  echo -e "pusht\t98\t$seed\t${sr:-NA}" >> "$R"
  echo "[$(date '+%F %T')] 98%-split head seed=$seed sr=${sr:-NA}" | tee -a "$LOG"
done
touch "$SCRIPT_DIR/logs/head_98split.DONE"
