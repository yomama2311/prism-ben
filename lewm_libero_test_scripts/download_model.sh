#!/usr/bin/env bash
set -euo pipefail
REPO="QinJialong/lewm_libero_v2_200"
DEST="${1:-/workspace/prism-ben/weights/lewm_libero_v2_200}"
mkdir -p "$DEST"
python -m pip install -U huggingface_hub
hf download "$REPO" config.yaml --local-dir "$DEST"
hf download "$REPO" last.ckpt --local-dir "$DEST"
if [[ "${DOWNLOAD_EPOCH58:-0}" == "1" ]]; then
  hf download "$REPO" 'lewm-epoch=epoch=058.ckpt' --local-dir "$DEST"
fi
ls -lh "$DEST/config.yaml" "$DEST/last.ckpt"
echo "Saved files to $DEST"
