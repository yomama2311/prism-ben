"""
Encode each task's goal description with CLIP-ViT-L/14 text encoder once.
The prompt is fixed per task, so the result is a single 768-d vector per
task — broadcast at training time alongside the per-frame VLM features.

Output:
    .stable-wm/vlm_cache/clip_text/<task>.npy   # (768,) fp16
    .stable-wm/vlm_cache/clip_text/meta.json    # model_id + prompts + dim

Run:  python clip_text_cache.py
Runs CPU-only (~1 s for all four tasks) so it doesn't fight the running
Qwen-VL cache for GPU.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


CLIP_ID = "openai/clip-vit-large-patch14"

# Keep these in sync with TASKS in vlm_cache.py.
PROMPTS = {
    "cube":    "Move the red cube to the green target position.",
    "pusht":   "Push the gray T-shaped block onto the red T-shaped target.",
    "reacher": "Move the red dot to the green target.",
    "tworoom": "Navigate from the current position to the green target.",
}


def main():
    stable_home = Path(os.environ.get(
        "STABLEWM_HOME", "/home/yuhai/project/prism-jepa/.stable-wm"
    ))
    out_dir = stable_home / "vlm_cache" / "clip_text"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[clip-text] loading {CLIP_ID} (CPU)…")
    model = CLIPModel.from_pretrained(CLIP_ID)
    processor = CLIPProcessor.from_pretrained(CLIP_ID)
    model.eval()
    text_dim = model.config.projection_dim
    print(f"[clip-text] projection_dim={text_dim}")

    embeddings = {}
    for task, prompt in PROMPTS.items():
        inputs = processor(text=[prompt], return_tensors="pt",
                           padding=True, truncation=True)
        with torch.no_grad():
            feat = model.get_text_features(**inputs)  # (1, text_dim)
        feat_np = feat[0].to(torch.float16).cpu().numpy()
        np.save(out_dir / f"{task}.npy", feat_np)
        embeddings[task] = {
            "prompt": prompt,
            "shape": list(feat_np.shape),
            "L2_norm": float(np.linalg.norm(feat_np.astype(np.float32))),
        }
        print(f"  {task:>8s}  prompt={prompt!r}  L2={embeddings[task]['L2_norm']:.3f}")

    meta = {
        "model_id": CLIP_ID,
        "projection_dim": int(text_dim),
        "dtype": "float16",
        "tasks": embeddings,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[clip-text] wrote {len(PROMPTS)} embeddings + meta to {out_dir}")


if __name__ == "__main__":
    main()
