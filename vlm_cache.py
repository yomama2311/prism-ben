"""
Cache Qwen2.5-VL-3B mean-pooled hidden states for every frame in a LeWM
expert h5 dataset. Output schema (per task):

    .stable-wm/vlm_cache/<task>/
        features.npy        # (N_frames, 2048) fp16 mmap
        meta.json           # {model_id, prompt, hidden_dim, n_frames,
                            #   batch_size, dtype, sha256_of_prompt, ...}

Why mean-pool: a per-token cache for 2M frames at (95, 2048, bf16) would be
~780 GB per task. Mean-pool collapses to (2048,) fp16 = 4 KB/frame ≈ 8 GB
per task. The head consumes the pooled vector directly (3-layer MLP).

Run:
    python vlm_cache.py --task cube --dry-run     # 1 episode, ~30 s
    python vlm_cache.py --task cube                # full ~6 h
    python vlm_cache.py --task cube --batch 16     # tweak batching
"""
import argparse
import json
import os
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  — registers Blosc codec
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Per-task expert h5 path (relative to STABLEWM_HOME) and language goal.
TASKS = {
    "cube": {
        "h5": "ogbench/cube_single_expert.h5",
        "prompt": "Move the red cube to the green target position.",
    },
    "pusht": {
        "h5": "pusht_expert_train.h5",
        "prompt": "Push the gray T-shaped block onto the red T-shaped target.",
    },
    "reacher": {
        "h5": "dmc/reacher_random.h5",
        "prompt": "Move the red dot to the green target.",
    },
    "tworoom": {
        "h5": "tworoom.h5",
        "prompt": "Navigate from the current position to the green target.",
    },
}


def build_inputs(processor, images, prompt, device, image_only=False):
    """Build batched processor inputs from a list of PIL images.

    If image_only=True, omit the task description from the message — the user
    turn contains only the image. This is the "let CLIP handle the goal text
    separately" variant validated empirically in Newt-PRISM.
    """
    if image_only:
        msgs_batch = [
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                ],
            }]
            for img in images
        ]
    else:
        msgs_batch = [
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }]
            for img in images
        ]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in msgs_batch
    ]
    image_inputs, video_inputs = process_vision_info(msgs_batch)
    inputs = processor(
        text=texts, images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="Process only the first episode, do not write the full cache.")
    ap.add_argument("--image-only", action="store_true",
                    help="Skip the task-description prompt; run VLM on image alone "
                         "(pair with CLIP text embeddings downstream).")
    ap.add_argument("--out-root", default=None,
                    help="Override .stable-wm/vlm_cache/<task>(_image_only)")
    args = ap.parse_args()

    stable_home = Path(os.environ.get("STABLEWM_HOME",
                                      "/home/yuhai/project/prism-jepa/.stable-wm"))
    task_cfg = TASKS[args.task]
    h5_path = stable_home / task_cfg["h5"]
    prompt = task_cfg["prompt"]
    variant_suffix = "_image_only" if args.image_only else ""
    out_dir = (Path(args.out_root) if args.out_root
               else stable_home / "vlm_cache" / f"{args.task}{variant_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cache] task={args.task}  h5={h5_path}")
    print(f"[cache] prompt={prompt!r}")
    print(f"[cache] out_dir={out_dir}")
    print(f"[cache] device={device}  batch={args.batch}  dry_run={args.dry_run}")

    print(f"\n[cache] loading model {MODEL_ID}…")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    hidden_dim = model.config.get_text_config().hidden_size
    print(f"[cache] model loaded in {time.time()-t0:.1f}s, hidden_dim={hidden_dim}")

    print(f"\n[cache] opening {h5_path}…")
    with h5py.File(h5_path, "r") as f:
        n_frames_total = f["pixels"].shape[0]
        ep_offset = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        n_episodes = len(ep_len)
        print(f"[cache] n_episodes={n_episodes}  n_frames={n_frames_total}")

        # In dry-run mode, only process episode 0.
        if args.dry_run:
            target_indices = np.arange(ep_offset[0], ep_offset[0] + ep_len[0])
            print(f"[cache] DRY RUN: episode 0 only, {len(target_indices)} frames")
        else:
            target_indices = np.arange(n_frames_total)

        # Allocate fp16 memmap for the full cache (or in-RAM array for dry run).
        n_target = len(target_indices)
        if args.dry_run:
            features = np.zeros((n_target, hidden_dim), dtype=np.float16)
        else:
            feat_path = out_dir / "features.npy"
            features = np.lib.format.open_memmap(
                str(feat_path), mode="w+", dtype=np.float16,
                shape=(n_target, hidden_dim),
            )
            print(f"[cache] memmap allocated at {feat_path}: "
                  f"{n_target} × {hidden_dim} fp16 = "
                  f"{n_target * hidden_dim * 2 / 1e9:.2f} GB")

        # Iterate in batches.
        t_start = time.time()
        n_done = 0
        with torch.no_grad():
            for batch_start in range(0, n_target, args.batch):
                batch_idx = target_indices[batch_start: batch_start + args.batch]
                # h5py wants sorted / increasing fancy indices
                frames_np = f["pixels"][list(batch_idx)]  # (b, 224, 224, 3) uint8
                images = [Image.fromarray(x) for x in frames_np]
                inputs = build_inputs(processor, images, prompt, device,
                                      image_only=args.image_only)
                out = model(**inputs, output_hidden_states=True, return_dict=True)
                hs = out.hidden_states[-1]  # (b, T, hidden)
                # Mean-pool over the token axis; respect attention_mask if present.
                if "attention_mask" in inputs:
                    mask = inputs.attention_mask.unsqueeze(-1).to(hs.dtype)
                    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                else:
                    pooled = hs.mean(dim=1)
                pooled_np = pooled.to(torch.float16).cpu().numpy()
                features[batch_start: batch_start + len(batch_idx)] = pooled_np
                n_done += len(batch_idx)

                if (batch_start // args.batch) % 50 == 0 or args.dry_run:
                    elapsed = time.time() - t_start
                    rate = n_done / max(elapsed, 1e-6)
                    remaining = (n_target - n_done) / max(rate, 1e-6)
                    print(f"[cache] {n_done:>7d}/{n_target} "
                          f"({100*n_done/n_target:5.1f}%) "
                          f"rate={rate:.1f} fr/s  "
                          f"ETA={remaining/3600:.2f} h")

    elapsed = time.time() - t_start
    print(f"\n[cache] done. {n_done} frames in {elapsed/3600:.2f} h "
          f"({n_done/elapsed:.1f} fr/s)")
    # Cast to fp32 for stats — fp16 sum-of-squared-deviations over 2M rows
    # overflows even when individual norms are bounded.
    norms_f32 = np.linalg.norm(np.asarray(features, dtype=np.float32), axis=1)
    print(f"[cache] feature L2 stats — "
          f"mean={float(norms_f32.mean()):.3f}  "
          f"std={float(norms_f32.std()):.3f}  "
          f"min={float(norms_f32.min()):.3f}  "
          f"max={float(norms_f32.max()):.3f}")

    # Write meta.
    meta = {
        "model_id": MODEL_ID,
        "task": args.task,
        "h5_path": str(h5_path),
        "prompt": prompt if not args.image_only else None,
        "image_only": args.image_only,
        "hidden_dim": hidden_dim,
        "n_frames": int(n_done),
        "batch_size": args.batch,
        "dtype": "float16",
        "pooling": "mean_over_tokens_with_attention_mask",
        "elapsed_sec": elapsed,
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        meta_path = out_dir / "meta.json"
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"[cache] wrote {meta_path}")
        # Flush memmap.
        del features
    else:
        print(f"[cache] DRY RUN — no files written. Schema would be:")
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
