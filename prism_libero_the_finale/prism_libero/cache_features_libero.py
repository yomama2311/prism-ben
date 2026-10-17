"""Step 3 — encode every LIBERO frame with the trained (frozen) LeWM JEPA encoder.

Writes  $STABLEWM_HOME/jepa_cache/<name>/ :
    features.npy      (N, D) fp16   projector(CLS(encoder(frame)))   — same as repo's jepa_cache.py
    goals.npy         (n_ep, D)     latent of each episode's last frame
    task_goals.npy    (n_tasks, D)  per-task goal latent used at deployment (medoid of the
                                    final-frame latents of that task's demos)
    action_stats.json mean/std of the actions (what train.py's column normalizer used)
    meta.json

Usage:
    python prism_libero/cache_features_libero.py --h5 $STABLEWM_HOME/libero/libero_spatial.h5 \
        --jepa-ckpt $STABLEWM_HOME/<run>/lewm_epoch_100_object.ckpt --name libero_spatial --img-size 224
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preprocess(frames_u8: np.ndarray, device, img_size: int) -> torch.Tensor:
    """(b,H,W,3) uint8 -> (b,3,S,S) ImageNet-normalised float (same as train.py's pipeline)."""
    x = torch.from_numpy(frames_u8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                          align_corners=False, antialias=x.shape[-1] > img_size)
    return (x - IM_MEAN.to(device)) / IM_STD.to(device)


@torch.no_grad()
def encode_frames(jepa, frames_u8, device, img_size):
    x = preprocess(frames_u8, device, img_size)
    out = jepa.encoder(x, interpolate_pos_encoding=True)
    return jepa.projector(out.last_hidden_state[:, 0])


def pick_medoid(latents: np.ndarray) -> np.ndarray:
    mean = latents.mean(0, keepdims=True)
    return latents[np.argmin(((latents - mean) ** 2).sum(1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--jepa-ckpt", required=True, help="*_object.ckpt dumped by train.py")
    ap.add_argument("--name", required=True, help="cache sub-folder name")
    ap.add_argument("--img-size", type=int, default=224, help="must equal train.py img_size")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-frames", type=int, default=0, help="debug: only first N frames")
    args = ap.parse_args()

    home = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
    out_dir = home / "jepa_cache" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    jepa = torch.load(args.jepa_ckpt, map_location=device, weights_only=False).eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    with h5py.File(args.h5, "r") as f:
        N = f["pixels"].shape[0]
        ep_len, ep_off = f["ep_len"][:].astype(np.int64), f["ep_offset"][:].astype(np.int64)
        actions = f["action"][:].astype(np.float32)
        if args.max_frames:
            N = min(N, args.max_frames)
        D = int(encode_frames(jepa, f["pixels"][0:1], device, args.img_size).shape[-1])
        feats = np.lib.format.open_memmap(out_dir / "features.npy", mode="w+",
                                          dtype=np.float16, shape=(N, D))
        t0 = time.time()
        for s in range(0, N, args.batch):
            e = min(s + args.batch, N)
            feats[s:e] = encode_frames(jepa, f["pixels"][s:e], device,
                                       args.img_size).to(torch.float16).cpu().numpy()
            if (s // args.batch) % 50 == 0:
                print(f"[cache] {e}/{N} ({100 * e / N:.1f}%)  {e / (time.time() - t0 + 1e-9):.0f} fr/s")
        feats.flush()

    # per-episode goals, per-task goal bank
    meta_in = json.load(open(str(args.h5) + ".meta.json"))
    ep_task = np.asarray(meta_in["ep_task_id"])
    n_ep_use = int(((ep_off + ep_len) <= N).sum())  # episodes fully inside the cached range
    F32 = np.asarray(feats, dtype=np.float32)
    goals = F32[(ep_off[:n_ep_use] + ep_len[:n_ep_use] - 1)]
    np.save(out_dir / "goals.npy", goals.astype(np.float16))
    n_tasks = len(meta_in["task_names"])
    task_goals = np.zeros((n_tasks, D), dtype=np.float32)
    for t in range(n_tasks):
        sel = goals[ep_task[:n_ep_use] == t]
        if len(sel):
            task_goals[t] = pick_medoid(sel)
    np.save(out_dir / "task_goals.npy", task_goals)

    valid = actions[~np.isnan(actions).any(1)]
    stats = dict(mean=valid.mean(0).tolist(), std=valid.std(0, ddof=1).tolist())  # == train.py
    json.dump(stats, open(out_dir / "action_stats.json", "w"), indent=2)

    sample = F32[np.random.default_rng(0).choice(N, size=min(2048, N), replace=False)]
    per_dim_std = sample.std(0)
    ok = bool(np.isfinite(sample).all() and (per_dim_std < 1e-4).sum() <= D // 2)
    json.dump(dict(
        h5=str(args.h5), jepa_ckpt=str(args.jepa_ckpt), z_dim=D, n_frames=int(N),
        n_episodes=int(n_ep_use), img_size=args.img_size, gate_pass=ok,
        feat_std_mean=float(per_dim_std.mean()), dead_dims=int((per_dim_std < 1e-4).sum()),
    ), open(out_dir / "meta.json", "w"), indent=2)
    print(f"[cache] done -> {out_dir}   D={D}  gate={'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
