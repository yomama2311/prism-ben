"""
Cache LeWM-JEPA pooled features for every frame of an expert h5 dataset, plus
one goal embedding per episode (last frame).

Output schema:
    .stable-wm/jepa_cache/<task>/
        features.npy   # (N_frames, D) fp16 memmap
        goals.npy      # (N_episodes, D) fp16
        meta.json

Run:
    python jepa_cache.py --task cube
    python jepa_cache.py --task cube --batch 256
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

TASKS = {
    "cube": {
        "h5": "ogbench/cube_single_expert.h5",
        "ckpt": "cube/lewm_object.ckpt",
    },
    "pusht": {
        "h5": "pusht_expert_train.h5",
        "ckpt": "pusht/lewm_object.ckpt",
    },
    "reacher": {
        "h5": "dmc/reacher_random.h5",
        "ckpt": "reacher/lewm_object.ckpt",
    },
    "tworoom": {
        "h5": "tworoom.h5",
        "ckpt": "tworooms/lewm_object.ckpt",
    },
}

# ImageNet normalization constants — matches eval.py's img_transform.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_uint8_batch(arr_uint8: np.ndarray, device) -> torch.Tensor:
    """(N, H, W, 3) uint8 → (N, 3, H, W) float32 on device, ImageNet-normalized."""
    t = torch.from_numpy(arr_uint8).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (t - mean) / std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true",
                    help="Process only the first 5 episodes to verify schema.")
    args = ap.parse_args()

    stable_home = Path(os.environ.get(
        "STABLEWM_HOME", "/home/yuhai/project/prism-jepa/.stable-wm"
    ))
    cfg = TASKS[args.task]
    h5_path = stable_home / cfg["h5"]
    ckpt_path = stable_home / cfg["ckpt"]
    out_dir = stable_home / "jepa_cache" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[jepa-cache] task={args.task}")
    print(f"[jepa-cache] h5  ={h5_path}")
    print(f"[jepa-cache] ckpt={ckpt_path}")
    print(f"[jepa-cache] out ={out_dir}")
    print(f"[jepa-cache] device={device}  batch={args.batch}  dry={args.dry_run}")

    print("\n[jepa-cache] loading JEPA module…")
    t0 = time.time()
    jepa = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    print(f"[jepa-cache] loaded in {time.time()-t0:.1f}s")

    encoder = jepa.encoder
    projector = jepa.projector
    # Probe hidden dim with a single dummy frame.
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        out = encoder(dummy, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        emb = projector(cls)
    hidden_dim = emb.shape[-1]
    print(f"[jepa-cache] hidden_dim={hidden_dim}, "
          f"projector type={type(projector).__name__}")

    print(f"\n[jepa-cache] opening {h5_path}…")
    with h5py.File(h5_path, "r") as f:
        n_frames = f["pixels"].shape[0]
        ep_offset = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        n_episodes = len(ep_len)

        if args.dry_run:
            n_episodes_use = 5
        else:
            n_episodes_use = n_episodes

        # The frames we actually need to encode = all frames in episodes [0, n_episodes_use)
        last_used_frame = int(ep_offset[n_episodes_use - 1] + ep_len[n_episodes_use - 1])
        target_indices = np.arange(0, last_used_frame)
        n_target = len(target_indices)
        print(f"[jepa-cache] n_episodes={n_episodes}  n_frames_total={n_frames}  "
              f"using {n_episodes_use} eps / {n_target} frames")

        # Goal indices = last frame of each episode used
        goal_indices = (ep_offset[:n_episodes_use] + ep_len[:n_episodes_use] - 1).astype(np.int64)
        # Sanity: are these indices all within target_indices range?
        assert goal_indices.max() < n_target

        # Allocate fp16 memmap for features.
        feat_path = out_dir / ("features.npy" if not args.dry_run else "features_dry.npy")
        features = np.lib.format.open_memmap(
            str(feat_path), mode="w+", dtype=np.float16,
            shape=(n_target, hidden_dim),
        )
        feat_size_gb = n_target * hidden_dim * 2 / 1e9
        print(f"[jepa-cache] feature memmap: {feat_path} "
              f"({n_target} × {hidden_dim} fp16 = {feat_size_gb:.2f} GB)")

        t_start = time.time()
        n_done = 0
        with torch.no_grad():
            for batch_start in range(0, n_target, args.batch):
                batch_idx = target_indices[batch_start:batch_start + args.batch]
                # h5py fancy indexing requires sorted; our range is contiguous so it's
                # already sorted, just slice rather than fancy-index.
                lo = int(batch_idx[0])
                hi = int(batch_idx[-1]) + 1
                frames_np = f["pixels"][lo:hi]  # (b, 224, 224, 3) uint8
                inputs = normalize_uint8_batch(frames_np, device)
                out = encoder(inputs, interpolate_pos_encoding=True)
                cls = out.last_hidden_state[:, 0]
                emb = projector(cls)  # (b, hidden_dim)
                features[batch_start:batch_start + len(batch_idx)] = (
                    emb.to(torch.float16).cpu().numpy()
                )
                n_done += len(batch_idx)

                if (batch_start // args.batch) % 50 == 0:
                    elapsed = time.time() - t_start
                    rate = n_done / max(elapsed, 1e-6)
                    eta = (n_target - n_done) / max(rate, 1e-6)
                    print(f"[jepa-cache] {n_done:>7d}/{n_target} "
                          f"({100*n_done/n_target:5.1f}%) "
                          f"rate={rate:.0f} fr/s  ETA={eta/60:.1f} min")

    elapsed = time.time() - t_start
    print(f"\n[jepa-cache] features done. {n_done} frames in {elapsed/60:.1f} min "
          f"({n_done/elapsed:.0f} fr/s)")

    # Build goals.npy by indexing the features memmap
    goals = features[goal_indices].copy()  # small read; fine in RAM
    goals_path = out_dir / ("goals.npy" if not args.dry_run else "goals_dry.npy")
    np.save(goals_path, goals)
    print(f"[jepa-cache] goals.npy: shape={goals.shape}  → {goals_path}")

    # Sanity prints (the GATE).
    print("\n=== Sanity (the gate) ===")
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(n_target, size=min(2048, n_target), replace=False)
    sample = np.asarray(features[sample_idx], dtype=np.float32)

    norms = np.linalg.norm(sample, axis=1)
    print(f"  3 random feature L2 norms: "
          f"{float(norms[0]):.3f} {float(norms[1]):.3f} {float(norms[2]):.3f}")
    print(f"  L2 norm range over sample: "
          f"min={float(norms.min()):.3f}  max={float(norms.max()):.3f}  "
          f"mean={float(norms.mean()):.3f}")

    per_dim_mean = sample.mean(axis=0)
    per_dim_std = sample.std(axis=0)
    n_dead = int((per_dim_std < 1e-4).sum())
    print(f"  per-dim std: min={float(per_dim_std.min()):.4f}  "
          f"median={float(np.median(per_dim_std)):.4f}  "
          f"max={float(per_dim_std.max()):.4f}")
    print(f"  per-dim mean: min={float(per_dim_mean.min()):.4f}  "
          f"max={float(per_dim_mean.max()):.4f}")
    print(f"  dead dims (std<1e-4): {n_dead}/{hidden_dim}")

    # Goal sanity
    g_norms = np.linalg.norm(goals.astype(np.float32), axis=1)
    print(f"  goals L2: min={float(g_norms.min()):.3f}  "
          f"max={float(g_norms.max()):.3f}  mean={float(g_norms.mean()):.3f}")

    # GATE: catastrophic-failure detection. Soft warnings vs hard stops.
    gate_pass = True
    if not np.isfinite(sample).all():
        print("  GATE FAIL: NaN/Inf in features")
        gate_pass = False
    if n_dead > hidden_dim // 2:
        print(f"  GATE FAIL: {n_dead}/{hidden_dim} dead dims (>50%)")
        gate_pass = False
    if float(norms.mean()) < 1e-3 or float(norms.mean()) > 1e6:
        print(f"  GATE FAIL: feature norm magnitude pathological "
              f"(mean={float(norms.mean()):.3e})")
        gate_pass = False
    print(f"  GATE: {'PASS' if gate_pass else 'FAIL'}")

    meta = {
        "task": args.task,
        "h5_path": str(h5_path),
        "ckpt_path": str(ckpt_path),
        "encoder_class": type(encoder).__name__,
        "projector_class": type(projector).__name__,
        "hidden_dim": int(hidden_dim),
        "n_frames": int(n_done),
        "n_episodes": int(n_episodes_use),
        "batch_size": int(args.batch),
        "dtype": "float16",
        "preprocessing": "uint8/255 → ImageNet mean/std",
        "elapsed_sec": float(elapsed),
        "gate_pass": bool(gate_pass),
        "dry_run": bool(args.dry_run),
        "norm_stats": {
            "mean": float(norms.mean()),
            "min": float(norms.min()),
            "max": float(norms.max()),
        },
        "n_dead_dims": int(n_dead),
    }
    meta_path = out_dir / ("meta.json" if not args.dry_run else "meta_dry.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n[jepa-cache] wrote {meta_path}")
    del features  # flush memmap


if __name__ == "__main__":
    main()
