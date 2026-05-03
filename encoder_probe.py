"""encoder_probe.py — 30-min diagnostic.

Linear-probe of frozen encoder features for next-action prediction on Cube.
Three encoders, identical recipe, single linear layer. The point is to compare
*what's already in the features* — not to train a good model.

Encoders:
  1. LeWM-JEPA           (CLS token from frozen LeWM ViT encoder; loaded from
                          .stable-wm/cube/lewm_object.ckpt)
  2. DINOv2-L/14          (CLS token from facebook/dinov2-large)
  3. Qwen2.5-VL-3B-vision (mean-pool of patch tokens from model.visual,
                           post-merger 2048-d; no language tower used)

Output: stdout table + probe_comparison.txt at project root.
"""
import json
import os
import random
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  — registers Blosc codec for h5py
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch import nn

SEED = 42
TRAIN_FRAMES = 2000
VAL_FRAMES = 500
TRAIN_EPS = (0, 9000)         # episodes [0, 9000)
VAL_EPS = (9000, 9500)        # held-out episodes
EPOCHS = 200
BATCH = 256
LR = 1e-3
WD = 1e-4

CUBE_H5 = "/home/yuhai/project/prism-jepa/.stable-wm/ogbench/cube_single_expert.h5"
JEPA_CKPT = "/home/yuhai/project/prism-jepa/.stable-wm/cube/lewm_object.ckpt"
DINOV2_ID = "facebook/dinov2-large"
QWEN_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def sample_valid_indices(rng, ep_offset, ep_len, ep_range, n, action_arr):
    lo, hi = ep_range
    pool = np.concatenate(
        [np.arange(ep_offset[e], ep_offset[e] + ep_len[e]) for e in range(lo, hi)]
    )
    valid = pool[~np.isnan(action_arr[pool]).any(axis=1)]
    sel = rng.choice(valid, size=n, replace=False)
    return np.sort(sel)


def imagenet_normalize_uint8(arr_uint8: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 → (N, 3, H, W) float32, ImageNet-normalized."""
    t = torch.from_numpy(arr_uint8.copy()).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (t - mean) / std


def encode_with_callable(fn, pixels_uint8, device, batch=32):
    feats = []
    pix_t = imagenet_normalize_uint8(pixels_uint8)
    for i in range(0, len(pix_t), batch):
        b = pix_t[i:i + batch].to(device)
        with torch.no_grad():
            out = fn(b)
        feats.append(out.detach().to(torch.float32).cpu())
    return torch.cat(feats, 0)


def encode_jepa(jepa, pixels_uint8, device, batch=32):
    encoder = jepa.encoder
    proj = jepa.projector

    def fn(x):
        out = encoder(x, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        return proj(cls)
    return encode_with_callable(fn, pixels_uint8, device, batch)


def encode_dinov2(dinov2, pixels_uint8, device, batch=32):
    def fn(x):
        out = dinov2(x)
        return out.last_hidden_state[:, 0]
    return encode_with_callable(fn, pixels_uint8, device, batch)


def encode_qwen_visual(qwen, processor, pixels_uint8, device, batch=8,
                       use_postmerger=False):
    """Mean-pool patch tokens from model.visual.

    use_postmerger=False: pre-merger ViT last_hidden_state (1280-d)
    use_postmerger=True : post-merger pooler_output (2048-d, Qwen's deployable visual rep)

    No language input, no LLM forward — only the visual module is invoked.
    """
    from qwen_vl_utils import process_vision_info
    visual = qwen.model.visual
    merge_size = visual.spatial_merge_size

    feats = []
    for i in range(0, len(pixels_uint8), batch):
        imgs = [Image.fromarray(x) for x in pixels_uint8[i:i + batch]]
        msgs = [
            [{"role": "user", "content": [{"type": "image", "image": img}]}]
            for img in imgs
        ]
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in msgs
        ]
        image_inputs, _ = process_vision_info(msgs)
        inputs = processor(
            text=texts, images=image_inputs, padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = visual(
                inputs.pixel_values.to(visual.dtype),
                grid_thw=inputs.image_grid_thw,
            )
        if use_postmerger:
            # pooler_output: (sum_post_merger_tokens, 2048) — tokens after spatial merge
            tokens_per_img = (
                inputs.image_grid_thw.prod(dim=1) // (merge_size ** 2)
            ).tolist()
            tokens_tensor = out.pooler_output
        else:
            # last_hidden_state: (sum_patch_tokens, 1280) — pre-merger ViT output
            tokens_per_img = inputs.image_grid_thw.prod(dim=1).tolist()
            tokens_tensor = out.last_hidden_state
        chunks = tokens_tensor.split(tokens_per_img, dim=0)
        pooled = torch.stack([c.mean(dim=0) for c in chunks])
        feats.append(pooled.detach().to(torch.float32).cpu())
    return torch.cat(feats, 0)


def train_probe(X_tr, y_tr, X_val, y_val, action_dim, device):
    set_seed()
    dim = X_tr.size(1)
    probe = nn.Linear(dim, action_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WD)

    X_tr = X_tr.to(device); y_tr = y_tr.to(device)
    X_val = X_val.to(device); y_val = y_val.to(device)
    n_tr = X_tr.size(0)

    best_val = float("inf")
    best_train = None
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(n_tr, device=device)
        probe.train()
        for s in range(0, n_tr, BATCH):
            b = perm[s:s + BATCH]
            opt.zero_grad()
            loss = ((probe(X_tr[b]) - y_tr[b]) ** 2).mean()
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            train_mse = ((probe(X_tr) - y_tr) ** 2).mean().item()
            val_mse = ((probe(X_val) - y_val) ** 2).mean().item()
        if val_mse < best_val:
            best_val = val_mse
            best_train = train_mse
    return best_val, best_train, time.time() - t0


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    overall_t0 = time.time()

    print("=== Loading h5 + sampling indices ===")
    with h5py.File(CUBE_H5, "r") as f:
        ep_offset = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        all_actions = f["action"][:].astype(np.float32)
    rng = np.random.RandomState(SEED)
    train_idx = sample_valid_indices(rng, ep_offset, ep_len, TRAIN_EPS, TRAIN_FRAMES, all_actions)
    val_idx = sample_valid_indices(rng, ep_offset, ep_len, VAL_EPS, VAL_FRAMES, all_actions)
    print(f"  train idx: {len(train_idx)} from episodes {TRAIN_EPS}")
    print(f"  val idx  : {len(val_idx)} from episodes {VAL_EPS}")

    all_idx = np.concatenate([train_idx, val_idx])
    sort_order = np.argsort(all_idx, kind="stable")
    sorted_idx = all_idx[sort_order]
    inv_order = np.argsort(sort_order, kind="stable")

    t0 = time.time()
    with h5py.File(CUBE_H5, "r") as f:
        sorted_pix = f["pixels"][list(sorted_idx)]
    pixels = sorted_pix[inv_order]
    train_pixels = pixels[:TRAIN_FRAMES]
    val_pixels = pixels[TRAIN_FRAMES:]
    print(f"  pixel load: {time.time()-t0:.1f}s, shape={pixels.shape}, dtype={pixels.dtype}")

    print("\n=== Fitting StandardScaler on train actions ===")
    train_actions = all_actions[train_idx]
    val_actions = all_actions[val_idx]
    scaler = StandardScaler().fit(train_actions)
    train_y = torch.from_numpy(scaler.transform(train_actions).astype(np.float32))
    val_y = torch.from_numpy(scaler.transform(val_actions).astype(np.float32))
    action_dim = train_y.size(1)
    print(f"  action_dim={action_dim}")
    print(f"  train_y std after norm: {train_y.std(0).numpy().round(3).tolist()}")

    results = []

    # ----- 1. LeWM-JEPA -----
    print("\n=== 1/3  LeWM-JEPA ===")
    t0 = time.time()
    jepa = torch.load(JEPA_CKPT, map_location=device, weights_only=False)
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in jepa.encoder.parameters())
    feat_train = encode_jepa(jepa, train_pixels, device)
    feat_val = encode_jepa(jepa, val_pixels, device)
    enc_time = time.time() - t0
    dim = feat_train.size(1)
    print(f"  encoder params: {n_params/1e6:.1f}M, dim: {dim}, encode: {enc_time:.1f}s")
    val_mse, train_mse, probe_time = train_probe(feat_train, train_y, feat_val, val_y, action_dim, device)
    print(f"  best val MSE: {val_mse:.4f}  train MSE @ best: {train_mse:.4f}  probe: {probe_time:.1f}s")
    results.append(("LeWM-JEPA", n_params, dim, val_mse, train_mse, enc_time + probe_time))
    del jepa
    torch.cuda.empty_cache()

    # ----- 2. DINOv2-L/14 -----
    print("\n=== 2/3  DINOv2-L/14 ===")
    t0 = time.time()
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(DINOV2_ID, dtype=torch.float32).to(device).eval()
    for p in dinov2.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in dinov2.parameters())
    feat_train = encode_dinov2(dinov2, train_pixels, device)
    feat_val = encode_dinov2(dinov2, val_pixels, device)
    enc_time = time.time() - t0
    dim = feat_train.size(1)
    print(f"  encoder params: {n_params/1e6:.1f}M, dim: {dim}, encode: {enc_time:.1f}s")
    val_mse, train_mse, probe_time = train_probe(feat_train, train_y, feat_val, val_y, action_dim, device)
    print(f"  best val MSE: {val_mse:.4f}  train MSE @ best: {train_mse:.4f}  probe: {probe_time:.1f}s")
    results.append(("DINOv2-L/14", n_params, dim, val_mse, train_mse, enc_time + probe_time))
    del dinov2
    torch.cuda.empty_cache()

    # ----- 3+4. Qwen2.5-VL-3B vision tower (pre- and post-merger) -----
    print("\n=== 3/4  Qwen2.5-VL-3B vision tower (pre-merger, 1280-d) ===")
    t0 = time.time()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_ID, dtype=torch.bfloat16, device_map=device
    ).eval()
    for p in qwen.parameters():
        p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(QWEN_ID)
    n_params = sum(p.numel() for p in qwen.model.visual.parameters())
    feat_train = encode_qwen_visual(qwen, processor, train_pixels, device, use_postmerger=False)
    feat_val = encode_qwen_visual(qwen, processor, val_pixels, device, use_postmerger=False)
    enc_time = time.time() - t0
    dim = feat_train.size(1)
    print(f"  visual params: {n_params/1e6:.1f}M, dim: {dim}, encode: {enc_time:.1f}s")
    val_mse, train_mse, probe_time = train_probe(feat_train, train_y, feat_val, val_y, action_dim, device)
    print(f"  best val MSE: {val_mse:.4f}  train MSE @ best: {train_mse:.4f}  probe: {probe_time:.1f}s")
    results.append(("Qwen-VL-vis (pre)", n_params, dim, val_mse, train_mse, enc_time + probe_time))

    print("\n=== 4/4  Qwen2.5-VL-3B vision tower (post-merger, 2048-d) ===")
    t0 = time.time()
    feat_train = encode_qwen_visual(qwen, processor, train_pixels, device, use_postmerger=True)
    feat_val = encode_qwen_visual(qwen, processor, val_pixels, device, use_postmerger=True)
    enc_time = time.time() - t0
    dim = feat_train.size(1)
    print(f"  visual+merger params: {n_params/1e6:.1f}M, dim: {dim}, encode: {enc_time:.1f}s")
    val_mse, train_mse, probe_time = train_probe(feat_train, train_y, feat_val, val_y, action_dim, device)
    print(f"  best val MSE: {val_mse:.4f}  train MSE @ best: {train_mse:.4f}  probe: {probe_time:.1f}s")
    results.append(("Qwen-VL-vis (post)", n_params, dim, val_mse, train_mse, enc_time + probe_time))
    del qwen
    torch.cuda.empty_cache()

    # ----- Output -----
    header = (
        f"{'Encoder':<18} {'Params':>10} {'Dim':>6} "
        f"{'Best Val MSE':>14} {'Train @ best':>14} {'Wall':>10}"
    )
    sep = "-" * len(header)
    table_lines = [header, sep]
    for name, params, dim, val_mse, train_mse, wall in results:
        table_lines.append(
            f"{name:<18} {params/1e6:>9.1f}M {dim:>6} "
            f"{val_mse:>14.4f} {train_mse:>14.4f} {wall:>9.1f}s"
        )

    val_mses = {r[0]: r[3] for r in results}
    jepa_mse = val_mses["LeWM-JEPA"]
    dino_mse = val_mses["DINOv2-L/14"]
    qwen_pre_mse = val_mses["Qwen-VL-vis (pre)"]
    qwen_post_mse = val_mses["Qwen-VL-vis (post)"]
    qwen_best_name, qwen_best_mse = (
        ("Qwen-VL-vis (post)", qwen_post_mse)
        if qwen_post_mse < qwen_pre_mse
        else ("Qwen-VL-vis (pre)", qwen_pre_mse)
    )
    jepa_v_dino = (jepa_mse - dino_mse) / jepa_mse * 100.0
    dino_v_qwen = (dino_mse - qwen_best_mse) / dino_mse * 100.0
    qwen_pre_v_post = (qwen_pre_mse - qwen_post_mse) / qwen_pre_mse * 100.0

    def verdict(pct, threshold):
        return f"meaningful (|{pct:+.1f}%| > {threshold}%)" if abs(pct) > threshold \
            else f"NOT meaningful (|{pct:+.1f}%| <= {threshold}%)"

    interp_lines = [
        "",
        "Interpretation:",
        f"  JEPA → DINOv2              val-MSE Δ = {jepa_v_dino:+6.1f}%   {verdict(jepa_v_dino, 20)}",
        f"  Qwen pre-merger → post     val-MSE Δ = {qwen_pre_v_post:+6.1f}%   (sanity on merger contribution)",
        f"  DINOv2 → {qwen_best_name:<19} val-MSE Δ = {dino_v_qwen:+6.1f}%   {verdict(dino_v_qwen, 10)}",
    ]
    # Recommendation
    best = min(results, key=lambda r: r[3])
    rec = (
        f"  Recommended backbone: {best[0]} "
        f"(best val MSE = {best[3]:.4f}, {best[1]/1e6:.0f}M params)."
    )
    interp_lines.append(rec)

    output = "\n".join(table_lines + interp_lines)
    print("\n" + output)

    out_path = Path("/home/yuhai/project/prism-jepa/probe_comparison.txt")
    with open(out_path, "w") as fh:
        fh.write("Encoder linear-probe comparison on Cube\n")
        fh.write(f"Train: {TRAIN_FRAMES} frames from episodes {TRAIN_EPS}\n")
        fh.write(f"Val:   {VAL_FRAMES} frames from episodes {VAL_EPS}\n")
        fh.write(f"Probe: nn.Linear(D->{action_dim}), AdamW lr={LR} wd={WD} bs={BATCH} "
                 f"{EPOCHS}ep, seed={SEED}\n")
        fh.write(f"Total wall time: {time.time()-overall_t0:.1f}s\n\n")
        fh.write(output + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
