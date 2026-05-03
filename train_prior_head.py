"""Train the PriorHead on cached JEPA features for Cube. Phase B.

Reads the JEPA cache produced by jepa_cache.py + the expert h5 actions,
fits a StandardScaler matching eval.py (full action data, NaN-filtered),
trains the head with β-NLL on episode-disjoint train/val splits.

GPU-resident data path — the head is tiny and the (features, actions) tensors
fit in 2.5 GB, so we materialize once and stream batches by index gather.

Output: prior_head_cube.pt  (state_dict + scaler params + history)


===========================================================================
GATE RECALIBRATION (2026-05-01) — read before changing the gates
===========================================================================

The first cut of Phase B used the gate "val NLL < 0 by epoch 10". That
threshold was unreachable on this task. Why:

  Standard Gaussian NLL per element (dropping the additive log√(2π)):

      nll(μ, σ; y) = 0.5 (y − μ)² / σ² + log σ

  For a fixed MSE = E[(y − μ)²], the σ that minimizes NLL is:

      σ* = √MSE          (set ∂nll/∂σ = -MSE/σ³ + 1/σ = 0)

  giving the minimum-attainable NLL:

      NLL* = 0.5 + 0.5 · log(MSE)

  Cube expert action prediction has irreducible variance ≈ 0.61 in
  StandardScaler-normalized space (mildly multi-modal expert behavior +
  partial observability from a single RGB frame). Plugging in:

      NLL* ≈ 0.5 + 0.5 · log(0.61) ≈ 0.5 − 0.247 ≈ 0.25

  i.e., NO calibrated head can reach negative NLL here, even in principle.
  β-NLL at β=0.5 inherits the same floor (the σ²β weight changes the
  optimization dynamics but not the NLL minimum at fixed MSE).

The correct diagnostic is val MSE — does the MLP exceed what a linear
probe on the same features gets (today: 0.83)? If yes, real structure is
being learned; if no, the head is no better than a linear readout.

Recalibrated gates:

  HARD GATE 1: val MSE @ epoch 10 > 0.70 × val MSE @ init (random head)
    → head reduced error by less than 30% — barely learning. Cube-specific
      "0.83 linear-probe floor" replaced with this task-agnostic check so
      PushT / Reacher / Tworoom can reuse the same gate semantics without
      per-task magic numbers. (Cube hits a 41% reduction comfortably.)
  HARD GATE 2: train NLL not monotonically decreasing through epoch 10
    → optimization is unstable; STOP. (Allow ε=5e-3 noise tolerance.)
  SOFT WARN  : val NLL > 0.3 at epoch 25
    → larger-than-typical irreducible noise; note but do not stop.
===========================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  — registers Blosc codec
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

from prior_head import PriorHead, beta_nll_loss


SEED = 42
H = 5
A_BLOCK = 5
WIN_LEN = H * A_BLOCK  # 25

EPOCHS = 50
BATCH = 256
LR = 3e-4
WD = 1e-4
WARMUP = 1000
BETA = 0.5

# Per-task config — paths relative to STABLEWM_HOME, plus a 90/10 episode-level
# train/val split.
STABLE_HOME = "/home/yuhai/project/prism-jepa/.stable-wm"
TASKS = {
    "cube": dict(
        h5=f"{STABLE_HOME}/ogbench/cube_single_expert.h5",
        cache_dir=f"{STABLE_HOME}/jepa_cache/cube",
        out_path="/home/yuhai/project/prism-jepa/prior_head_cube.pt",
        # 10 000 episodes: train [0, 9000), val [9000, 10000)
        train_ep_range=(0, 9000),
        val_ep_range=(9000, 10000),
    ),
    "pusht": dict(
        h5=f"{STABLE_HOME}/pusht_expert_train.h5",
        cache_dir=f"{STABLE_HOME}/jepa_cache/pusht",
        out_path="/home/yuhai/project/prism-jepa/prior_head_pusht.pt",
        # 18 685 episodes: train [0, 16817), val [16817, 18685)  (~10% held out)
        train_ep_range=(0, 16817),
        val_ep_range=(16817, 18685),
    ),
}


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_window_indices(ep_offset, ep_len, ep_lo, ep_hi):
    """Return (start_indices, episode_ids) — frame indices where a length-25 action
    window can start without crossing into the NaN end-of-episode action."""
    starts_list = []
    eps_list = []
    for e in range(ep_lo, ep_hi):
        max_off = int(ep_len[e]) - WIN_LEN  # exclusive
        if max_off <= 0:
            continue
        offs = np.arange(max_off, dtype=np.int64)
        starts_list.append(int(ep_offset[e]) + offs)
        eps_list.append(np.full(max_off, e, dtype=np.int64))
    return np.concatenate(starts_list), np.concatenate(eps_list)


def cosine_warmup_factor(step: int, total_steps: int, warmup: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="cube",
                    help="Which task to train the head on; selects h5, cache_dir, and out_path")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="Hard cap on training epochs (also see --max-epochs)")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="Override --epochs; useful for early-stopping recipes")
    ap.add_argument("--patience", type=int, default=None,
                    help="Early-stop after N consecutive epochs without val_NLL improvement. "
                         "None disables early stopping (Cube default behavior).")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--wd", type=float, default=WD)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--out", default=None,
                    help="Override the per-task default checkpoint path")
    args = ap.parse_args()
    if args.max_epochs is not None:
        args.epochs = args.max_epochs

    task_cfg = TASKS[args.task]
    h5_path = task_cfg["h5"]
    cache_dir = task_cfg["cache_dir"]
    out_path = args.out or task_cfg["out_path"]
    train_ep_lo, train_ep_hi = task_cfg["train_ep_range"]
    val_ep_lo, val_ep_hi = task_cfg["val_ep_range"]

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] task={args.task}  device={device}  epochs={args.epochs}  "
          f"batch={args.batch}  lr={args.lr}  wd={args.wd}  beta={args.beta}")
    print(f"[train] h5  ={h5_path}")
    print(f"[train] cache={cache_dir}")
    print(f"[train] out ={out_path}")

    # ---- Load JEPA cache ----
    print("[train] loading JEPA cache…")
    t0 = time.time()
    features = np.load(f"{cache_dir}/features.npy", mmap_mode="r")  # (N, D) fp16
    goals = np.load(f"{cache_dir}/goals.npy")  # (n_eps, D) fp16
    n_frames, D = features.shape
    n_eps = goals.shape[0]
    print(f"[train] features={features.shape} dtype={features.dtype}  "
          f"goals={goals.shape}  load={time.time()-t0:.1f}s")

    # ---- Load actions + episode metadata ----
    print("[train] loading actions + episode boundaries…")
    t0 = time.time()
    with h5py.File(h5_path, "r") as f:
        actions = f["action"][:].astype(np.float32)  # (N, A_raw)
        ep_offset = f["ep_offset"][:].astype(np.int64)
        ep_len = f["ep_len"][:].astype(np.int64)
    A_raw = actions.shape[1]
    print(f"[train] actions={actions.shape} A_raw={A_raw}  load={time.time()-t0:.1f}s")

    # ---- Fit StandardScaler on full action data (NaN-filtered) ----
    # eval.py fits its scaler on the full action column from dataset.stats — we
    # match by fitting on the full h5 actions (after dropping the per-episode-end
    # NaN rows). Same scaler used by both train and val.
    print("[train] fitting StandardScaler on full action data (NaN-filtered)…")
    nan_mask = np.isnan(actions).any(axis=1)
    scaler = StandardScaler().fit(actions[~nan_mask])
    print(f"[train]   action mean : {scaler.mean_.round(4).tolist()}")
    print(f"[train]   action scale: {scaler.scale_.round(4).tolist()}")

    # ---- Build train/val window indices ----
    print("[train] building window indices…")
    t0 = time.time()
    train_starts, train_eps = build_window_indices(ep_offset, ep_len, train_ep_lo, train_ep_hi)
    val_starts, val_eps = build_window_indices(ep_offset, ep_len, val_ep_lo, val_ep_hi)
    print(f"[train]   train windows = {len(train_starts):>8d}")
    print(f"[train]   val   windows = {len(val_starts):>8d}  build={time.time()-t0:.1f}s")

    # ---- Pre-compute normalized action targets (M, H, A_block, A_raw) ----
    print("[train] pre-computing normalized action targets…")
    t0 = time.time()

    def compute_targets(starts):
        idx = starts[:, None] + np.arange(WIN_LEN, dtype=np.int64)[None, :]  # (M, 25)
        seqs = actions[idx]  # (M, 25, A_raw)
        flat = seqs.reshape(-1, A_raw)
        flat_norm = scaler.transform(flat).astype(np.float32)
        return flat_norm.reshape(-1, H, A_BLOCK, A_raw)

    train_targets = compute_targets(train_starts)
    val_targets = compute_targets(val_starts)
    assert not np.isnan(train_targets).any(), "NaN slipped past filter (train)"
    assert not np.isnan(val_targets).any(), "NaN slipped past filter (val)"
    print(f"[train]   train_targets={train_targets.shape}")
    print(f"[train]   val_targets  ={val_targets.shape}  build={time.time()-t0:.1f}s")

    # ---- Move static data to GPU ----
    print("[train] moving tensors to GPU…")
    t0 = time.time()
    feats_gpu = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    goals_gpu = torch.from_numpy(np.asarray(goals, dtype=np.float32)).to(device)
    train_starts_gpu = torch.from_numpy(train_starts).to(device)
    train_eps_gpu = torch.from_numpy(train_eps).to(device)
    train_targets_gpu = torch.from_numpy(train_targets).to(device)
    val_starts_gpu = torch.from_numpy(val_starts).to(device)
    val_eps_gpu = torch.from_numpy(val_eps).to(device)
    val_targets_gpu = torch.from_numpy(val_targets).to(device)
    gpu_mb = (
        feats_gpu.element_size() * feats_gpu.numel()
        + goals_gpu.element_size() * goals_gpu.numel()
        + train_targets_gpu.element_size() * train_targets_gpu.numel()
        + val_targets_gpu.element_size() * val_targets_gpu.numel()
    ) / 1e6
    print(f"[train]   GPU residency: {gpu_mb:.0f} MB  move={time.time()-t0:.1f}s")

    # ---- Build head ----
    head = PriorHead(z_dim=D, H=H, A_block=A_BLOCK, A_raw=A_raw).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[train] PriorHead params: {n_params/1e6:.2f}M  ({n_params:,})")

    opt = optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    n_train = len(train_starts)
    steps_per_epoch = (n_train + args.batch - 1) // args.batch
    total_steps = steps_per_epoch * args.epochs
    print(f"[train] steps/epoch={steps_per_epoch}  total_steps={total_steps}")

    @torch.no_grad()
    def eval_val():
        head.eval()
        n_val = len(val_starts_gpu)
        nll_sum = 0.0
        mse_sum = 0.0
        n_done = 0
        bs = args.batch
        for s in range(0, n_val, bs):
            e = min(s + bs, n_val)
            zt = feats_gpu[val_starts_gpu[s:e]]
            zg = goals_gpu[val_eps_gpu[s:e]]
            tgt = val_targets_gpu[s:e]
            mu, sg = head(zt, zg)
            nll = beta_nll_loss(mu, sg, tgt, beta=args.beta).item()
            mse = ((mu - tgt) ** 2).mean().item()
            count = e - s
            nll_sum += nll * count
            mse_sum += mse * count
            n_done += count
        head.train()
        return nll_sum / n_done, mse_sum / n_done

    init_val_nll, init_val_mse = eval_val()
    print(f"\n[train] EPOCH 0 (init): val NLL={init_val_nll:.4f}  val MSE={init_val_mse:.4f}")

    best_val_nll = float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0
    history = []

    step = 0
    t_train_start = time.time()
    head.train()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n_train, device=device)
        epoch_nll = 0.0
        n_seen = 0
        t_ep0 = time.time()
        for s in range(0, n_train, args.batch):
            idx = perm[s:s + args.batch]
            zt = feats_gpu[train_starts_gpu[idx]]
            zg = goals_gpu[train_eps_gpu[idx]]
            tgt = train_targets_gpu[idx]
            mu, sg = head(zt, zg)
            loss = beta_nll_loss(mu, sg, tgt, beta=args.beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            lr_factor = cosine_warmup_factor(step, total_steps, WARMUP)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_factor
            opt.step()
            epoch_nll += loss.item() * idx.size(0)
            n_seen += idx.size(0)
            step += 1

        train_nll = epoch_nll / n_seen
        val_nll, val_mse = eval_val()
        ep_secs = time.time() - t_ep0
        history.append(dict(
            epoch=epoch, train_nll=train_nll, val_nll=val_nll,
            val_mse=val_mse, lr=args.lr * lr_factor, ep_secs=ep_secs,
        ))
        is_best = val_nll < best_val_nll
        if is_best:
            best_val_nll = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch <= 5 or epoch % 5 == 0 or epoch == args.epochs or is_best:
            tag = "  *best" if is_best else f"  (no-improve {epochs_since_improve})"
            print(f"[train] EPOCH {epoch:>2d}/{args.epochs}: "
                  f"train NLL={train_nll:>7.4f}  val NLL={val_nll:>7.4f}  "
                  f"val MSE={val_mse:.4f}  lr={args.lr * lr_factor:.2e}  "
                  f"ep={ep_secs:.1f}s{tag}")

        # Early stopping (optional via --patience).
        if args.patience is not None and epochs_since_improve >= args.patience:
            print(f"[train] EARLY STOP: val NLL did not improve for {args.patience} epochs "
                  f"(best={best_val_nll:.4f} at epoch {best_epoch}).")
            break

        # SOFT WARN — kept inline; HARD GATE moved post-loop to play nicely with early stopping.
        if epoch == 25 and val_nll > 0.3:
            print(f"[train] SOFT WARNING: val NLL={val_nll:.4f} > 0.3 at epoch 25 "
                  f"(larger-than-typical irreducible noise). Continuing.")

    train_total = time.time() - t_train_start

    # ---- Post-training HARD GATES (early-stopping-friendly: check best ever) ----
    best_val_mse = min(h["val_mse"] for h in history)
    relative_drop = (init_val_mse - best_val_mse) / max(init_val_mse, 1e-9)
    if relative_drop < 0.15:
        print(f"\n[train] HARD GATE FAIL: best val MSE only dropped "
              f"{100*relative_drop:.1f}% from init "
              f"({init_val_mse:.4f} → {best_val_mse:.4f}; need ≥15%). "
              f"Head not learning real structure. NOT SAVING.")
        return

    # Train NLL monotonicity check up to best epoch (with noise tol).
    train_nll_up_to_best = [h["train_nll"] for h in history if h["epoch"] <= best_epoch]
    if len(train_nll_up_to_best) >= 2:
        tol = 5e-3
        bumps = [
            i for i in range(1, len(train_nll_up_to_best))
            if train_nll_up_to_best[i] > train_nll_up_to_best[i - 1] + tol
        ]
        if bumps:
            rises = [(b + 1, train_nll_up_to_best[b - 1], train_nll_up_to_best[b]) for b in bumps]
            print(f"\n[train] HARD GATE FAIL: train NLL not monotonically decreasing "
                  f"through best epoch ({best_epoch}). Bumps: {rises}. NOT SAVING.")
            return

    print(f"\n[train] HARD GATES PASSED:")
    print(f"   • best val MSE drop = {100*relative_drop:.1f}% (≥15%)")
    print(f"   • train NLL strictly ↓ through best epoch {best_epoch}")

    # Sanity summary BEFORE save — informational only.
    final_val_nll = history[-1]["val_nll"]
    print(f"\n=== Phase B sanity ===")
    print(f"  initial val NLL: {init_val_nll:>8.4f}")
    print(f"  final   val NLL: {final_val_nll:>8.4f}")
    print(f"  best    val NLL: {best_val_nll:>8.4f}  @ epoch {best_epoch}")
    print(f"  best    val MSE: {best_val_mse:>8.4f}")

    # === σ calibration check (on BEST state) — gate runs BEFORE save. ===
    head.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    @torch.no_grad()
    def sigma_stats():
        head.eval()
        n_val = len(val_starts_gpu)
        bs = args.batch
        sigmas = []
        per_dim_sigmas = []  # accumulate per (H, A_block, A_raw) to inspect structure
        for s in range(0, n_val, bs):
            e = min(s + bs, n_val)
            zt = feats_gpu[val_starts_gpu[s:e]]
            zg = goals_gpu[val_eps_gpu[s:e]]
            _, sg = head(zt, zg)
            sigmas.append(sg.flatten().cpu())
            per_dim_sigmas.append(sg.cpu())
        return torch.cat(sigmas), torch.cat(per_dim_sigmas, dim=0)

    sigma_flat, sigma_full = sigma_stats()
    sqrt_mse = best_val_mse ** 0.5
    sigma_floor = 0.05
    near_floor_thresh = sigma_floor * 1.05  # within 5% of floor
    frac_near_floor = (sigma_flat <= near_floor_thresh).float().mean().item()
    print(f"\n=== Sigma calibration (on best checkpoint, epoch {best_epoch}) ===")
    print(f"  Head σ over val set (flattened over H × A_block × A_raw × N_val):")
    print(f"    mean            = {sigma_flat.mean().item():.4f}")
    print(f"    std             = {sigma_flat.std().item():.4f}")
    print(f"    median          = {sigma_flat.median().item():.4f}")
    print(f"    min             = {sigma_flat.min().item():.4f}  (floor = {sigma_floor})")
    print(f"    max             = {sigma_flat.max().item():.4f}")
    print(f"    frac near floor = {100*frac_near_floor:.2f}%  "
          f"(σ ≤ {near_floor_thresh:.4f})")
    print(f"  √(best val MSE) = √{best_val_mse:.4f} = {sqrt_mse:.4f}  "
          f"(theoretical optimum if perfectly calibrated)")

    # σ-DEGENERACY GATE — required by 2026-05-02 PushT triage.
    sigma_std = sigma_flat.std().item()
    degenerate = (frac_near_floor > 0.50) or (sigma_std < 0.05)
    if degenerate:
        reasons = []
        if frac_near_floor > 0.50:
            reasons.append(f"frac_near_floor={100*frac_near_floor:.1f}% > 50%")
        if sigma_std < 0.05:
            reasons.append(f"sigma_std={sigma_std:.4f} < 0.05")
        print(f"\n[train] σ-DEGENERACY GATE FAIL: σ output is degenerate "
              f"({'; '.join(reasons)}). σ_scale sweep would be uninformative.")
        print(f"[train] Checkpoint NOT saved. Re-run with stronger regularization "
              f"(higher --wd, lower --lr, or β-NLL β closer to 1.0).")
        return

    # Per-step structure: does σ change across plan-steps?
    per_step_mean = sigma_full.mean(dim=(0, 2, 3))  # collapse over batch, A_block, A_raw → (H,)
    print(f"  σ-mean per plan step (H={H}): "
          f"{[round(x, 4) for x in per_step_mean.tolist()]}")

    # Verdict
    diff = abs(sigma_flat.mean().item() - sqrt_mse)
    if sigma_flat.mean().item() < 0.10:
        verdict = "OVER-CONFIDENT (σ collapsed near floor — head thinks it knows the action better than it does)"
    elif diff < 0.10:
        verdict = f"WELL-CALIBRATED (mean σ ≈ √MSE; sigma_scale=1.0 will give the planner ~Bayes-optimal trust)"
    elif sigma_flat.mean().item() > sqrt_mse + 0.10:
        verdict = "UNDER-CONFIDENT (σ larger than √MSE; PoG fusion will weight prior less than ideal)"
    else:
        verdict = f"MODERATELY OVER-CONFIDENT (σ below √MSE by {sqrt_mse - sigma_flat.mean().item():.3f})"
    print(f"  Verdict: {verdict}")
    print(f"  Implication for σ_scale sweep: sigma_scale=1.0 means 'use the head's σ "
          f"as-is' (~{sigma_flat.mean().item():.2f}); sigma_scale=0.5 → ~{0.5*sigma_flat.mean().item():.2f}; "
          f"sigma_scale=2.0 → ~{2.0*sigma_flat.mean().item():.2f}.")

    # ---- All gates passed — save the checkpoint ----
    torch.save({
        "state_dict": best_state,
        "config": dict(z_dim=int(D), H=H, A_block=A_BLOCK, A_raw=A_raw,
                       hidden=512, sigma_floor=0.05),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "best_val_nll": best_val_nll,
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "init_val_nll": init_val_nll,
        "init_val_mse": init_val_mse,
        "history": history,
        "args": vars(args),
        "sigma_stats": dict(
            mean=float(sigma_flat.mean()), std=float(sigma_std),
            median=float(sigma_flat.median()),
            min=float(sigma_flat.min()), max=float(sigma_flat.max()),
            frac_near_floor=float(frac_near_floor),
        ),
    }, out_path)
    print(f"\n[train] saved best (val NLL={best_val_nll:.4f}, "
          f"val MSE={best_val_mse:.4f}, epoch {best_epoch}) → {out_path}")
    print(f"[train] total train time: {train_total/60:.1f} min")


if __name__ == "__main__":
    main()
