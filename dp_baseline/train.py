"""Diffusion Policy training script for PRISM-JEPA baseline.

Trains DPModel with DDPM noise schedule on the same demonstrations used
to train PRISM's PriorHead (`pusht_expert_train.h5` / `cube_single_expert.h5`).

Standard DP training recipe (Chi 2023):
  - DDPM forward noise, K_train=100, squaredcos_cap_v2 schedule
  - MSE loss on predicted noise (epsilon prediction)
  - AdamW optimizer, cosine LR schedule with warmup
  - Gradient clipping at 1.0
  - EMA of model weights (decay 0.999) — eval uses EMA, not raw

Usage:
    python -m dp_baseline.train --task pusht --num-steps 100000
    python -m dp_baseline.train --task cube  --num-steps 100000

Output: dp_baseline_<task>.pt  (state_dict + EMA + action norm stats + config)
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader

from dp_baseline.dataset import DPDataset
from dp_baseline.model import DPModel
from dp_baseline.scheduler import DDPMScheduler


TASK_CONFIG = {
    "pusht": dict(
        h5="/home/yuhai/project/prism-jepa/.stable-wm/pusht_expert_train.h5",
        action_dim=2,
    ),
    "cube": dict(
        h5="/home/yuhai/project/prism-jepa/.stable-wm/ogbench/cube_single_expert.h5",
        action_dim=5,
    ),
}


# ---------------------------------------------------------------------------
# EMA helper (Chi 2023 uses decay 0.999, no warmup)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.999):
    """In-place EMA: ema_w ← decay * ema_w + (1-decay) * model_w."""
    ema_params = dict(ema_model.named_parameters())
    for name, p in model.named_parameters():
        if p.requires_grad:
            ema_params[name].data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    # Also copy buffers (BN running stats, etc.)
    ema_buffers = dict(ema_model.named_buffers())
    for name, b in model.named_buffers():
        ema_buffers[name].data.copy_(b.data)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def make_lr_scheduler(opt, num_steps, warmup_steps):
    warmup = LambdaLR(opt, lr_lambda=lambda s: min((s + 1) / warmup_steps, 1.0))
    cosine = CosineAnnealingLR(opt, T_max=max(1, num_steps - warmup_steps))
    return SequentialLR(opt, [warmup, cosine], milestones=[warmup_steps])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_CONFIG))
    ap.add_argument("--num-steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--action-horizon", type=int, default=25)
    ap.add_argument("--goal-offset-steps", type=int, default=25,
                    help="Goal frame is at window_start + goal_offset_steps "
                         "(env-steps). Match cfg.eval.goal_offset_steps (=25) "
                         "to eliminate the train/eval goal distribution mismatch.")
    ap.add_argument("--val-every-steps", type=int, default=2000)
    ap.add_argument("--log-every-steps", type=int, default=100)
    ap.add_argument("--ckpt-every-steps", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="output ckpt path (default: dp_baseline_<task>.pt)")
    ap.add_argument("--endframe-goal", action="store_true",
                    help="Rebuttal: goal = each episode's last frame (the original "
                         "pre-fix convention, 75-175 steps beyond the eval goal)")
    ap.add_argument("--random-goal-offset", action="store_true",
                    help="Rebuttal: sample the goal distance k ~ U{1..goal_offset_steps} "
                         "per training sample so the training distribution covers the "
                         "separations receding-horizon inference induces")
    ap.add_argument("--corrupt-ep-frac", type=float, default=0.0,
                    help="Rebuttal §5: fraction of train episodes with "
                         "uniform-random actions (val split untouched)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    cfg = TASK_CONFIG[args.task]
    out_path = args.out or f"/home/yuhai/project/prism-jepa/dp_baseline_{args.task}.pt"
    out_path = Path(out_path)
    log_path = out_path.with_suffix(".log")
    print(f"[train] task={args.task}  out={out_path}  log={log_path}")
    print(f"[train] config={cfg}")
    print(f"[train] num_steps={args.num_steps}  batch_size={args.batch_size}  "
          f"lr={args.lr}  weight_decay={args.weight_decay}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    # ---- Data ----
    t0 = time.time()
    train_ds = DPDataset(cfg["h5"], action_horizon=args.action_horizon,
                         goal_offset_steps=args.goal_offset_steps,
                         split="train", seed=args.seed,
                         corrupt_ep_frac=args.corrupt_ep_frac,
                         random_goal_offset=args.random_goal_offset,
                         endframe_goal=args.endframe_goal)
    val_ds = DPDataset(cfg["h5"], action_horizon=args.action_horizon,
                       goal_offset_steps=args.goal_offset_steps,
                       split="val", seed=args.seed)
    val_ds.copy_stats_from(train_ds)
    print(f"[data] init {time.time() - t0:.1f}s — "
          f"train_windows={len(train_ds):,}  val_windows={len(val_ds):,}")
    print(f"[data] action_min={train_ds.action_min.tolist()}")
    print(f"[data] action_max={train_ds.action_max.tolist()}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True,
        persistent_workers=False, drop_last=False,
    )

    # ---- Model + scheduler ----
    model = DPModel(action_dim=cfg["action_dim"]).to(device)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    print(f"[model] DPModel params: {model.num_params / 1e6:.2f} M")

    noise_sched = DDPMScheduler()
    print(f"[sched] DDPM K_train={noise_sched.num_train_timesteps}  "
          f"schedule=squaredcos_cap_v2")

    # ---- Optimizer ----
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = make_lr_scheduler(opt, args.num_steps, args.warmup_steps)

    # ---- Training loop ----
    step = 0
    best_val = float("inf")
    t_start = time.time()
    log_buf = []

    def log_line(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_buf.append(line + "\n")
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    log_line(f"=== begin training {args.task} for {args.num_steps} steps ===")

    train_iter = iter(train_loader)
    train_loss_accum = 0.0
    train_loss_count = 0

    while step < args.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        obs = batch["obs"].to(device, non_blocking=True)
        goal_obs = batch["goal_obs"].to(device, non_blocking=True)
        action_seq = batch["action_seq"].to(device, non_blocking=True)
        B = obs.shape[0]

        # Forward diffusion: sample noise + timestep, add to clean action seq
        t = torch.randint(0, noise_sched.num_train_timesteps, (B,), device=device)
        noise = torch.randn_like(action_seq)
        noisy = noise_sched.add_noise(action_seq, noise, t)

        # Predict noise and compute MSE
        pred = model(noisy, t, obs, goal_obs)
        loss = F.mse_loss(pred, noise)

        # Backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        lr_sched.step()

        # EMA
        ema_update(ema_model, model, decay=args.ema_decay)

        train_loss_accum += loss.item()
        train_loss_count += 1
        step += 1

        # ---- Periodic logging ----
        if step % args.log_every_steps == 0:
            avg_loss = train_loss_accum / max(train_loss_count, 1)
            elapsed = time.time() - t_start
            steps_per_sec = step / max(elapsed, 1e-3)
            eta_s = (args.num_steps - step) / max(steps_per_sec, 1e-3)
            lr_now = opt.param_groups[0]["lr"]
            log_line(f"step {step:>6d}/{args.num_steps}  "
                     f"loss={avg_loss:.4f}  lr={lr_now:.2e}  "
                     f"{steps_per_sec:.2f} steps/s  ETA={eta_s/60:.1f}min")
            train_loss_accum = 0.0
            train_loss_count = 0

        # ---- Periodic validation ----
        if step % args.val_every_steps == 0 or step == args.num_steps:
            val_loss = _validate(ema_model, val_loader, noise_sched, device)
            is_best = val_loss < best_val
            best_val = min(best_val, val_loss)
            log_line(f"  ↳ val_loss={val_loss:.4f}  "
                     f"(best={best_val:.4f}{' *' if is_best else ''})")

        # ---- Periodic checkpoint ----
        if step % args.ckpt_every_steps == 0 or step == args.num_steps:
            _save_ckpt(out_path, model, ema_model, train_ds, args, step, best_val)
            log_line(f"  ↳ saved ckpt @ step {step} → {out_path}")

    log_line(f"=== done in {(time.time() - t_start)/60:.1f} min ===")
    log_line(f"final best_val_loss = {best_val:.4f}")


@torch.no_grad()
def _validate(ema_model, val_loader, noise_sched, device):
    ema_model.eval()
    total, count = 0.0, 0
    for batch in val_loader:
        obs = batch["obs"].to(device, non_blocking=True)
        goal_obs = batch["goal_obs"].to(device, non_blocking=True)
        action_seq = batch["action_seq"].to(device, non_blocking=True)
        B = obs.shape[0]
        t = torch.randint(0, noise_sched.num_train_timesteps, (B,), device=device)
        noise = torch.randn_like(action_seq)
        noisy = noise_sched.add_noise(action_seq, noise, t)
        pred = ema_model(noisy, t, obs, goal_obs)
        total += F.mse_loss(pred, noise).item() * B
        count += B
    ema_model.train()
    return total / max(count, 1)


def _save_ckpt(out_path, model, ema_model, train_ds, args, step, best_val):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "action_min": train_ds.action_min,
        "action_max": train_ds.action_max,
        "action_dim": int(train_ds.action_dim),
        "action_horizon": int(train_ds.action_horizon),
        "config": vars(args),
        "step": step,
        "best_val_loss": best_val,
    }, out_path)


if __name__ == "__main__":
    main()
