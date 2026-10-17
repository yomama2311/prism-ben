"""Step 4 — train the chunked prior head (or the original flat head as an ablation) on the
cached LIBERO latents produced by cache_features_libero.py.

    python prism_libero/train_chunk_prior_head.py --h5 $STABLEWM_HOME/libero/libero_spatial.h5 \
        --name libero_spatial --chunk-steps 5 --max-chunks 4

Data layout used for one sample (episode e, frame t), L = chunk_steps * A_block raw steps:
    z_t        = feat[t]
    anchors    = feat[t + k*L]  k = 0..K     (clamped to the last frame -> "hold at goal")
    z_g        = last-frame latent of the episode (or of another demo of the same task with
                 prob. --goal-swap-prob, which mimics the deployment goal bank)
    action tgt = normalised raw actions [t, t + K*L), masked where they run past the episode

Loss = masked beta-NLL(actions)                       (decoder, noisy/free-run anchors)
     + lambda_anc * ( MSE(anc_tf) + gamma_free * MSE(anc_free) )   (latent path)
Model selection uses the *free-running* val NLL, i.e. what the planner sees at deployment.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import json
import math
import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import optim

from chunk_prior_head import ChunkedPriorHead, head_config, masked_beta_nll
from prior_head import PriorHead


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def cosine_warmup(step, total, warmup):
    if step < warmup:
        return step / max(warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def stratified_split(ep_task: np.ndarray, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for t in np.unique(ep_task):
        eps = np.where(ep_task == t)[0]
        rng.shuffle(eps)
        n_val = max(1, int(round(val_frac * len(eps)))) if len(eps) > 1 else 0
        va += eps[:n_val].tolist()
        tr += eps[n_val:].tolist()
    return np.array(sorted(tr)), np.array(sorted(va))


class GpuData:
    """Everything lives on the device; batches are built by index gather."""

    def __init__(self, feats, acts_n, ep_off, ep_len, ep_task, train_eps, device):
        self.feats = torch.from_numpy(feats).to(device)
        self.acts = torch.from_numpy(np.nan_to_num(acts_n, nan=0.0)).to(device)
        self.ep_off = torch.from_numpy(ep_off).to(device)
        self.ep_len = torch.from_numpy(ep_len).to(device)
        self.ep_task = torch.from_numpy(ep_task).to(device)
        n_tasks = int(ep_task.max()) + 1
        per_task = [train_eps[ep_task[train_eps] == t] for t in range(n_tasks)]
        mx = max(1, max(len(p) for p in per_task))
        pad = np.zeros((n_tasks, mx), dtype=np.int64)
        cnt = np.zeros(n_tasks, dtype=np.int64)
        for t, p in enumerate(per_task):
            cnt[t] = len(p)
            if len(p):
                pad[t, : len(p)] = p
                pad[t, len(p):] = p[0]
        self.task_eps = torch.from_numpy(pad).to(device)
        self.task_cnt = torch.from_numpy(cnt).to(device)
        self.device = device

    def samples(self, eps, window=1, policy="full"):
        """Sample windows without crossing episode boundaries.

        full requires all K*C*A_block actions and the final anchor to exist;
        padded retains the old masking/clamping behavior for ablations.
        """
        ee, tt = [], []
        for e in eps:
            length = int(self.ep_len[int(e)])
            n = length - window if policy == "full" else length - 1
            if n > 0:
                ee.append(np.full(n, int(e), dtype=np.int64))
                tt.append(np.arange(n, dtype=np.int64))
        if not ee:
            raise ValueError("No valid windows. Reduce --max-chunks/--chunk-steps or use --window-policy padded")
        return (torch.from_numpy(np.concatenate(ee)).to(self.device),
                torch.from_numpy(np.concatenate(tt)).to(self.device))

    def batch(self, ep, t, K, C, Ab, goal_swap_prob=0.0):
        L = C * Ab
        off, ln = self.ep_off[ep], self.ep_len[ep]
        k = torch.arange(K + 1, device=ep.device)
        a_idx = off[:, None] + torch.minimum(t[:, None] + k * L, (ln - 1)[:, None])
        anchors = self.feats[a_idx]                                   # (B, K+1, D)
        j = torch.arange(K * L, device=ep.device)
        raw_t = t[:, None] + j
        valid = raw_t <= (ln - 2)[:, None]                            # (B, K*L)
        tgt_idx = off[:, None] + torch.minimum(raw_t, (ln - 2)[:, None])
        B, A = ep.numel(), self.acts.shape[1]
        tgt = self.acts[tgt_idx].view(B, K, C, Ab, A)
        mask = valid.view(B, K, C, Ab, 1)
        ep_g = ep
        if goal_swap_prob > 0:
            task = self.ep_task[ep]
            r = (torch.rand(B, device=ep.device) * self.task_cnt[task]).long()
            swapped = self.task_eps[task, r]
            ep_g = torch.where(torch.rand(B, device=ep.device) < goal_swap_prob, swapped, ep)
        z_g = self.feats[self.ep_off[ep_g] + self.ep_len[ep_g] - 1]
        return anchors, z_g, tgt, mask


def forward_any(head, arch, anchors, z_g, K, C, anchor_noise, p_pred, train=True):
    """Returns mu, sigma (B,K,C,Ab,A), anc_tf, anc_free (None for the flat head)."""
    if arch == "chunked":
        if train:
            o = head.forward_train(anchors, z_g, anchor_noise=anchor_noise, p_pred=p_pred)
            return o["mu"], o["sigma"], o["anc_tf"], o["anc_free"]
        free = head.predict_anchors(anchors[:, 0], z_g, K)
        mus, sgs = [], []
        for k in range(K):
            m, s = head.decode_chunk(free[:, k], free[:, k + 1], z_g, k)
            mus.append(m); sgs.append(s)
        return torch.stack(mus, 1), torch.stack(sgs, 1), None, free[:, 1:]
    mu, sg = head(anchors[:, 0], z_g)                                  # (B, K*C, Ab, A)
    B = mu.size(0)
    return mu.view(B, K, C, *mu.shape[2:]), sg.view(B, K, C, *sg.shape[2:]), None, None


@torch.no_grad()
def evaluate(head, arch, data, ep, t, args, K, C, Ab, feat_std):
    head.eval()
    n = ep.numel()
    nll_sum = 0.0; nll_count = 0
    cnt = 0
    se = torch.zeros(K, device=ep.device); se_n = torch.zeros(K, device=ep.device)
    anc_free = torch.zeros(K, device=ep.device); anc_hold = torch.zeros(K, device=ep.device)
    anc_lin = torch.zeros(K, device=ep.device); sig_all = []
    for s in range(0, n, args.batch):
        e_, t_ = ep[s:s + args.batch], t[s:s + args.batch]
        anchors, z_g, tgt, mask = data.batch(e_, t_, K, C, Ab, goal_swap_prob=0.0)
        mu, sg, _, free = forward_any(head, arch, anchors, z_g, K, C, 0, 0, train=False)
        valid_elements = float(mask.expand_as(mu).sum())
        nll_sum += masked_beta_nll(mu, sg, tgt, mask, args.beta).item() * valid_elements
        nll_count += valid_elements
        cnt += e_.numel()
        m = mask.float()
        se += (((mu - tgt) ** 2) * m).sum(dim=(0, 2, 3, 4)); se_n += (m.expand_as(mu)).sum(dim=(0, 2, 3, 4))
        sig_all.append(sg.flatten().cpu())
        if free is not None:
            true = anchors[:, 1:]
            anc_free += (((free - true) / feat_std) ** 2).mean(-1).sum(0)
            anc_hold += (((anchors[:, :1] - true) / feat_std) ** 2).mean(-1).sum(0)
            frac = (torch.arange(1, K + 1, device=ep.device).float() / K).view(1, K, 1)
            lin = anchors[:, :1] + frac * (z_g[:, None] - anchors[:, :1])
            anc_lin += (((lin - true) / feat_std) ** 2).mean(-1).sum(0)
    head.train()
    return dict(
        nll=nll_sum / max(nll_count, 1), mse_per_chunk=(se / se_n.clamp(min=1)).cpu().tolist(),
        mse=float(se.sum() / se_n.sum().clamp(min=1)),
        anc_free=(anc_free / cnt).cpu().tolist(), anc_hold=(anc_hold / cnt).cpu().tolist(),
        anc_lin=(anc_lin / cnt).cpu().tolist(), sigma=torch.cat(sig_all),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--name", required=True, help="cache folder name under $STABLEWM_HOME/jepa_cache")
    ap.add_argument("--arch", choices=["chunked", "flat"], default="chunked",
                    help="flat = the repo's original single-MLP PriorHead over K*C steps (ablation)")
    ap.add_argument("--chunk-steps", type=int, default=5, help="C: plan-steps per chunk")
    ap.add_argument("--a-block", type=int, default=5, help="env steps per plan-step (= frameskip)")
    ap.add_argument("--max-chunks", type=int, default=4, help="K: chunks per training window")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--sigma-floor", type=float, default=0.05)
    ap.add_argument("--anchor-noise", type=float, default=0.1)
    ap.add_argument("--p-pred", type=float, default=0.3)
    ap.add_argument("--lambda-anc", type=float, default=1.0)
    ap.add_argument("--gamma-free", type=float, default=0.5)
    ap.add_argument("--goal-swap-prob", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--val-stride", type=int, default=3, help="evaluate on every n-th val sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force-save", action="store_true", help="legacy alias; gate no longer deletes checkpoints")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=8, help="epochs without improvement before early stopping; 0 disables")
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--monitor", choices=["val_nll", "val_mse"], default="val_nll")
    ap.add_argument("--window-policy", choices=["full", "padded"], default="full",
                    help="full: only windows containing all K*C*Ab raw actions")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="directory for best.pt, last.pt, epoch_010.pt, history.jsonl, summary.json")
    ap.add_argument("--resume", default=None, help="resume training from last.pt or epoch_NNN.pt")
    args = ap.parse_args()

    K, C, Ab = args.max_chunks, args.chunk_steps, args.a_block
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    home = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
    cache = home / "jepa_cache" / args.name
    out_path = args.out or f"prior_head_{args.name}_{args.arch}.pt"

    feats = np.load(cache / "features.npy", mmap_mode="r")
    feats = np.asarray(feats, dtype=np.float32)
    N, D = feats.shape
    # World-model statistics are in action_stats.json. The PRIOR head must fit
    # its own action scaler on prior-training episodes only (no validation leak).
    stats = json.load(open(cache / "action_stats.json"))
    with h5py.File(args.h5, "r") as f:
        actions = f["action"][:N].astype(np.float32)
        ep_off = f["ep_offset"][:].astype(np.int64)
        ep_len = f["ep_len"][:].astype(np.int64)
    meta = json.load(open(str(args.h5) + ".meta.json"))
    ep_task = np.asarray(meta["ep_task_id"], dtype=np.int64)
    keep = (ep_off + ep_len) <= N
    ep_off, ep_len, ep_task = ep_off[keep], ep_len[keep], ep_task[keep]
    A_raw = actions.shape[1]
    tr_eps, va_eps = stratified_split(ep_task, args.val_frac, args.seed)
    tr_act_indices = np.concatenate([
        np.arange(ep_off[int(e)], ep_off[int(e)] + ep_len[int(e)] - 1)
        for e in tr_eps if ep_len[int(e)] > 1
    ])
    tr_actions = actions[tr_act_indices]
    tr_actions = tr_actions[np.isfinite(tr_actions).all(1)]
    a_mean = np.mean(tr_actions, axis=0).astype(np.float32)
    a_std = np.std(tr_actions, axis=0, ddof=1).astype(np.float32)
    if (a_std <= 1e-8).any():
        raise ValueError("Prior-training action dimension has zero variance")
    acts_n = (actions - a_mean) / a_std
    print(f"[train] arch={args.arch}  N={N} D={D} A_raw={A_raw}  episodes={len(ep_len)}  "
          f"K={K} C={C} A_block={Ab}  (chunk={C * Ab} env steps, window={K * C * Ab})")
    print(f"[train] prior train-only scaler mean={a_mean.round(4)} std={a_std.round(4)}")
    data = GpuData(feats, acts_n, ep_off, ep_len, ep_task, tr_eps, device)
    tr_ep, tr_t = data.samples(tr_eps, window=K*C*Ab, policy=args.window_policy)
    va_ep, va_t = data.samples(va_eps, window=K*C*Ab, policy=args.window_policy)
    va_ep, va_t = va_ep[:: args.val_stride], va_t[:: args.val_stride]
    print(f"[train] train eps={len(tr_eps)} samples={tr_ep.numel()}   val eps={len(va_eps)} samples={va_ep.numel()}")

    # Calculate latent feature scale on training episodes only, avoiding val leakage.
    tr_frames = np.concatenate([np.arange(ep_off[e], ep_off[e] + ep_len[e]) for e in tr_eps])
    feat_std = torch.from_numpy(feats[tr_frames].std(0)).to(device).clamp(min=1e-3)
    if args.arch == "chunked":
        head = ChunkedPriorHead(D, Ab, A_raw, chunk_steps=C, max_chunks=K, hidden=args.hidden,
                                sigma_floor=args.sigma_floor).to(device)
        head.feat_std.copy_(feat_std)
        cfg = head_config(head)
    else:
        head = PriorHead(D, H=K * C, A_block=Ab, A_raw=A_raw, hidden=args.hidden,
                         sigma_floor=args.sigma_floor).to(device)
        cfg = dict(arch="flat", z_dim=D, H=K * C, A_block=Ab, A_raw=A_raw, hidden=args.hidden,
                   sigma_floor=args.sigma_floor)
    n_par = sum(p.numel() for p in head.parameters())
    print(f"[train] head params: {n_par / 1e6:.2f}M")

    opt = optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    n_train = tr_ep.numel()
    total = math.ceil(n_train / args.batch) * args.epochs
    init = evaluate(head, args.arch, data, va_ep, va_t, args, K, C, Ab, feat_std)
    print(f"[train] EPOCH 0: val NLL={init['nll']:.4f}  MSE={init['mse']:.4f}")

    # Checkpoints are ALWAYS saved; the diagnostic quality gate is metadata, not a
    # destructive filter. A failed gate must not silently leave an old model in place.
    ckpt_dir = Path(args.checkpoint_dir or (Path(str(out_path)).with_suffix("" ).as_posix() + "_checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "split.json").write_text(json.dumps(
        dict(train_episodes=tr_eps.tolist(), val_episodes=va_eps.tolist(),
             window_policy=args.window_policy, seed=args.seed), indent=2))

    def save_ckpt(path, epoch, metric, best_epoch, best_metric, history, quality=None, train_state=None):
        state = {
            "state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            "config": cfg, "arch": args.arch, "scaler_mean": a_mean.tolist(),
            "scaler_scale": a_std.tolist(), "chunk_steps": C, "a_block": Ab,
            "max_chunks": K, "epoch": int(epoch), "monitor": args.monitor,
            "monitor_value": float(metric), "best_epoch": int(best_epoch),
            "best_val_metric": float(best_metric), "history": history,
            "args": vars(args), "quality_gate": quality or {},
            "split_file": str(ckpt_dir / "split.json"),
        }
        if train_state:
            state.update(optimizer_state=opt.state_dict(), global_step=step,
                         best_bad_epochs=bad_epochs)
        tmp = str(path) + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)

    best, best_ep, hist, step, bad_epochs = float("inf"), 0, [], 0, 0
    start_epoch = 1
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resumed.get("config") != cfg:
            raise ValueError("Resume checkpoint architecture/config does not match")
        head.load_state_dict(resumed["state_dict"])
        if "optimizer_state" not in resumed:
            raise ValueError("Resume requires last.pt or periodic epoch checkpoint with optimizer_state")
        opt.load_state_dict(resumed["optimizer_state"])
        # Move optimizer state to the active device after restoring.
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        step = int(resumed["global_step"])
        start_epoch = int(resumed["epoch"]) + 1
        best_ep = int(resumed["best_epoch"])
        best = float(resumed["best_val_metric"])
        bad_epochs = int(resumed.get("best_bad_epochs", 0))
        hist = list(resumed.get("history", []))
        print(f"[resume] epoch={start_epoch} best_epoch={best_ep} best={best:.6f}", flush=True)

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        perm = torch.randperm(n_train, device=device)
        run, seen = 0.0, 0
        for s in range(0, n_train, args.batch):
            idx = perm[s:s + args.batch]
            anchors, z_g, tgt, mask = data.batch(tr_ep[idx], tr_t[idx], K, C, Ab, args.goal_swap_prob)
            mu, sg, anc_tf, anc_free = forward_any(head, args.arch, anchors, z_g, K, C,
                                                   args.anchor_noise, args.p_pred, train=True)
            loss = masked_beta_nll(mu, sg, tgt, mask, args.beta)
            if anc_tf is not None:
                true = anchors[:, 1:]
                l_tf = (((anc_tf - true) / feat_std) ** 2).mean()
                l_fr = (((anc_free - true) / feat_std) ** 2).mean()
                loss = loss + args.lambda_anc * (l_tf + args.gamma_free * l_fr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            for g in opt.param_groups:
                g["lr"] = args.lr * cosine_warmup(step, total, args.warmup)
            opt.step()
            run += loss.item() * idx.numel(); seen += idx.numel(); step += 1
        ev = evaluate(head, args.arch, data, va_ep, va_t, args, K, C, Ab, feat_std)
        monitor_value = ev["nll"] if args.monitor == "val_nll" else ev["mse"]
        row = dict(epoch=epoch, train_loss=run / max(seen, 1), val_nll=ev["nll"],
                   val_mse=ev["mse"], lr=opt.param_groups[0]["lr"],
                   anchor_mse=ev["anc_free"], sigma_mean=float(ev["sigma"].mean()))
        hist.append(row)
        improved = monitor_value < best - args.min_delta
        if improved:
            best, best_ep, bad_epochs = monitor_value, epoch, 0
            save_ckpt(ckpt_dir / "best.pt", epoch, monitor_value, best_ep, best, hist)
        else:
            bad_epochs += 1
        save_ckpt(ckpt_dir / "last.pt", epoch, monitor_value, best_ep, best, hist, train_state=True)
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            save_ckpt(ckpt_dir / f"epoch_{epoch:03d}.pt", epoch, monitor_value, best_ep, best, hist, train_state=True)
        with (ckpt_dir / "history.jsonl").open("w") as f:
            for h in hist:
                f.write(json.dumps(h, allow_nan=False) + "\n")
        with (ckpt_dir / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_nll", "val_mse", "lr", "anchor_mse", "sigma_mean"])
            writer.writeheader(); writer.writerows(hist)
        print(f"[EPOCH] arch={args.arch} epoch={epoch} train_loss={row['train_loss']:.6f} "
              f"val_nll={ev['nll']:.6f} val_mse={ev['mse']:.6f} "
              f"monitor={monitor_value:.6f} best_epoch={best_ep} "
              f"{'*best' if improved else ''}", flush=True)
        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"[EARLY_STOP] patience={args.patience} best_epoch={best_ep}", flush=True)
            break

    if not (ckpt_dir / "best.pt").exists():
        raise RuntimeError("No best checkpoint created; check NaNs in validation metrics")
    best_data = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(best_data["state_dict"])
    fin = evaluate(head, args.arch, data, va_ep, va_t, args, K, C, Ab, feat_std)
    print("\n=== validation report (best epoch %d, free-running anchors) ===" % best_ep)
    print("  action MSE per chunk k=0..K-1 :", [round(x, 3) for x in fin["mse_per_chunk"]])
    if args.arch == "chunked":
        print("  anchor MSE  learned           :", [round(x, 3) for x in fin["anc_free"]])
        print("  anchor MSE  hold (a_k = z_t)  :", [round(x, 3) for x in fin["anc_hold"]])
        print("  anchor MSE  linear z_t->z_g   :", [round(x, 3) for x in fin["anc_lin"]])
    sg = fin["sigma"]
    near_floor = float((sg <= args.sigma_floor * 1.05).float().mean())
    sigma_var = float(sg.std())
    drop = (init["mse"] - fin["mse"]) / max(init["mse"], 1e-9)
    degenerate = near_floor > 0.5 or sigma_var < 0.05
    quality = {"pass": bool(drop >= 0.15 and not degenerate),
               "mse_drop_fraction": float(drop), "sigma_degenerate": bool(degenerate),
               "sigma_near_floor_fraction": near_floor, "sigma_std": sigma_var}
    print(f"[GATE] {'PASS' if quality['pass'] else 'FAIL'} MSE drop={100*drop:.1f}% "
          f"sigma_std={sigma_var:.4f} near_floor={near_floor:.1%}; checkpoint RETAINED", flush=True)
    # Annotate best checkpoint with final diagnostics and preserve its epoch.
    save_ckpt(ckpt_dir / "best.pt", best_ep, best, best_ep, best, hist, quality=quality)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_dir / "best.pt", out_path)
    summary = dict(arch=args.arch, best_epoch=best_ep, best_metric=best,
                   monitor=args.monitor, completed_epochs=len(hist), quality_gate=quality,
                   best_checkpoint=str(ckpt_dir / "best.pt"),
                   last_checkpoint=str(ckpt_dir / "last.pt"),
                   deployment_checkpoint=str(out_path),
                   best_val_nll=float(fin["nll"]), best_val_mse=float(fin["mse"]),
                   mse_per_chunk=fin["mse_per_chunk"],
                   anchor_learned=fin["anc_free"], anchor_hold=fin["anc_hold"],
                   anchor_linear=fin["anc_lin"], duration_seconds=time.time()-t_start)
    (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[BEST] {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
