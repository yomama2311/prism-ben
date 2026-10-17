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

    def samples(self, eps):
        """All (episode, t) with t in [0, len-2]."""
        ee, tt = [], []
        for e in eps:
            n = int(self.ep_len[e]) - 1
            if n > 0:
                ee.append(np.full(n, e, dtype=np.int64)); tt.append(np.arange(n, dtype=np.int64))
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
    nll_sum = 0.0; cnt = 0
    se = torch.zeros(K, device=ep.device); se_n = torch.zeros(K, device=ep.device)
    anc_free = torch.zeros(K, device=ep.device); anc_hold = torch.zeros(K, device=ep.device)
    anc_lin = torch.zeros(K, device=ep.device); sig_all = []
    for s in range(0, n, args.batch):
        e_, t_ = ep[s:s + args.batch], t[s:s + args.batch]
        anchors, z_g, tgt, mask = data.batch(e_, t_, K, C, Ab, goal_swap_prob=0.0)
        mu, sg, _, free = forward_any(head, arch, anchors, z_g, K, C, 0, 0, train=False)
        nll_sum += masked_beta_nll(mu, sg, tgt, mask, args.beta).item() * e_.numel(); cnt += e_.numel()
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
        nll=nll_sum / cnt, mse_per_chunk=(se / se_n.clamp(min=1)).cpu().tolist(),
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
    ap.add_argument("--force-save", action="store_true")
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
    stats = json.load(open(cache / "action_stats.json"))
    a_mean, a_std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
    with h5py.File(args.h5, "r") as f:
        actions = f["action"][:N].astype(np.float32)
        ep_off = f["ep_offset"][:].astype(np.int64)
        ep_len = f["ep_len"][:].astype(np.int64)
    meta = json.load(open(str(args.h5) + ".meta.json"))
    ep_task = np.asarray(meta["ep_task_id"], dtype=np.int64)
    keep = (ep_off + ep_len) <= N
    ep_off, ep_len, ep_task = ep_off[keep], ep_len[keep], ep_task[keep]
    A_raw = actions.shape[1]
    acts_n = (actions - a_mean) / a_std
    print(f"[train] arch={args.arch}  N={N} D={D} A_raw={A_raw}  episodes={len(ep_len)}  "
          f"K={K} C={C} A_block={Ab}  (chunk={C * Ab} env steps, window={K * C * Ab})")

    tr_eps, va_eps = stratified_split(ep_task, args.val_frac, args.seed)
    data = GpuData(feats, acts_n, ep_off, ep_len, ep_task, tr_eps, device)
    tr_ep, tr_t = data.samples(tr_eps)
    va_ep, va_t = data.samples(va_eps)
    va_ep, va_t = va_ep[:: args.val_stride], va_t[:: args.val_stride]
    print(f"[train] train eps={len(tr_eps)} samples={tr_ep.numel()}   val eps={len(va_eps)} samples={va_ep.numel()}")

    feat_std = torch.from_numpy(feats[: min(N, 200000)].std(0)).to(device).clamp(min=1e-3)
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

    best, best_state, best_ep, hist, step = float("inf"), None, 0, [], 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
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
        hist.append(dict(epoch=epoch, train_loss=run / seen, val_nll=ev["nll"], val_mse=ev["mse"],
                         anc_free=ev["anc_free"]))
        is_best = ev["nll"] < best
        if is_best:
            best, best_ep = ev["nll"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if epoch <= 3 or epoch % 5 == 0 or epoch == args.epochs or is_best:
            print(f"[train] EPOCH {epoch:>3d}/{args.epochs} loss={run / seen:.4f} "
                  f"val NLL={ev['nll']:.4f} MSE={ev['mse']:.4f}{'  *best' if is_best else ''}")

    head.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    fin = evaluate(head, args.arch, data, va_ep, va_t, args, K, C, Ab, feat_std)
    print("\n=== validation report (best epoch %d, free-running anchors) ===" % best_ep)
    print("  action MSE per chunk k=0..K-1 :", [round(x, 3) for x in fin["mse_per_chunk"]])
    if args.arch == "chunked":
        print("  anchor MSE  learned           :", [round(x, 3) for x in fin["anc_free"]])
        print("  anchor MSE  hold (a_k = z_t)  :", [round(x, 3) for x in fin["anc_hold"]])
        print("  anchor MSE  linear z_t->z_g   :", [round(x, 3) for x in fin["anc_lin"]])
    sg = fin["sigma"]
    near_floor = (sg <= args.sigma_floor * 1.05).float().mean().item()
    print(f"  sigma mean={sg.mean():.3f} std={sg.std():.3f} frac_near_floor={100 * near_floor:.1f}%  "
          f"sqrt(MSE)={fin['mse'] ** 0.5:.3f}")

    drop = (init["mse"] - fin["mse"]) / max(init["mse"], 1e-9)
    degenerate = near_floor > 0.5 or sg.std().item() < 0.05
    if (drop < 0.15 or degenerate) and not args.force_save:
        print(f"[train] GATE FAIL (MSE drop {100 * drop:.1f}% < 15% or degenerate sigma={degenerate}); "
              f"not saving. Use --force-save to override.")
        return
    torch.save(dict(
        state_dict=best_state, config=cfg, arch=args.arch,
        scaler_mean=a_mean.tolist(), scaler_scale=a_std.tolist(),
        chunk_steps=C, a_block=Ab, max_chunks=K, best_val_nll=best, best_epoch=best_ep,
        history=hist, args=vars(args),
        sigma_stats=dict(mean=float(sg.mean()), std=float(sg.std()), frac_near_floor=near_floor),
    ), out_path)
    print(f"[train] saved -> {out_path}   ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
