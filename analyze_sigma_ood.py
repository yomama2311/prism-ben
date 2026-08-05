"""σ-OOD calibration study for CoRL rebuttal §4 (AC-4; BHdk-Q1; SfQM-W5,Q7).

Question: does the prior head's predicted σ actually increase on OOD inputs,
or is the PoG fallback only algebraic?

Conditions (per task, on the episode-disjoint VAL split):
  A  in_dist        z_t = val frame,  z_g = own episode's goal (endframe)
                    — reproduces the training-time calibration reference.
  B  mismatch_goal  z_t = val frame,  z_g = goal of a DIFFERENT val episode
                    — semantic OOD: novel (state, goal) combination. This is
                    exactly the "novel state-goal pair" regime reviewers ask
                    about (and the tworoom failure mode).
  C  noise_z        z_t ~ N(mean, std) moment-matched to val features
                    — extreme representation-level OOD sanity check.

Metrics: mean/median σ, ratio vs in-dist, and the mean per-dim PoG weight the
prior receives at plan time: w = τ_p / (τ_p + τ_π) = 1 / (1 + (s·σ_p)²) with
σ_π = var_scale = 1.0 and s = 1 (paper setting).

Visual-corruption and random-rollout conditions need the encoder / env and are
run separately (analyze_sigma_ood_visual.py) to keep this script cache-only.

Run:  python analyze_sigma_ood.py            # both tasks, CPU
Out:  logs/sigma_ood_results.tsv + stdout table
"""
from __future__ import annotations

import json
import os

import h5py
import hdf5plugin  # noqa: F401 — registers Blosc codec
import numpy as np
import torch

from prior_head import PriorHead

STABLE_HOME = "/home/yuhai/project/prism-jepa/.stable-wm"
OUT_TSV = "/home/yuhai/project/prism-jepa/logs/sigma_ood_results.tsv"
M_SAMPLES = 20_000
SEED = 0

TASKS = {
    "pusht": dict(
        h5=f"{STABLE_HOME}/pusht_expert_train.h5",
        cache_dir=f"{STABLE_HOME}/jepa_cache/pusht",
        ckpt="/home/yuhai/project/prism-jepa/prior_head_pusht.pt",
        val_ep_range=(16817, 18685),
    ),
    "cube": dict(
        h5=f"{STABLE_HOME}/ogbench/cube_single_expert.h5",
        cache_dir=f"{STABLE_HOME}/jepa_cache/cube",
        ckpt="/home/yuhai/project/prism-jepa/prior_head_cube.pt",
        val_ep_range=(9000, 10000),
    ),
}

WIN_LEN = 25  # H * A_block — matches train_prior_head.py


def build_window_starts(ep_offset, ep_len, ep_lo, ep_hi, win_len=WIN_LEN):
    """Same logic as train_prior_head.build_window_indices."""
    starts, eps = [], []
    for e in range(ep_lo, ep_hi):
        max_off = int(ep_len[e]) - win_len
        if max_off <= 0:
            continue
        offs = np.arange(max_off)
        starts.append(int(ep_offset[e]) + offs)
        eps.append(np.full(max_off, e))
    return np.concatenate(starts), np.concatenate(eps)


@torch.no_grad()
def sigma_stats(head, z_t, z_g, batch=4096):
    """Return flat σ over all samples×dims, plus PoG weight at s=1, σ_π=1."""
    sig_all = []
    for s in range(0, len(z_t), batch):
        _, sigma = head(z_t[s : s + batch], z_g[s : s + batch])
        sig_all.append(sigma.reshape(sigma.shape[0], -1))
    sig = torch.cat(sig_all)          # (M, H*A_block*A_raw)
    w = 1.0 / (1.0 + sig.pow(2))      # per-dim PoG weight on the prior
    return dict(
        sigma_mean=sig.mean().item(),
        sigma_median=sig.median().item(),
        sigma_p90=sig.quantile(0.90).item(),
        pog_weight_mean=w.mean().item(),
    )


def run_task(task, cfg, rng):
    print(f"\n================ {task} ================")
    ckpt = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    hp = ckpt["config"]
    head = PriorHead(z_dim=hp["z_dim"], H=hp["H"], A_block=hp["A_block"],
                     A_raw=hp["A_raw"], hidden=hp["hidden"],
                     sigma_floor=hp["sigma_floor"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    saved = ckpt.get("sigma_stats", {})
    print(f"[{task}] head loaded; saved val σ̄={saved.get('mean', float('nan')):.4f} "
          f"(loader-validation target)")

    features = np.load(f"{cfg['cache_dir']}/features.npy", mmap_mode="r")
    goals = np.load(f"{cfg['cache_dir']}/goals.npy")
    with h5py.File(cfg["h5"], "r") as f:
        ep_offset = f["ep_offset"][:].astype(np.int64)
        ep_len = f["ep_len"][:].astype(np.int64)

    lo, hi = cfg["val_ep_range"]
    starts, eps = build_window_starts(ep_offset, ep_len, lo, hi)
    idx = rng.choice(len(starts), size=min(M_SAMPLES, len(starts)), replace=False)
    starts, eps = starts[idx], eps[idx]

    z_t = torch.from_numpy(np.ascontiguousarray(features[starts]).astype(np.float32))
    z_g_own = torch.from_numpy(goals[eps].astype(np.float32))

    # B: goal from a different val episode (guaranteed != own)
    other = rng.integers(lo, hi, size=len(eps))
    clash = other == eps
    other[clash] = lo + (other[clash] - lo + 1) % (hi - lo)
    z_g_mis = torch.from_numpy(goals[other].astype(np.float32))

    # C: moment-matched Gaussian noise embeddings
    mu_f, sd_f = z_t.mean(0, keepdim=True), z_t.std(0, keepdim=True)
    z_t_noise = mu_f + sd_f * torch.randn_like(z_t)

    conds = {
        "A_in_dist": (z_t, z_g_own),
        "B_mismatch_goal": (z_t, z_g_mis),
        "C_noise_z": (z_t_noise, z_g_own),
    }
    rows = []
    ref = None
    for name, (zt, zg) in conds.items():
        st = sigma_stats(head, zt, zg)
        if ref is None:
            ref = st["sigma_mean"]
        ratio = st["sigma_mean"] / ref
        rows.append((task, name, st["sigma_mean"], st["sigma_median"],
                     st["sigma_p90"], ratio, st["pog_weight_mean"]))
        print(f"  {name:<16} σ̄={st['sigma_mean']:.4f}  med={st['sigma_median']:.4f}  "
              f"p90={st['sigma_p90']:.4f}  ratio={ratio:.2f}x  "
              f"PoG-weight={st['pog_weight_mean']:.3f}")
    return rows


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    all_rows = []
    for task, cfg in TASKS.items():
        all_rows += run_task(task, cfg, rng)
    with open(OUT_TSV, "w") as f:
        f.write("task\tcondition\tsigma_mean\tsigma_median\tsigma_p90\tratio_vs_indist\tpog_weight_mean\n")
        for r in all_rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"\n[done] wrote {OUT_TSV}")


if __name__ == "__main__":
    main()
