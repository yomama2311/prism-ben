"""Ensemble σ-OOD calibration study for CoRL rebuttal §4 — companion to
analyze_sigma_ood.py (which showed the SINGLE head's σ is aleatoric-only and
does NOT inflate on OOD inputs).

Ensemble of E=5 heads (seeds 0-4, identical recipe). Mixture moments:
    μ_ens  = mean_i μ_i
    σ_ens² = mean_i σ_i²  +  var_i μ_i
             \_aleatoric_/    \_epistemic_/
The epistemic term (head disagreement) is what should grow off-distribution.

Same three conditions as the single-head study:
  A in_dist / B mismatch_goal / C noise_z
Reported per condition: total σ̄_ens, aleatoric/epistemic split, ratio vs
in-dist, and PoG weight at s=1, σ_π=1.

Run:  python analyze_sigma_ood_ensemble.py
Out:  logs/sigma_ood_ensemble_results.tsv
"""
from __future__ import annotations

import os

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from prior_head import PriorHead

STABLE_HOME = "/home/yuhai/project/prism-jepa/.stable-wm"
OUT_TSV = "/home/yuhai/project/prism-jepa/logs/sigma_ood_ensemble_results.tsv"
M_SAMPLES = 20_000
SEED = 0
ENS_SEEDS = [0, 1, 2, 3, 4]

TASKS = {
    "pusht": dict(
        h5=f"{STABLE_HOME}/pusht_expert_train.h5",
        cache_dir=f"{STABLE_HOME}/jepa_cache/pusht",
        ckpt_fmt="/home/yuhai/project/prism-jepa/prior_head_pusht_ens{}.pt",
        val_ep_range=(16817, 18685),
    ),
    "cube": dict(
        h5=f"{STABLE_HOME}/ogbench/cube_single_expert.h5",
        cache_dir=f"{STABLE_HOME}/jepa_cache/cube",
        ckpt_fmt="/home/yuhai/project/prism-jepa/prior_head_cube_ens{}.pt",
        val_ep_range=(9000, 10000),
    ),
}

WIN_LEN = 25


def build_window_starts(ep_offset, ep_len, ep_lo, ep_hi, win_len=WIN_LEN):
    starts, eps = [], []
    for e in range(ep_lo, ep_hi):
        max_off = int(ep_len[e]) - win_len
        if max_off <= 0:
            continue
        starts.append(int(ep_offset[e]) + np.arange(max_off))
        eps.append(np.full(max_off, e))
    return np.concatenate(starts), np.concatenate(eps)


def load_ensemble(ckpt_fmt):
    heads = []
    for s in ENS_SEEDS:
        ckpt = torch.load(ckpt_fmt.format(s), map_location="cpu", weights_only=False)
        hp = ckpt["config"]
        h = PriorHead(z_dim=hp["z_dim"], H=hp["H"], A_block=hp["A_block"],
                      A_raw=hp["A_raw"], hidden=hp["hidden"],
                      sigma_floor=hp["sigma_floor"])
        h.load_state_dict(ckpt["state_dict"])
        h.eval()
        heads.append(h)
    return heads


@torch.no_grad()
def ens_stats(heads, z_t, z_g, batch=4096):
    tot_all, alea_all, epi_all = [], [], []
    for s in range(0, len(z_t), batch):
        mus, vars_ = [], []
        for h in heads:
            mu, sig = h(z_t[s : s + batch], z_g[s : s + batch])
            mus.append(mu.reshape(mu.shape[0], -1))
            vars_.append(sig.reshape(sig.shape[0], -1).pow(2))
        MU = torch.stack(mus)          # (E, B, D_a)
        VAR = torch.stack(vars_)       # (E, B, D_a)
        alea = VAR.mean(0)             # mean_i σ_i²
        epi = MU.var(0, unbiased=False)  # var_i μ_i
        tot_all.append(alea + epi)
        alea_all.append(alea)
        epi_all.append(epi)
    tot = torch.cat(tot_all)
    alea = torch.cat(alea_all)
    epi = torch.cat(epi_all)
    sig_ens = tot.sqrt()
    w = 1.0 / (1.0 + tot)  # PoG weight at s=1, σ_π=1
    return dict(
        sigma_mean=sig_ens.mean().item(),
        sigma_median=sig_ens.median().item(),
        alea_sigma_mean=alea.sqrt().mean().item(),
        epi_sigma_mean=epi.sqrt().mean().item(),
        pog_weight_mean=w.mean().item(),
    )


def run_task(task, cfg, rng):
    print(f"\n================ {task} (E={len(ENS_SEEDS)}) ================")
    heads = load_ensemble(cfg["ckpt_fmt"])

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

    other = rng.integers(lo, hi, size=len(eps))
    clash = other == eps
    other[clash] = lo + (other[clash] - lo + 1) % (hi - lo)
    z_g_mis = torch.from_numpy(goals[other].astype(np.float32))

    mu_f, sd_f = z_t.mean(0, keepdim=True), z_t.std(0, keepdim=True)
    z_t_noise = mu_f + sd_f * torch.randn_like(z_t)

    conds = {
        "A_in_dist": (z_t, z_g_own),
        "B_mismatch_goal": (z_t, z_g_mis),
        "C_noise_z": (z_t_noise, z_g_own),
    }
    rows, ref = [], None
    for name, (zt, zg) in conds.items():
        st = ens_stats(heads, zt, zg)
        if ref is None:
            ref = st["sigma_mean"]
        ratio = st["sigma_mean"] / ref
        rows.append((task, name, st["sigma_mean"], st["sigma_median"],
                     st["alea_sigma_mean"], st["epi_sigma_mean"], ratio,
                     st["pog_weight_mean"]))
        print(f"  {name:<16} σ̄_ens={st['sigma_mean']:.4f}  "
              f"[alea {st['alea_sigma_mean']:.4f} | epi {st['epi_sigma_mean']:.4f}]  "
              f"ratio={ratio:.2f}x  PoG-w={st['pog_weight_mean']:.3f}")
    return rows


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    all_rows = []
    for task, cfg in TASKS.items():
        all_rows += run_task(task, cfg, rng)
    with open(OUT_TSV, "w") as f:
        f.write("task\tcondition\tsigma_ens_mean\tsigma_ens_median\t"
                "aleatoric_sigma_mean\tepistemic_sigma_mean\tratio_vs_indist\tpog_weight_mean\n")
        for r in all_rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"\n[done] wrote {OUT_TSV}")


if __name__ == "__main__":
    main()
