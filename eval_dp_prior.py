"""eval_dp_prior.py — rebuttal §3 hybrid entry. DP-guided planning baselines.

Mirrors eval_prism_head.py byte-for-byte on preprocessing/world/eval logic
(Critical Lesson #1: StandardScaler on ALL non-pixel columns). The only change
is the policy: DPPriorPolicy sources the planner prior from the paper's own
Diffusion Policy checkpoint instead of the PriorHead.

Usage:
    python eval_dp_prior.py --config-name=pusht policy=pusht/lewm solver=mppi \\
        solver.num_samples=128 +dpprior.injection_mode=pog \\
        +dpprior.ckpt=dp_baseline_pusht.pt eval.num_eval=50 seed=0
    # injection_mode ∈ {best_of_n, replace, pog}
"""
import os

os.environ["MUJOCO_GL"] = "egl"

import sys
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

sys.path.insert(0, str(Path(__file__).parent))
from prism_mppi import PrismMPPISolver  # noqa: E402
from dp_prior_policy import DPPriorPolicy  # noqa: E402


def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name, keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


@hydra.main(version_base=None, config_path="./config/eval", config_name="cube")
def run(cfg: DictConfig):
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    dp_cfg = cfg.get("dpprior", {})
    injection_mode = dp_cfg.get("injection_mode", "pog")
    n_dp_samples = dp_cfg.get("n_dp_samples", 16)
    sigma_scale = dp_cfg.get("sigma_scale", 1.0)
    sigma_clamp_max = float(dp_cfg.get("sigma_clamp_max", float("inf")))
    dp_ckpt = dp_cfg.get("ckpt", None)
    assert dp_ckpt is not None, "Pass +dpprior.ckpt=<dp_baseline_<task>.pt>"

    print("\n" + "=" * 64)
    print(f"  DP-prior hybrid eval — MODE = {injection_mode!r}")
    print(f"  n_dp_samples   = {n_dp_samples}")
    print(f"  sigma_scale    = {sigma_scale}")
    print(f"  sigma_clamp_max= {sigma_clamp_max}")
    print(f"  dp_ckpt        = {dp_ckpt}")
    print(f"  num_eval       = {cfg.eval.num_eval}")
    print(f"  policy         = {cfg.policy}")
    print(f"  seed           = {cfg.seed}")
    print("=" * 64 + "\n")

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    dataset = get_dataset(cfg, cfg.eval.dataset_name)

    # CRITICAL — match eval.py's StandardScaler over all non-pixel cols.
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ("pixels",):
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    assert cfg.policy != "random", "Hybrid probe requires a real model checkpoint"
    model = swm.policy.AutoCostModel(cfg.policy).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    plan_config = swm.PlanConfig(**cfg.plan_config)
    solver_kwargs = OmegaConf.to_container(cfg.solver, resolve=True)
    solver_kwargs.pop("_target_", None)
    solver_kwargs.pop("model", None)
    solver = PrismMPPISolver(model=model, **solver_kwargs)

    policy = DPPriorPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform=transform,
        dp_ckpt=dp_ckpt,
        injection_mode=injection_mode,
        n_dp_samples=n_dp_samples,
        sigma_scale=sigma_scale,
        sigma_clamp_max=sigma_clamp_max,
        device="cuda",
        verbose=True,
    )

    results_path = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent

    ep_indices = np.unique(
        dataset.get_col_data(
            "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        )
    )
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
    )
    elapsed = time.time() - start_time

    sr = metrics["success_rate"]
    dp_ms = np.array(policy.dp_ms) if policy.dp_ms else np.array([float("nan")])
    print(f"\n=== DP-PRIOR RESULT ({injection_mode}, N={n_dp_samples}) ===")
    print(f"success_rate: {sr}")
    print(f"dp_sample_ms_per_replan: mean={np.nanmean(dp_ms):.1f}  "
          f"p50={np.nanmedian(dp_ms):.1f}  n_replans={len(policy.dp_ms)}")
    print(f"wall time:    {elapsed:.1f}s")
    print(f"n_eval:       {cfg.eval.num_eval}")


if __name__ == "__main__":
    run()
