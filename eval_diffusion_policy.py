"""eval_diffusion_policy.py — Diffusion Policy baseline eval entry point.

Mirrors eval_prism_head.py's StandardScaler / transform / world setup so DP
runs against the same eval episodes as PRISM-MPPI / PRISM-CEM. The only
new pieces are DiffusionPolicyPolicy and the ``+dp.*`` config namespace.

Compared to eval_prism_head.py:
  - No JEPA object loaded (DP has its own vision encoder)
  - No head ckpt (DP has its own model in ``dp_baseline_<task>.pt``)
  - No injection_mode / sigma_scale knobs (DP is a policy, not a prior)
  - Solver is constructed but never invoked — kept only to satisfy
    WorldModelPolicy's parent init

Usage:
    # PushT
    python eval_diffusion_policy.py --config-name=pusht policy=pusht/lewm \\
        +dp.ckpt=/home/yuhai/project/prism-jepa/dp_baseline_pusht.pt \\
        eval.num_eval=50 seed=0

    # Cube
    python eval_diffusion_policy.py --config-name=cube policy=cube/lewm \\
        +dp.ckpt=/home/yuhai/project/prism-jepa/dp_baseline_cube.pt \\
        eval.num_eval=50 seed=0
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
from dp_baseline.policy import DiffusionPolicyPolicy  # noqa: E402


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
    # ---- DP config ----
    dp_cfg = cfg.get("dp", {})
    ckpt_path = dp_cfg.get("ckpt", None)
    assert ckpt_path is not None, (
        "Must pass +dp.ckpt=/path/to/dp_baseline_<task>.pt"
    )
    n_action_steps = int(dp_cfg.get("n_action_steps", 8))
    num_inference_steps = int(dp_cfg.get("num_inference_steps", 10))
    use_ema = bool(dp_cfg.get("use_ema", True))

    print("\n" + "=" * 64)
    print(f"  Diffusion Policy eval")
    print(f"  ckpt                = {ckpt_path}")
    print(f"  n_action_steps      = {n_action_steps}")
    print(f"  num_inference_steps = {num_inference_steps}")
    print(f"  use_ema             = {use_ema}")
    print(f"  num_eval            = {cfg.eval.num_eval}")
    print(f"  seed                = {cfg.seed}")
    print("=" * 64 + "\n")

    # ---- Env / dataset (same as eval_prism_head.py) ----
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {
        "pixels": img_transform(cfg),
        "goal":   img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        proc = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        proc.fit(col_data)
        process[col] = proc
        if col != "action":
            process[f"goal_{col}"] = proc

    # ---- Build a placeholder solver (DP never invokes it) ----
    # We still need it so WorldModelPolicy.__init__ doesn't choke.
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    assert cfg.policy != "random", "DP eval requires a real WM ckpt for the env config"
    model = swm.policy.AutoCostModel(cfg.policy).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    plan_config = swm.PlanConfig(**cfg.plan_config)
    solver_kwargs = OmegaConf.to_container(cfg.solver, resolve=True)
    solver_kwargs.pop("_target_", None)
    solver_kwargs.pop("model", None)
    # Default to vanilla MPPISolver as the dummy. We never call solve().
    from stable_worldmodel.solver import MPPISolver
    solver_kwargs.pop("temperature", None)
    solver_kwargs.pop("topk", None)
    placeholder_solver = MPPISolver(model=model, **solver_kwargs)

    # ---- Build DP policy ----
    policy = DiffusionPolicyPolicy(
        solver=placeholder_solver,
        config=plan_config,
        process=process,
        transform=transform,
        ckpt_path=ckpt_path,
        n_action_steps=n_action_steps,
        num_inference_steps=num_inference_steps,
        use_ema=use_ema,
        device="cuda",
        verbose=True,
    )

    results_path = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent

    # ---- Sample eval episodes (identical logic to eval_prism_head.py) ----
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices = np.unique(dataset.get_col_data(col_name))
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_dict = {ep: max_start_idx[i] for i, ep in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    # Rebuttal: optionally restrict eval to DP's held-out (never-trained)
    # episodes — same split formula as DPDataset (perm seed 42, val_ratio 0.02).
    if dp_cfg.get("eval_ep_filter", None) == "dp_val":
        ep_col_all = dataset.get_col_data(col_name)
        n_eps_total = len(np.unique(ep_col_all))
        rng_split = np.random.default_rng(42)
        n_val = max(1, int(n_eps_total * 0.02))
        val_set = rng_split.permutation(n_eps_total)[:n_val]
        valid_mask = valid_mask & np.isin(ep_col_all, val_set)
        print(f"[dp_val filter] eval restricted to DP's {n_val} held-out episodes; "
              f"valid start rows: {int(valid_mask.sum())}")
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_idx = g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    random_idx = np.sort(valid_indices[random_idx])
    eval_episodes = dataset.get_row_data(random_idx)[col_name]
    eval_start_idx = dataset.get_row_data(random_idx)["step_idx"]

    world.set_policy(policy)

    start = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
    )
    wall = time.time() - start

    print(f"\n=== Diffusion Policy eval result ===")
    print(f"success_rate: {metrics['success_rate']}")
    print(f"wall time:    {wall:.1f}s")

    out_path = results_path / "dp_results.txt"
    with out_path.open("a") as f:
        f.write("\n==== Diffusion Policy ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write(f"\nsuccess_rate: {metrics['success_rate']}\n")
        f.write(f"wall_time_s: {wall:.1f}\n")
        f.write(f"ckpt: {ckpt_path}\n")
        f.write(f"n_action_steps: {n_action_steps}\n")
        f.write(f"num_inference_steps: {num_inference_steps}\n")


if __name__ == "__main__":
    run()
