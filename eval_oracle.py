"""
PRISM-LeWM Oracle Probe.

Identical to eval.py except:
  - Uses PrismMPPISolver (with PoG fusion) in place of MPPISolver
  - Uses OracleWorldModelPolicy in place of WorldModelPolicy
  - Oracle prior comes from KNN lookup against expert dataset

Usage:
    python eval_oracle.py policy=pusht/lewm solver=mppi \\
        ++oracle.sigma=0.1 ++oracle.dataset_h5=/path/to/pusht_expert_train.h5

The probe answers: does PoG-fused MPPI sampling improve sample efficiency
under LeWM's goal-conditioned planning, given a near-perfect (oracle) prior?
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import time
import sys
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from prism_mppi import PrismMPPISolver  # noqa: E402
from oracle_policy import OracleWorldModelPolicy  # noqa: E402


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


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
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Eval LeWM with PRISM-MPPI + oracle action prior."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # ---- Oracle config (with defaults) ----
    oracle_cfg = cfg.get("oracle", {})
    oracle_sigma = oracle_cfg.get("sigma", 0.1)
    oracle_match_key = oracle_cfg.get("match_key", "state")
    dataset_h5 = oracle_cfg.get(
        "dataset_h5",
        str(Path(swm.data.utils.get_cache_dir()) / f"{cfg.eval.dataset_name}.h5"),
    )
    oracle_max_rows = oracle_cfg.get("max_rows", 200_000)
    print(f"\n[Oracle config]")
    print(f"  sigma         = {oracle_sigma}")
    print(f"  match_key     = {oracle_match_key}")
    print(f"  dataset_h5    = {dataset_h5}")
    print(f"  max_rows      = {oracle_max_rows}")
    print()

    # ---- Build env / world ----
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    process = {
        "action": preprocessing.MinMaxScaler(feature_range=(-1, 1)),
    }
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process["action"].fit(dataset.get_col_data("action"))
    ep_indices = np.unique(
        dataset.get_col_data(
            "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        )
    )

    # ---- Build model + PRISM-MPPI solver + Oracle policy ----
    assert cfg.policy != "random", "Oracle probe requires a real model checkpoint"
    model = swm.policy.AutoCostModel(cfg.policy)
    model = model.to("cuda")
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    config = swm.PlanConfig(**cfg.plan_config)

    # Force MPPI solver — override hydra config if needed to use PrismMPPISolver
    # Solver yaml's _target_ should be MPPISolver; we instantiate PrismMPPISolver
    # directly here with the same kwargs.
    solver_kwargs = OmegaConf.to_container(cfg.solver, resolve=True)
    solver_kwargs.pop("_target_", None)
    solver_kwargs.pop("model", None)  # we pass it directly
    solver = PrismMPPISolver(model=model, **solver_kwargs)

    policy = OracleWorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
        dataset_h5_path=dataset_h5,
        dataset_max_rows=oracle_max_rows,
        oracle_sigma=oracle_sigma,
        match_key=oracle_match_key,
        verbose=True,
    )

    results_path = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent

    # ---- Sample eval episodes (same logic as eval.py) ----
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

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
    end_time = time.time()

    print(f"\n=== ORACLE PRISM-MPPI RESULT ===")
    print(f"success_rate: {metrics['success_rate']}")
    print(f"wall time: {end_time - start_time:.1f}s")

    out_path = results_path / "pusht_oracle_results.txt"
    with out_path.open("a") as f:
        f.write("\n==== ORACLE PROBE RUN ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write(f"\nsuccess_rate: {metrics['success_rate']}\n")
        f.write(f"wall_time_s: {end_time - start_time:.1f}\n")


if __name__ == "__main__":
    run()
