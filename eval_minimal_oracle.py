"""Minimal change to eval.py: ONLY swap WorldModelPolicy → OracleWorldModelPolicy.
Same solver (stock MPPI). If SR drops vs eval.py, bug is in OracleWorldModelPolicy."""

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

sys.path.insert(0, str(Path(__file__).parent))
# Pure inline subclass, NO imports from oracle_policy — to isolate any side effects
class OracleWorldModelPolicy(swm.policy.WorldModelPolicy):
    """Pure no-op subclass."""
    pass


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
    return np.array([np.max(step_idx[episode_idx == e]) + 1 for e in episodes])


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name, keys_to_cache=cfg.dataset.keys_to_cache, cache_dir=dataset_path,
    )


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    process = {"action": preprocessing.MinMaxScaler(feature_range=(-1, 1))}
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process["action"].fit(dataset.get_col_data("action"))

    ep_indices = np.unique(dataset.get_col_data(
        "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ))

    model = swm.policy.AutoCostModel(cfg.policy)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)  # ← STOCK MPPISolver

    # KEY DIFFERENCE: use OracleWorldModelPolicy, but it'll fail set_prior assertion
    # with stock MPPISolver. So we monkey-patch a no-op set_prior on the solver.
    if not hasattr(solver, 'set_prior'):
        solver.set_prior = lambda *a, **kw: None

    policy = OracleWorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform,
    )
    results_path = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent

    # ---- (rest copied verbatim from eval.py) ----
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]
    world.set_policy(policy)

    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
    )
    print(f"\n=== ORACLE-WMP + STOCK MPPI ===")
    print(f"success_rate: {metrics['success_rate']}")


if __name__ == "__main__":
    run()
