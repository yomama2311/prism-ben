"""eval_replay_oracle.py — protocol-ceiling control for the CoRL rebuttal.

Question: what does this evaluation protocol actually measure?

`swm.World.evaluate_from_dataset` starts each episode at a row (ep, t) taken
from the dataset and sets the goal to that same episode's state at t+25. So the
*logged* action sequence a[t : t+25] reaches the goal by construction. This
script measures that ceiling empirically: a policy that simply replays the
logged actions, with no model, no policy network, and no planning.

If replay scores ~100%, then any method that can reproduce the demonstration
distribution on in-distribution starts (e.g. a 19.3M diffusion policy trained
on 98% of these episodes) can approach the ceiling, and the protocol cannot
separate "learned the dynamics" from "learned the demonstrations" in this
regime. Comparisons within the WM-planning family (all of which see the same
protocol) remain informative.

Usage:
    python eval_replay_oracle.py --config-name=pusht policy=pusht/lewm \
        solver=mppi eval.num_eval=50 seed=0
"""
import os

os.environ["MUJOCO_GL"] = "egl"

import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from stable_worldmodel.policy import WorldModelPolicy
from torchvision.transforms import v2 as transforms

sys.path.insert(0, str(Path(__file__).parent))
from prism_mppi import PrismMPPISolver  # noqa: E402

H5_BY_TASK = {
    "pusht": "pusht_expert_train.h5",
    "cube": "ogbench/cube_single_expert.h5",
}


class ReplayPolicy(WorldModelPolicy):
    """Replays the dataset's logged actions for each env's (episode, start)."""

    def __init__(self, solver, config, process=None, transform=None,
                 actions=None, ep_offset=None, eval_episodes=None,
                 eval_start_idx=None, **kwargs):
        super().__init__(solver=solver, config=config, process=process,
                         transform=transform, **kwargs)
        self._actions = actions          # (N_frames, A) raw env units
        self._ep_offset = ep_offset
        self._eps = np.asarray(eval_episodes)
        self._starts = np.asarray(eval_start_idx)
        self._k = 0                      # env-step counter
        print(f"[Replay] {len(self._eps)} envs; first (ep,start)="
              f"({self._eps[0]}, {self._starts[0]})")

    def reset_goal_cache(self):
        self._k = 0

    def get_action(self, info_dict: dict, **kwargs):
        rows = self._ep_offset[self._eps] + self._starts + self._k
        rows = np.clip(rows, 0, len(self._actions) - 1)
        act = self._actions[rows]                     # (n_envs, A) raw units
        act = np.nan_to_num(act, nan=0.0)
        self._k += 1
        return act.reshape(*self.env.action_space.shape)


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


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    task = str(cfg.policy).split("/")[0]
    h5_path = Path(swm.data.utils.get_cache_dir()) / H5_BY_TASK[task]
    print(f"\n{'='*60}\n  REPLAY-ORACLE protocol ceiling — task={task}\n"
          f"  h5={h5_path}\n  num_eval={cfg.eval.num_eval}  seed={cfg.seed}\n{'='*60}\n")

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name, keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=Path(cfg.cache_dir or swm.data.utils.get_cache_dir()))

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        p = preprocessing.StandardScaler()
        d = dataset.get_col_data(col)
        p.fit(d[~np.isnan(d).any(axis=1)])
        process[col] = p
        if col != "action":
            process[f"goal_{col}"] = p

    model = swm.policy.AutoCostModel(cfg.policy).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    solver_kwargs = OmegaConf.to_container(cfg.solver, resolve=True)
    solver_kwargs.pop("_target_", None); solver_kwargs.pop("model", None)
    solver = PrismMPPISolver(model=model, **solver_kwargs)

    with h5py.File(h5_path, "r") as f:
        actions = f["action"][:].astype(np.float32)
        ep_offset = f["ep_offset"][:].astype(np.int64)

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices = np.unique(dataset.get_col_data(col_name))
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start = {e: episode_len[i] - cfg.eval.goal_offset_steps - 1
                 for i, e in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start[e] for e in dataset.get_col_data(col_name)])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    g = np.random.default_rng(cfg.seed)
    idx = g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    idx = np.sort(valid_indices[idx])
    eval_episodes = dataset.get_row_data(idx)[col_name]
    eval_start_idx = dataset.get_row_data(idx)["step_idx"]

    policy = ReplayPolicy(solver=solver, config=swm.PlanConfig(**cfg.plan_config),
                          process=process, transform=transform,
                          actions=actions, ep_offset=ep_offset,
                          eval_episodes=eval_episodes, eval_start_idx=eval_start_idx)
    world.set_policy(policy)

    t0 = time.time()
    metrics = world.evaluate_from_dataset(
        dataset, start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=Path(swm.data.utils.get_cache_dir(), cfg.policy).parent)

    print(f"\n=== REPLAY-ORACLE RESULT ===")
    print(f"success_rate: {metrics['success_rate']}")
    print(f"wall time:    {time.time() - t0:.1f}s   n_eval: {cfg.eval.num_eval}")


if __name__ == "__main__":
    run()
