"""Debug: isolate which part of OracleWorldModelPolicy breaks SR."""
import os
os.environ["MUJOCO_GL"] = "egl"

import sys
sys.path.insert(0, "/home/yuhai/project/lewm-prism")

import time
import hydra
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
import numpy as np
import stable_worldmodel as swm
from sklearn import preprocessing
import stable_pretraining as spt
import torch
from torchvision.transforms import v2 as transforms
from prism_mppi import PrismMPPISolver
from oracle_policy import OracleWorldModelPolicy


def img_t(cfg):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def run_eval(policy_cls, label, **policy_kwargs):
    """Run eval with a given policy class. Returns SR%."""
    print(f"\n{'='*60}\n  RUN: {label}\n{'='*60}")
    with initialize_config_dir(
        config_dir="/home/yuhai/project/lewm-prism/config/eval",
        version_base=None,
    ):
        cfg = compose(
            config_name="pusht",
            overrides=["policy=pusht/lewm", "solver=mppi", "eval.num_eval=5"],
        )

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    process = {"action": preprocessing.MinMaxScaler(feature_range=(-1, 1))}
    trf = {"pixels": img_t(cfg), "goal": img_t(cfg)}
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir="/home/yuhai/project/lewm-prism/.stable-wm",
    )
    process["action"].fit(dataset.get_col_data("action"))

    model = swm.policy.AutoCostModel(cfg.policy).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    config = swm.PlanConfig(**cfg.plan_config)

    solver_kwargs = OmegaConf.to_container(cfg.solver, resolve=True)
    solver_kwargs.pop("_target_", None)
    solver_kwargs.pop("model", None)

    if policy_cls is swm.policy.WorldModelPolicy:
        # Stock policy with stock MPPI solver (control)
        from stable_worldmodel.solver import MPPISolver
        solver = MPPISolver(model=model, **solver_kwargs)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=trf,
        )
    else:
        # PRISM solver + Oracle policy
        solver = PrismMPPISolver(model=model, **solver_kwargs)
        policy = policy_cls(
            solver=solver, config=config, process=process, transform=trf,
            **policy_kwargs,
        )

    # Episode sampling
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices = np.unique(dataset.get_col_data(col))
    episode_idx = dataset.get_col_data(col)
    step_idx = dataset.get_col_data("step_idx")
    episode_len = np.array(
        [np.max(step_idx[episode_idx == e]) + 1 for e in ep_indices]
    )
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_d = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_per_row = np.array([max_d[e] for e in episode_idx])
    valid = np.nonzero(step_idx <= max_per_row)[0]
    g = np.random.default_rng(cfg.seed)
    rei = np.sort(valid[g.choice(len(valid) - 1, size=cfg.eval.num_eval, replace=False)])
    eval_episodes = dataset.get_row_data(rei)[col]
    eval_start_idx = dataset.get_row_data(rei)["step_idx"]

    world.set_policy(policy)
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path="/tmp/debug_oracle",
    )
    sr = metrics["success_rate"]
    print(f"\n  RESULT: {label} → SR = {sr}%")
    return sr


# ===== Run battery of tests =====

# Test 1: stock WorldModelPolicy + stock MPPISolver — expected 80%
sr1 = run_eval(swm.policy.WorldModelPolicy, "Stock WMP + stock MPPI")

# Test 2: OracleWorldModelPolicy + PrismMPPISolver, set_prior fired but fusion DISABLED
class OracleNoFusion(OracleWorldModelPolicy):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.solver._disable_fusion = True

sr2 = run_eval(
    OracleNoFusion, "OracleWMP + PrismMPPI, FUSION DISABLED",
    dataset_h5_path="/home/yuhai/project/lewm-prism/.stable-wm/pusht_expert_train.h5",
    dataset_max_rows=200_000, oracle_sigma=0.1, match_key="state", verbose=False,
)

# Test 3: OracleWorldModelPolicy + PrismMPPISolver, fusion ON, sigma huge (no influence)
sr3 = run_eval(
    OracleWorldModelPolicy, "OracleWMP + PrismMPPI, sigma=10000 (no influence)",
    dataset_h5_path="/home/yuhai/project/lewm-prism/.stable-wm/pusht_expert_train.h5",
    dataset_max_rows=200_000, oracle_sigma=10000.0, match_key="state", verbose=False,
)

print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
print(f"  Stock WMP + stock MPPI:        {sr1:.0f}%")
print(f"  OracleWMP + PRISM, NO FUSION:   {sr2:.0f}%")
print(f"  OracleWMP + PRISM, sigma=10000: {sr3:.0f}%")
print(f"\n  Bug isolation:")
print(f"    if 1 == 2 → bug is in PoG fusion code")
print(f"    if 1 != 2 → bug is in OracleWMP override")
