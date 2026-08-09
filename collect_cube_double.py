"""Collect a pixel dataset for OGBench cube-double using the official expert oracle.

Mirrors the cube-single dataset's construction (swm.World.record_dataset with
ExpertPolicy); only env_type changes single -> double.

Usage: python collect_cube_double.py --episodes 5 --name cube_double_smoke
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse, time
from pathlib import Path
import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench.expert_policy import ExpertPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=5)
ap.add_argument("--num-envs", type=int, default=16)
ap.add_argument("--name", default="cube_double_smoke")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

t0 = time.time()
world = swm.World(
    env_name="swm/OGBCube-v0", num_envs=a.num_envs, max_episode_steps=201,
    history_size=1, frame_skip=1, env_type="double", ob_type="states",
    multiview=False, width=224, height=224, visualize_info=False,
    terminate_at_goal=False, image_shape=(224, 224), mode='data_collection',
)
print(f"[collect] world up in {time.time()-t0:.1f}s  action_space={world.envs.action_space}")
world.set_policy(ExpertPolicy(policy_type="markov_oracle", action_noise=0.1, seed=a.seed))
t1 = time.time()
world.record_dataset(a.name, episodes=a.episodes, seed=a.seed,
                     cache_dir=Path(os.environ["STABLEWM_HOME"]) / "ogbench")
dt = time.time() - t1
p = Path(os.environ["STABLEWM_HOME"]) / "ogbench" / f"{a.name}.h5"
sz = p.stat().st_size / 1e9 if p.exists() else float("nan")
print(f"[collect] {a.episodes} episodes in {dt:.1f}s  ({dt/max(a.episodes,1):.2f}s/ep)  size={sz:.2f}GB")
print(f"[collect] projected: 2000 ep -> {dt/max(a.episodes,1)*2000/60:.0f} min, {sz/max(a.episodes,1)*2000:.0f} GB")
