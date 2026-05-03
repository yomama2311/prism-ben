"""Phase C self-check.

Exercises the PrismHeadPolicy + PrismMPPISolver hook for each injection_mode
without invoking the world model / cost / actual eval. Verifies:

  - Head loads, runs forward with correct output shapes.
  - JEPA encoder runs on synthetic pixels with correct output shape.
  - injection_mode='none'       → solver._prior_mean is None after the policy
                                   does its pre-work.
  - injection_mode='warm_start' → solver has μ_prior set, σ_prior set, but
                                   the warm_start branch in init_action_distrib
                                   IGNORES σ_prior (returned std == default).
  - injection_mode='pog'        → solver applies PoG fusion; fused std differs
                                   from default.
  - injection_mode='bc_only'    → bypass path; returns numpy action of shape
                                   env.action_space.shape (n_envs, A_raw).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Box
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from prism_mppi import PrismMPPISolver  # noqa: E402
from prism_head_policy import PrismHeadPolicy  # noqa: E402


HEAD_CKPT = "/home/yuhai/project/prism-jepa/prior_head_cube.pt"
JEPA_OBJ = "/home/yuhai/project/prism-jepa/.stable-wm/cube/lewm_object.ckpt"
N_ENVS = 2  # synthetic batch size
H_PX = W_PX = 224


class _FakeEnv:
    """Minimal env stub for set_env-equivalent state."""
    def __init__(self, num_envs, action_space):
        self.num_envs = num_envs
        self.action_space = action_space


class _FakeCostModel:
    """Stub model — we never call it. Solver only needs it for solve()."""
    def get_cost(self, *a, **k):
        raise RuntimeError("self-check should not invoke the cost model")


class _PlanConfig:
    horizon = 5
    receding_horizon = 5
    action_block = 5
    warm_start = False  # field on swm.PlanConfig used by upstream policy logic


def make_fake_info_dict(n_envs=N_ENVS):
    """Return an info_dict with synthetic uint8 pixels and goal (n_envs, H, W, 3)."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (n_envs, H_PX, W_PX, 3), dtype=np.uint8)
    goal = rng.integers(0, 255, (n_envs, H_PX, W_PX, 3), dtype=np.uint8)
    return {"pixels": pixels, "goal": goal}


def make_action_scaler(A_raw=5):
    """Fit a dummy StandardScaler so inverse_transform is well-defined."""
    rng = np.random.default_rng(1)
    return StandardScaler().fit(rng.normal(size=(1000, A_raw)))


def setup_policy(mode: str, sigma_scale: float = 1.0):
    """Build a PrismHeadPolicy with a configured solver and a fake env."""
    A_raw = 5  # cube
    action_space = Box(low=-1, high=1, shape=(N_ENVS, A_raw), dtype=np.float32)

    solver = PrismMPPISolver(
        model=_FakeCostModel(),
        batch_size=1, num_samples=8, n_steps=1, topk=4, temperature=0.5,
        var_scale=1.0, device="cuda", seed=42,
    )
    solver.configure(action_space=action_space, n_envs=N_ENVS, config=_PlanConfig)

    process = {"action": make_action_scaler(A_raw)}

    policy = PrismHeadPolicy(
        solver=solver, config=_PlanConfig, process=process, transform=None,
        head_ckpt=HEAD_CKPT, jepa_object_path=JEPA_OBJ,
        injection_mode=mode, sigma_scale=sigma_scale,
        device="cuda", verbose=False,
    )
    # Mimic set_env so policy / solver are wired up.
    policy.env = _FakeEnv(N_ENVS, action_space)
    from collections import deque
    policy._action_buffer = deque(maxlen=policy.flatten_receding_horizon)

    return policy


def run_check_planner_modes():
    info = make_fake_info_dict()
    print("\n[selfcheck] === planner-using modes (none / warm_start / pog) ===")

    for mode in ("none", "warm_start", "pog"):
        policy = setup_policy(mode, sigma_scale=1.0)

        # Reproduce what get_action does up to the point of set_prior, but skip
        # the actual solver call (we're not running the cost model).
        z_t = policy._compute_z_t(info)
        policy._ensure_goal_z(info)
        mu, sigma = policy._run_head(z_t, policy._goal_z)

        assert z_t.shape == (N_ENVS, 192), z_t.shape
        assert policy._goal_z.shape == (N_ENVS, 192), policy._goal_z.shape
        # μ, σ must be (n_envs, H, A_block * A_raw) = (N_ENVS, 5, 25)
        assert mu.shape == (N_ENVS, 5, 25), mu.shape
        assert sigma.shape == (N_ENVS, 5, 25), sigma.shape
        assert (sigma >= policy.head.sigma_floor).all(), "σ floor violated"

        if mode == "none":
            policy.solver.clear_prior()
            assert policy.solver._prior_mean is None, \
                f"mode=none must leave solver._prior_mean=None; got tensor"
            assert policy.solver._prior_sigma is None
        else:
            policy.solver.set_prior(mu, sigma)
            assert policy.solver._prior_mean is not None
            assert policy.solver._prior_sigma is not None
            assert policy.solver._prior_mean.shape == mu.shape
            # set_prior multiplies σ by prior_sigma_scale
            expected_sigma = sigma.cpu() * policy.sigma_scale
            assert torch.allclose(policy.solver._prior_sigma.cpu(), expected_sigma, atol=1e-5), \
                "prior_sigma_scale should be applied at set_prior time"

        # Now exercise init_action_distrib to confirm fusion behavior matches mode.
        mean_init, std_init = policy.solver.init_action_distrib()

        if mode == "none":
            # No prior set. init_action_distrib must equal upstream.
            assert (mean_init == 0).all(), \
                f"none: mean must be zero (upstream default); got nonzero"
            assert torch.allclose(std_init, torch.full_like(std_init, policy.solver.var_scale)), \
                f"none: std must be var_scale={policy.solver.var_scale}"

        elif mode == "warm_start":
            # Mean replaced with μ_p, std must equal default var_scale (σ NOT used).
            mu_p_expected = mu.to(mean_init.device, dtype=mean_init.dtype)
            if mu_p_expected.shape != mean_init.shape:
                mu_p_expected = mu_p_expected.expand_as(mean_init)
            assert torch.allclose(mean_init, mu_p_expected), \
                "warm_start: mean must equal prior mean"
            assert torch.allclose(std_init, torch.full_like(std_init, policy.solver.var_scale)), \
                "warm_start: std must equal default var_scale (head's σ ignored)"

        elif mode == "pog":
            # Std must differ from var_scale (PoG fusion narrowed it).
            assert not torch.allclose(std_init, torch.full_like(std_init, policy.solver.var_scale)), \
                "pog: fused std should NOT equal the default var_scale"
            assert (std_init >= policy.solver.sigma_floor - 1e-6).all(), \
                f"pog: fused std below floor {policy.solver.sigma_floor}"

        print(f"  {mode:<10}  ✓  z_t={tuple(z_t.shape)}  mu={tuple(mu.shape)}  "
              f"σ_init.mean={std_init.mean().item():.4f}")


def run_check_bc_only():
    print("\n[selfcheck] === bc_only ===")
    policy = setup_policy("bc_only", sigma_scale=1.0)
    info = make_fake_info_dict()

    # bc_only must NOT touch the solver state.
    assert policy.solver.injection_mode == "none", \
        f"bc_only: solver mode must be 'none' (we bypass anyway); got {policy.solver.injection_mode!r}"

    n_pops = policy.flatten_receding_horizon  # 25
    actions = []
    for step in range(n_pops):
        action = policy.get_action(info)
        actions.append(action)

    # Solver state must remain pristine — bc_only never sets a prior.
    assert policy.solver._prior_mean is None, "bc_only: solver._prior_mean should stay None"
    assert policy.solver._prior_sigma is None

    # Output shape per call must match env.action_space.shape (n_envs, A_raw).
    A_raw = 5
    for i, a in enumerate(actions):
        assert isinstance(a, np.ndarray), f"step {i}: expected np.ndarray, got {type(a)}"
        assert a.shape == (N_ENVS, A_raw), f"step {i}: shape {a.shape} != {(N_ENVS, A_raw)}"

    print(f"  bc_only    ✓  popped {n_pops} actions, each shape {actions[0].shape}")
    # Sanity: after popping all, buffer empty → next call would re-encode and refill.
    assert len(policy._action_buffer) == 0, "buffer should be empty after popping all"
    one_more = policy.get_action(info)
    assert one_more.shape == (N_ENVS, A_raw)
    print(f"  bc_only    ✓  re-plan triggered cleanly when buffer drained")


def main():
    print("=" * 64)
    print("  Phase C self-check")
    print("=" * 64)
    run_check_planner_modes()
    run_check_bc_only()
    print("\n[selfcheck] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
