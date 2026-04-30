"""
Oracle prior policy: wraps WorldModelPolicy and injects an oracle action prior
into the solver before each plan() call.

The "oracle" is constructed by KNN lookup against a pre-loaded expert dataset:
  - At each get_action() call, take the agent's current state.
  - Find the nearest expert state in the dataset.
  - Use the expert's action at that state as μ_prior.
  - Repeat across the planning horizon (constant prior, since we don't yet know
    where the agent will be at future steps).

This is a probe: if PoG fusion + oracle prior helps MPPI converge faster /
reach higher SR, the PoG mechanism in LeWM is validated and we can move on
to a real VLM-based prior (which would predict similar state-action mappings).
"""

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from stable_worldmodel.policy import WorldModelPolicy


class OracleWorldModelPolicy(WorldModelPolicy):
    """WorldModelPolicy with KNN-based oracle action prior injection."""

    def __init__(
        self,
        solver,
        config,
        process=None,
        transform=None,
        dataset_h5_path: str = None,
        dataset_max_rows: int = 200_000,
        oracle_sigma: float = 0.1,
        match_key: str = 'state',
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            solver=solver, config=config, process=process, transform=transform, **kwargs
        )
        assert dataset_h5_path is not None, "Must pass dataset_h5_path"
        assert hasattr(solver, 'set_prior'), \
            "Solver must implement set_prior() — use PrismMPPISolver"

        self.oracle_sigma = oracle_sigma
        self.match_key = match_key
        self.verbose = verbose

        # Pre-load expert (state, action) pairs and build KNN index
        with h5py.File(dataset_h5_path, 'r') as f:
            n = f[match_key].shape[0]
            n_use = min(n, dataset_max_rows)
            rng = np.random.default_rng(0)
            idx = rng.choice(n, n_use, replace=False)
            idx.sort()
            states = f[match_key][idx]
            actions = f['action'][idx]

        self._expert_states = states.astype(np.float32)
        self._expert_actions = actions.astype(np.float32)
        self._kdtree = cKDTree(self._expert_states)
        if self.verbose:
            print(f"[Oracle] Loaded {n_use} expert (state, action) pairs")
            print(f"[Oracle] state dim: {self._expert_states.shape[1]}, "
                  f"action dim: {self._expert_actions.shape[1]}")

        self._n_get_action_calls = 0

    def get_action(self, info_dict: dict, **kwargs):
        """Inject oracle prior into solver, then delegate to parent."""
        # Get current state for each env (before _prepare_info modifies info_dict)
        current_states = self._extract_current_state(info_dict)

        # KNN lookup: find nearest expert state for each env
        _, indices = self._kdtree.query(current_states, k=1)
        oracle_actions = self._expert_actions[indices]  # (n_envs, action_dim)

        # Build constant prior over planning horizon
        H = self.cfg.horizon * self.cfg.action_block  # action_dim is flattened
        # action_block frames repeated, but we just constant-fill
        n_envs = oracle_actions.shape[0]
        action_dim = self.solver.action_dim  # flattened including action_block
        # Repeat oracle action to fill horizon × action_block dims
        # oracle is per-step (action_dim_raw,) but solver expects (n_envs, horizon, action_dim_flattened)
        # action_dim_flattened = action_dim_raw * action_block
        action_dim_raw = oracle_actions.shape[1]
        block = self.cfg.action_block
        # tile across the action_block dim
        oracle_block = np.tile(oracle_actions, (1, block))  # (n_envs, action_dim_raw * block)
        # tile across horizon
        oracle_seq = np.tile(
            oracle_block[:, None, :], (1, self.cfg.horizon, 1)
        )  # (n_envs, horizon, action_dim_flattened)

        mu = torch.from_numpy(oracle_seq).float()
        sigma = torch.ones_like(mu) * self.oracle_sigma
        self.solver.set_prior(mu, sigma)

        if self.verbose and self._n_get_action_calls % 10 == 0:
            print(f"[Oracle step {self._n_get_action_calls}] "
                  f"state[0]={current_states[0]}, oracle_action[0]={oracle_actions[0]}")
        self._n_get_action_calls += 1

        return super().get_action(info_dict, **kwargs)

    def _extract_current_state(self, info_dict: dict) -> np.ndarray:
        """Extract current state per env as a numpy array of shape (n_envs, state_dim).

        Handles both shapes: (n_envs, history, state_dim) and (n_envs, state_dim).
        """
        v = info_dict[self.match_key]
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        else:
            v = np.asarray(v)

        if v.ndim == 3:
            # (n_envs, history, state_dim) — take latest frame
            v = v[:, -1]
        elif v.ndim == 2:
            # (n_envs, state_dim) — already correct
            pass
        else:
            raise ValueError(
                f"Unexpected shape for {self.match_key}: {v.shape}"
            )
        return v.astype(np.float32)
