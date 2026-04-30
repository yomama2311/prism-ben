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

        # Pre-load expert (state, action) pairs and build KNN index.
        # Store ALL actions (not subsampled) so we can look up future action
        # sequences for "smart" oracle plans, not just single-step actions.
        with h5py.File(dataset_h5_path, 'r') as f:
            n = f[match_key].shape[0]
            n_use = min(n, dataset_max_rows)
            rng = np.random.default_rng(0)
            sub_idx = rng.choice(n, n_use, replace=False)
            sub_idx.sort()
            states = f[match_key][sub_idx]
            self._all_actions = f['action'][:].astype(np.float32)  # full sequence
            self._n_total = n

        self._expert_states = states.astype(np.float32)
        self._sub_idx = sub_idx  # maps KNN result idx → original dataset row idx
        self._kdtree = cKDTree(self._expert_states)
        # Backward-compat: single-step actions for KNN result (subsampled)
        self._expert_actions = self._all_actions[sub_idx]
        if self.verbose:
            print(f"[Oracle] Loaded {n_use} expert (state, action) pairs")
            print(f"[Oracle] state dim: {self._expert_states.shape[1]}, "
                  f"action dim: {self._expert_actions.shape[1]}")

        self._n_get_action_calls = 0

    def get_action(self, info_dict: dict, **kwargs):
        """Inject oracle prior into solver, then delegate to parent."""
        # Get current state for each env (before _prepare_info modifies info_dict)
        current_states = self._extract_current_state(info_dict)

        # KNN lookup in raw state space (dataset and env states both un-normalized
        # at this point — _prepare_info hasn't run yet)
        _, sub_indices = self._kdtree.query(current_states, k=1)
        # Map subsampled KNN index back to original dataset row index
        orig_indices = self._sub_idx[sub_indices]  # (n_envs,)

        block = self.cfg.action_block
        H = self.cfg.horizon
        plan_steps = H * block  # 25 env steps for pusht (H=5, block=5)
        n_envs = orig_indices.shape[0]
        raw_action_dim = self._all_actions.shape[1]

        # SMART oracle: take expert's NEXT plan_steps actions starting from matched state.
        # This is much better than tiling a single action over the horizon.
        # Clip to dataset bounds.
        plan_actions = np.zeros(
            (n_envs, plan_steps, raw_action_dim), dtype=np.float32
        )
        for e in range(n_envs):
            start = orig_indices[e]
            end = min(start + plan_steps, self._n_total)
            seg = self._all_actions[start:end]
            plan_actions[e, :seg.shape[0]] = seg
            # If we ran off the end of the dataset, repeat the last action
            if seg.shape[0] < plan_steps:
                plan_actions[e, seg.shape[0]:] = seg[-1]

        # CRITICAL: normalize oracle actions to match solver's StandardScaler space.
        if self.process is not None and "action" in self.process:
            flat = plan_actions.reshape(-1, raw_action_dim)
            flat = self.process["action"].transform(flat)
            plan_actions = flat.reshape(n_envs, plan_steps, raw_action_dim)

        # Reshape (n_envs, plan_steps, raw_dim) → (n_envs, horizon, raw_dim*block)
        # Each horizon step contains `block` consecutive raw actions, flattened.
        oracle_seq = plan_actions.reshape(n_envs, H, block * raw_action_dim)

        mu = torch.from_numpy(oracle_seq).float()
        sigma = torch.ones_like(mu) * self.oracle_sigma
        self.solver.set_prior(mu, sigma)

        if self.verbose and self._n_get_action_calls % 10 == 0:
            print(f"[Oracle step {self._n_get_action_calls}] "
                  f"state[0]={current_states[0]}, plan[0,0]={plan_actions[0, 0]}")
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
