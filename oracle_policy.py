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
        match_keys: list = None,        # multi-key concat: e.g. ['state', 'goal_state']
        h5_match_keys: list = None,     # corresponding h5 keys (default: same as match_keys)
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            solver=solver, config=config, process=process, transform=transform, **kwargs
        )
        assert dataset_h5_path is not None, "Must pass dataset_h5_path"
        assert hasattr(solver, 'set_prior'), \
            "Solver must implement set_prior() — use PrismMPPISolver"

        # Normalize match_keys: prefer multi-key list; fall back to legacy single match_key
        if match_keys is not None and len(match_keys) > 0:
            self.match_keys = list(match_keys)
        else:
            self.match_keys = [match_key]
        # h5 may use different field names than runtime info_dict (e.g. tworoom:
        # info_dict has 'state'/'goal_state' but h5 has 'proprio'/'pos_target')
        if h5_match_keys is not None and len(h5_match_keys) > 0:
            self.h5_match_keys = list(h5_match_keys)
        else:
            self.h5_match_keys = self.match_keys

        # Backward-compat: keep `match_key` attribute for any consumer
        self.match_key = self.match_keys[0]

        self.verbose = verbose
        self.oracle_sigma = oracle_sigma

        # Pre-load expert (state, action) pairs and build KNN index.
        with h5py.File(dataset_h5_path, 'r') as f:
            available_keys = list(f.keys())

            # Auto-fallback for the FIRST h5 key only (preserves backward compat)
            if self.h5_match_keys[0] not in available_keys:
                fallback_priority = ['state', 'proprio', 'observation']
                fallback = next((k for k in fallback_priority if k in available_keys), None)
                if fallback is None:
                    raise KeyError(
                        f"h5 match key '{self.h5_match_keys[0]}' not in dataset. "
                        f"Available: {available_keys}"
                    )
                if self.verbose:
                    print(f"[Oracle] h5 match key '{self.h5_match_keys[0]}' not found; "
                          f"falling back to '{fallback}'")
                self.h5_match_keys[0] = fallback
                self.match_keys[0] = fallback
                self.match_key = fallback

            # Verify all keys exist
            for k in self.h5_match_keys:
                if k not in available_keys:
                    raise KeyError(
                        f"h5 match key '{k}' not in dataset. Available: {available_keys}"
                    )

            # Filter NaN actions (reacher_random has NaN at episode boundaries)
            all_actions_np = f['action'][:].astype(np.float32)
            valid_mask = ~np.isnan(all_actions_np).any(axis=1)
            n_total_raw = all_actions_np.shape[0]
            if not valid_mask.all():
                n_dropped = (~valid_mask).sum()
                if self.verbose:
                    print(f"[Oracle] Dropping {n_dropped}/{n_total_raw} rows with NaN actions")
                all_actions_np = np.where(np.isnan(all_actions_np), 0.0, all_actions_np)

            self._all_actions = all_actions_np
            self._valid_mask = valid_mask
            self._n_total = n_total_raw

            # Sample states from valid rows; concat multi-key h5 fields
            valid_idx = np.nonzero(valid_mask)[0]
            n_use = min(len(valid_idx), dataset_max_rows)
            rng = np.random.default_rng(0)
            sub_idx = np.sort(rng.choice(valid_idx, n_use, replace=False))
            # Build multi-key concatenated state vector
            state_parts = []
            for hk in self.h5_match_keys:
                arr = f[hk][sub_idx]
                if arr.ndim == 1:
                    arr = arr[:, None]
                state_parts.append(arr.astype(np.float32))
            states = np.concatenate(state_parts, axis=1)
            self._h5_dim_per_key = [p.shape[1] for p in state_parts]

        self._expert_states = states
        self._sub_idx = sub_idx  # maps KNN result idx → original dataset row idx
        self._kdtree = cKDTree(self._expert_states)
        # Backward-compat: single-step actions for KNN result (subsampled)
        self._expert_actions = self._all_actions[sub_idx]
        if self.verbose:
            print(f"[Oracle] Loaded {n_use} expert pairs")
            print(f"[Oracle] match_keys (runtime): {self.match_keys}")
            print(f"[Oracle] h5_match_keys: {self.h5_match_keys} → dims {self._h5_dim_per_key}")
            print(f"[Oracle] concatenated state dim: {self._expert_states.shape[1]}, "
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

        Concatenates multiple match_keys (e.g. ['state', 'goal_state']) so that
        KNN matching disambiguates between expert trajectories with different
        goals. For single-key default, behaves identically to the old version.
        """
        parts = []
        for k in self.match_keys:
            v = info_dict[k]
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy()
            else:
                v = np.asarray(v)
            if v.ndim == 3:
                v = v[:, -1]   # (n_envs, history, dim) → (n_envs, dim)
            elif v.ndim == 2:
                pass
            elif v.ndim == 1:
                v = v[:, None]  # (n_envs,) → (n_envs, 1)
            else:
                raise ValueError(f"Unexpected shape for '{k}': {v.shape}")
            parts.append(v.astype(np.float32))
        v = np.concatenate(parts, axis=1)
        return v.astype(np.float32)
