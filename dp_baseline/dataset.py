"""DP-format dataset that reads our h5 demonstrations.

Each item is a (obs, goal_obs, action_seq) triple suitable for DP training:
  - obs:        (3, 224, 224) RGB at the window start step
  - goal_obs:   (3, 224, 224) RGB at the episode's final step (goal)
  - action_seq: (T_a, action_dim) normalized to [-1, 1] for DP

Window-sampling logic mirrors `train_prior_head.py:build_window_indices` so
DP and PRISM's PriorHead train on the same set of (start_frame, T_a-step
action sequence) pairs, eliminating data-side confounds.

Episode boundary handling:
  - `max_off = ep_len - max(action_horizon, goal_offset_steps)` (exclusive)
    keeps both the T_a-step action window and the +goal_offset_steps goal
    frame inside the episode (also skips the NaN end-of-episode action).
  - Goal frame is at `window_start + goal_offset_steps`, matching the
    eval-time convention (swm.World: `end_steps = start + goal_offset_steps`).
    Default 25 env-steps = matches `cfg.eval.goal_offset_steps`.

Image preprocessing:
  - uint8 / 255 → float32 in [0, 1]
  - Subtract ImageNet mean / divide ImageNet std (matches the JEPA
    encoder's preprocessing — keeps the visual pipeline comparable).

Action preprocessing:
  - Min-max normalize each dim to [-1, 1] using stats computed from
    NaN-filtered training actions. DDPMScheduler.clip_sample=True clips
    x_0 predictions to [-1, 1], so this alignment is necessary.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  — registers Blosc codec for compressed pixels datasets
import numpy as np
import torch
from torch.utils.data import Dataset


_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class DPDataset(Dataset):
    """Sliding-window dataset for DP training on our h5 demonstrations.

    Args:
        h5_path:        path to the demo h5 (e.g. .stable-wm/pusht_expert_train.h5)
        action_horizon: T_a — number of action steps per window (match PRISM
                        plan length: H * action_block = 25)
        split:          'train' or 'val'
        val_ratio:      fraction of episodes held out for validation
        seed:           rng seed for the train/val split
    """

    def __init__(
        self,
        h5_path: str | Path,
        action_horizon: int = 25,
        corrupt_ep_frac: float = 0.0,
        goal_offset_steps: int = 25,
        split: str = "train",
        val_ratio: float = 0.02,
        seed: int = 42,
    ):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"
        self.h5_path = str(h5_path)
        self.action_horizon = action_horizon
        self.goal_offset_steps = goal_offset_steps
        self.split = split

        # ---- Read episode boundaries + ALL actions (small, fits in RAM) ----
        with h5py.File(self.h5_path, "r") as f:
            self.ep_len = f["ep_len"][:].astype(np.int64)
            self.ep_offset = f["ep_offset"][:].astype(np.int64)
            self.actions = f["action"][:].astype(np.float32)   # (N_frames, action_dim)

        self.n_episodes = len(self.ep_len)
        self.action_dim = self.actions.shape[1]
        self.n_frames = self.actions.shape[0]

        # ---- 90/10 episode-level split (same convention as train_prior_head.py) ----
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.n_episodes)
        n_val = max(1, int(self.n_episodes * val_ratio))
        val_eps = set(perm[:n_val].tolist())
        train_eps = set(perm[n_val:].tolist())
        selected_eps = val_eps if split == "val" else train_eps

        # ---- Build start indices: every position where a length-T_a window AND
        #      the +goal_offset_steps goal frame both fit inside the episode ----
        window_span = max(action_horizon, goal_offset_steps)
        starts_list, eps_list = [], []
        for e in sorted(selected_eps):
            max_off = int(self.ep_len[e]) - window_span
            if max_off <= 0:
                continue
            offs = np.arange(max_off, dtype=np.int64)
            starts_list.append(int(self.ep_offset[e]) + offs)
            eps_list.append(np.full(max_off, e, dtype=np.int64))
        self.starts = np.concatenate(starts_list) if starts_list else np.array([], dtype=np.int64)
        self.eps = np.concatenate(eps_list) if eps_list else np.array([], dtype=np.int64)

        # ---- Goal frame index per WINDOW: start + goal_offset_steps ----
        # Matches the eval-time goal frame computed by swm.World
        # (`end_steps = start + goal_offset_steps`, world.py:791), eliminating
        # the train/eval goal-distribution mismatch that hurt the previous
        # end-of-episode goal setup.
        self.goal_indices = self.starts + goal_offset_steps

        # ---- Action normalization (compute on training split only) ----
        if split == "train":
            valid_mask = ~np.isnan(self.actions).any(axis=1)
            valid_actions = self.actions[valid_mask]
            self.action_min = valid_actions.min(axis=0).astype(np.float32)
            self.action_max = valid_actions.max(axis=0).astype(np.float32)
        else:
            # val dataset will need normalization stats too; defer to caller
            # — typically use .copy_stats_from(train_ds) after init
            self.action_min = np.zeros(self.action_dim, dtype=np.float32)
            self.action_max = np.ones(self.action_dim, dtype=np.float32)

        # ---- Rebuttal §5: episode-level action corruption (train split only).
        # Bounds above are computed from CLEAN actions first, so normalization
        # stays aligned with the uncorrupted convention (mirrors
        # train_prior_head.py --corrupt-ep-frac).
        if corrupt_ep_frac > 0.0 and split == "train":
            c_rng = np.random.default_rng(123)
            train_ep_list = np.array(sorted(train_eps))
            n_corrupt = int(round(corrupt_ep_frac * len(train_ep_list)))
            corrupt_set = c_rng.choice(train_ep_list, size=n_corrupt, replace=False)
            for e in corrupt_set:
                off, ln = int(self.ep_offset[e]), int(self.ep_len[e])
                keep_nan = np.isnan(self.actions[off : off + ln]).any(axis=1)
                rand_seq = c_rng.uniform(self.action_min, self.action_max,
                                         size=(ln, self.action_dim)).astype(np.float32)
                rand_seq[keep_nan] = np.nan
                self.actions[off : off + ln] = rand_seq
            print(f"[DPDataset] CORRUPTION: randomized actions of {n_corrupt}/"
                  f"{len(train_ep_list)} train episodes ({100*corrupt_ep_frac:.0f}%)")

        # h5 file handle is opened lazily per dataloader worker (None on init)
        self._h5_handle: h5py.File | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def copy_stats_from(self, other: "DPDataset") -> None:
        """Copy action normalization stats from another (typically train) split."""
        self.action_min = other.action_min.copy()
        self.action_max = other.action_max.copy()

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Map raw action to [-1, 1] via min-max normalization."""
        scale = self.action_max - self.action_min
        scale = np.where(scale < 1e-8, 1.0, scale)
        return 2.0 * (action - self.action_min) / scale - 1.0

    def denormalize_action(self, action_norm: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Inverse of normalize_action: [-1, 1] → raw action units."""
        scale = self.action_max - self.action_min
        scale = np.where(scale < 1e-8, 1.0, scale)
        if isinstance(action_norm, torch.Tensor):
            scale_t = torch.from_numpy(scale).to(action_norm.device, dtype=action_norm.dtype)
            min_t = torch.from_numpy(self.action_min).to(action_norm.device, dtype=action_norm.dtype)
            return (action_norm + 1.0) * 0.5 * scale_t + min_t
        return (action_norm + 1.0) * 0.5 * scale + self.action_min

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def _h5(self) -> h5py.File:
        """Lazily open h5 file. Each dataloader worker gets its own handle."""
        if self._h5_handle is None:
            self._h5_handle = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5_handle

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[idx])
        goal_idx = int(self.goal_indices[idx])

        f = self._h5()

        # --- Image fetch (h5 disk I/O dominates the per-sample cost) ---
        obs_raw = f["pixels"][start]                # (224, 224, 3) uint8
        goal_raw = f["pixels"][goal_idx]            # (224, 224, 3) uint8

        # --- Action sequence (already in RAM, no h5 hit) ---
        action_seq_raw = self.actions[start : start + self.action_horizon]   # (T_a, A)
        # Action stats were computed on NaN-filtered training; in-window NaN
        # is impossible by construction (max_off skips episode-end NaN).
        action_seq_norm = self.normalize_action(action_seq_raw)

        # --- Image preprocessing: uint8 → ImageNet-normalized float32 (C, H, W) ---
        def _prep_img(arr_hwc: np.ndarray) -> np.ndarray:
            x = arr_hwc.astype(np.float32) / 255.0       # [0, 1]
            x = np.transpose(x, (2, 0, 1))               # (3, 224, 224)
            return (x - _IMG_MEAN) / _IMG_STD

        return {
            "obs": torch.from_numpy(_prep_img(obs_raw)),
            "goal_obs": torch.from_numpy(_prep_img(goal_raw)),
            "action_seq": torch.from_numpy(action_seq_norm).float(),
        }

    # ------------------------------------------------------------------
    # Worker-safety: don't pickle open file handles
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_handle"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == "__main__":
    import sys
    import time

    H5_PATHS = {
        "pusht": "/home/yuhai/project/prism-jepa/.stable-wm/pusht_expert_train.h5",
        "cube":  "/home/yuhai/project/prism-jepa/.stable-wm/ogbench/cube_single_expert.h5",
    }

    for name, h5 in H5_PATHS.items():
        if not Path(h5).exists():
            print(f"\n[skip] {name}: h5 not found at {h5}")
            continue
        print(f"\n=== {name}: {h5} ===")
        t0 = time.time()
        ds_train = DPDataset(h5, action_horizon=25, split="train")
        ds_val = DPDataset(h5, action_horizon=25, split="val")
        ds_val.copy_stats_from(ds_train)
        print(f"  init: {time.time() - t0:.1f}s")
        print(f"  n_episodes: {ds_train.n_episodes}  (train: {len(ds_train) // 50}, "
              f"val: {len(ds_val) // 50})  [approx episode counts]")
        print(f"  n_windows: train={len(ds_train):,}  val={len(ds_val):,}")
        print(f"  action_dim: {ds_train.action_dim}")
        print(f"  action range (raw): "
              f"[{ds_train.action_min.tolist()}, {ds_train.action_max.tolist()}]")

        # Probe a few items
        sample = ds_train[0]
        print(f"  sample[0]: obs={tuple(sample['obs'].shape)}  "
              f"goal_obs={tuple(sample['goal_obs'].shape)}  "
              f"action_seq={tuple(sample['action_seq'].shape)} "
              f"(dtype {sample['action_seq'].dtype})")
        assert sample["obs"].shape == (3, 224, 224)
        assert sample["goal_obs"].shape == (3, 224, 224)
        assert sample["action_seq"].shape == (25, ds_train.action_dim)

        # Action normalization sanity: should be in [-1, 1] (modulo eps)
        a_min, a_max = sample["action_seq"].min().item(), sample["action_seq"].max().item()
        print(f"  action_seq range after norm: [{a_min:.3f}, {a_max:.3f}]  "
              f"(should be within [-1, 1])")
        assert -1.01 <= a_min and a_max <= 1.01, f"action norm out of range: [{a_min}, {a_max}]"

        # DataLoader speed check (small batch)
        from torch.utils.data import DataLoader
        loader = DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=2)
        t0 = time.time()
        n = 0
        for batch in loader:
            n += 1
            if n >= 5:
                break
        print(f"  DataLoader (bs=8, workers=2): 5 batches in {time.time() - t0:.1f}s")

    print("\n[OK] dataset.py smoke tests passed.")
