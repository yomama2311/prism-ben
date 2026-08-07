"""DPPriorPolicy — rebuttal §3 hybrids: a Diffusion Policy as the planner's prior.

Answers 1Rwd-W2 / SfQM-Q6 / AC-3: "use a strong policy to guide the SAME world
model + MPPI planner" — three faithful instantiations, all sourcing the prior
from the paper's own DP checkpoint (dp_baseline_<task>.pt, action_horizon=25
= exactly the planner's H × A_block window):

  'best_of_n' — sample N action sequences from DP, score each with the world
                model's goal cost, execute the best (SfQM's exact suggestion:
                "sample multiple action candidates from policy, choose best").
                No MPPI iterations.
  'replace'   — PiJEPA-style: empirical mean + clamped std of the N DP samples
                REPLACE the planner's init distribution (chahe2026pijepa).
  'pog'       — DP moments fused into the init distribution via our PoG rule
                (apples-to-apples with PRISM: same fusion, different prior src).

All modes operate in eval.py's StandardScaler action space (Critical Lesson #1/2):
DP samples come out in DP's [-1,1] min-max space → denorm to raw env units →
process['action'].transform → (n_envs, H, A_block·A_raw).

Per-replan DP sampling latency is accumulated in self.dp_ms (CUDA-synced) for
the rebuttal's prior-compute column.
"""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_worldmodel.policy import WorldModelPolicy

sys.path.insert(0, str(Path(__file__).parent))
from dp_baseline.model import DPModel  # noqa: E402
from dp_baseline.scheduler import DDPMScheduler  # noqa: E402

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DPPriorPolicy(WorldModelPolicy):
    VALID_MODES = ("best_of_n", "replace", "pog")

    def __init__(
        self,
        solver,
        config,
        process=None,
        transform=None,
        dp_ckpt: str = None,
        injection_mode: str = "pog",
        n_dp_samples: int = 16,
        sigma_scale: float = 1.0,
        sigma_clamp_max: float = float('inf'),
        num_inference_steps: int = 10,
        use_ema: bool = True,
        device: str = "cuda",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(solver=solver, config=config, process=process,
                         transform=transform, **kwargs)
        assert injection_mode in self.VALID_MODES, \
            f"injection_mode must be one of {self.VALID_MODES}, got {injection_mode!r}"
        assert dp_ckpt is not None, "Must pass dp_ckpt"
        assert hasattr(solver, "set_prior"), "Solver must be PrismMPPISolver"

        self.injection_mode = injection_mode
        self.n_dp_samples = n_dp_samples
        self.sigma_scale = sigma_scale
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.verbose = verbose

        # ---- Load DP ckpt (identical to dp_baseline/policy.py) ----
        ckpt = torch.load(dp_ckpt, map_location=device, weights_only=False)
        self.action_dim_raw = int(ckpt["action_dim"])
        self.action_horizon = int(ckpt["action_horizon"])
        self.action_min = np.asarray(ckpt["action_min"], dtype=np.float32)
        self.action_max = np.asarray(ckpt["action_max"], dtype=np.float32)
        self.dp_model = DPModel(action_dim=self.action_dim_raw).to(device)
        weights_key = "ema_model" if use_ema and "ema_model" in ckpt else "model"
        self.dp_model.load_state_dict(ckpt[weights_key])
        self.dp_model.eval()
        for p in self.dp_model.parameters():
            p.requires_grad_(False)
        self.noise_sched = DDPMScheduler()

        # Planner window must exactly match DP's prediction window.
        self._H = config.horizon
        self._A_block = config.action_block
        assert self._H * self._A_block == self.action_horizon, (
            f"plan window {self._H}x{self._A_block} != DP action_horizon "
            f"{self.action_horizon}"
        )

        # solver mode: best_of_n bypasses the solver entirely
        solver_mode = "none" if injection_mode == "best_of_n" else injection_mode
        self.solver.injection_mode = solver_mode
        self.solver.prior_sigma_scale = sigma_scale
        self.solver.prior_sigma_clamp_max = sigma_clamp_max

        self._im_mean = _IMAGENET_MEAN.to(device)
        self._im_std = _IMAGENET_STD.to(device)
        self.dp_ms: list[float] = []  # per-replan DP sampling latency

        if verbose:
            print(f"[DPPrior] mode={injection_mode}  N={n_dp_samples}  "
                  f"sigma_scale={sigma_scale}  ddim_steps={num_inference_steps}  "
                  f"weights={weights_key}  dp_horizon={self.action_horizon}")

    # -- helpers ------------------------------------------------------

    def reset_goal_cache(self):
        self._action_buffer.clear()

    def _normalize_pixels(self, raw) -> torch.Tensor:
        if torch.is_tensor(raw):
            arr = raw.detach().cpu().numpy()
        else:
            arr = np.asarray(raw)
        if arr.ndim == 4:
            arr = arr[:, None]
        last = arr[:, -1]
        if last.dtype == np.uint8:
            t = torch.from_numpy(last).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
        else:
            t = torch.from_numpy(last.astype(np.float32)).to(self.device).permute(0, 3, 1, 2)
            if t.max() > 1.5:
                t = t / 255.0
        return (t - self._im_mean) / self._im_std

    @torch.no_grad()
    def _dp_sample_scaled(self, info_dict: dict) -> torch.Tensor:
        """Sample N DP action sequences per env, in StandardScaler space.

        Returns (n_envs, N, H, A_block·A_raw) float32 on self.device.
        """
        obs = self._normalize_pixels(info_dict["pixels"])
        goal = self._normalize_pixels(info_dict["goal"])
        n_envs = obs.shape[0]
        N = self.n_dp_samples

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cond = self.dp_model.encode_obs(obs, goal)                     # (n, C)
        cond_rep = cond.repeat_interleave(N, dim=0)                    # (n·N, C)
        seq_norm = self.noise_sched.sample(
            self.dp_model, cond_rep,
            sample_shape=(n_envs * N, self.action_horizon, self.action_dim_raw),
            num_inference_steps=self.num_inference_steps,
            device=self.device,
        )                                                              # (n·N, 25, A) in [-1,1]
        torch.cuda.synchronize()
        self.dp_ms.append((time.perf_counter() - t0) * 1e3)

        # [-1,1] → raw env units (same as dp_baseline/policy.py)
        scale = np.where((self.action_max - self.action_min) < 1e-8, 1.0,
                         self.action_max - self.action_min)
        seq_raw = (seq_norm.detach().cpu().numpy() + 1.0) * 0.5 * scale + self.action_min

        # raw → StandardScaler space (the solver's action space)
        flat = seq_raw.reshape(-1, self.action_dim_raw)
        flat = self.process["action"].transform(flat)
        seq = torch.from_numpy(
            flat.reshape(n_envs, N, self._H, self._A_block * self.action_dim_raw)
        ).float().to(self.device)
        return seq

    # -- main entry ---------------------------------------------------

    def get_action(self, info_dict: dict, **kwargs):
        if self.injection_mode == "best_of_n":
            return self._get_action_best_of_n(info_dict, **kwargs)

        # replace / pog: refresh the prior only when the buffer forces a replan
        if len(self._action_buffer) == 0:
            samples = self._dp_sample_scaled(info_dict)     # (n, N, H, D)
            mu = samples.mean(dim=1)                        # (n, H, D)
            sigma = samples.std(dim=1, unbiased=False).clamp(min=0.05)
            self.solver.set_prior(mu, sigma)
        return super().get_action(info_dict, **kwargs)

    def _get_action_best_of_n(self, info_dict: dict, **kwargs):
        """Score N DP candidates with the WM goal cost; execute the best.

        Mirrors MPPISolver.solve's info expansion + get_cost contract, and
        PrismHeadPolicy._get_action_bc_only's receding-horizon buffer.
        """
        if len(self._action_buffer) == 0:
            # DP sampling consumes RAW pixels; the WM cost consumes the
            # prepared (transformed/processed) info — order matters.
            candidates = self._dp_sample_scaled(info_dict)  # (n, N, H, D)
            info_dict = self._prepare_info(info_dict)
            n_envs, N = candidates.shape[:2]
            bs = self.solver.batch_size
            best_rows = []
            for s in range(0, n_envs, bs):
                e = min(s + bs, n_envs)
                cur = e - s
                expanded = {}
                for k, v in info_dict.items():
                    v_b = v[s:e]
                    if torch.is_tensor(v):
                        v_b = v_b.unsqueeze(1).expand(cur, N, *v_b.shape[1:])
                    elif isinstance(v, np.ndarray):
                        v_b = np.repeat(v_b[:, None, ...], N, axis=1)
                    expanded[k] = v_b
                costs = self.solver.model.get_cost(expanded, candidates[s:e])
                best = costs.argmin(dim=1)                  # (cur,)
                idx = torch.arange(cur, device=candidates.device)
                best_rows.append(candidates[s:e][idx, best])  # (cur, H, D)
            plan = torch.cat(best_rows)                     # (n, H, D)
            plan = plan.reshape(n_envs, self.flatten_receding_horizon,
                                self.action_dim_raw).cpu()
            self._action_buffer.extend(plan.transpose(0, 1))

        action = self._action_buffer.popleft()
        action = action.reshape(*self.env.action_space.shape).numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action
