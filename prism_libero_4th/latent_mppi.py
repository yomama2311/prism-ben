"""Frozen-sigma PoG/MPPI with causal 3-latent + 2-action-block LeWM history.

Action proposals live in the *world-model normalized* space. Each candidate is
converted to raw LIBERO controller actions, clipped (and optionally gripper-
binarized) BEFORE being passed to LeWM. Exactly those raw actions are returned
for execution. This removes a train/evaluation action mismatch in the original.
"""
from __future__ import annotations

import torch


def pog_fusion(mean, std, mu_p, sigma_p, sigma_floor: float = 0.05):
    if sigma_floor <= 0:
        raise ValueError("sigma_floor must be positive")
    eps = 1e-8
    sigma_p = sigma_p.clamp_min(sigma_floor)
    tau_b = 1.0 / (std.square() + eps)
    tau_p = 1.0 / (sigma_p.square() + eps)
    tau = tau_b + tau_p
    return (tau_b * mean + tau_p * mu_p) / tau, tau.rsqrt().clamp_min(sigma_floor)


class LatentPoGMPPI:
    MODES = ("pog", "warm_start", "none")

    def __init__(self, jepa, horizon=5, action_dim=35, num_samples=32, n_steps=3,
                 topk=0, temperature=0.1, var_scale=1.0, sigma_floor=0.05,
                 history_size=3, device="cpu", seed=42, gripper_mode="continuous"):
        if history_size != 3:
            raise ValueError("This LIBERO runner expects LeWM history_size=3")
        if action_dim % 7 != 0 or action_dim != 35:
            raise ValueError("This checkpoint expects 5 raw x 7D actions = 35")
        if num_samples < 2 or n_steps < 1 or temperature <= 0:
            raise ValueError("invalid MPPI hyperparameters")
        if gripper_mode not in ("continuous", "binary"):
            raise ValueError("gripper_mode must be continuous or binary")
        self.jepa = jepa.eval()
        self.H, self.A = int(horizon), int(action_dim)
        self.S, self.n_steps, self.topk = int(num_samples), int(n_steps), int(topk)
        self.T, self.var_scale = float(temperature), float(var_scale)
        self.sigma_floor = float(sigma_floor)
        self.device, self.gripper_mode = device, gripper_mode
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def _executed_raw(self, proposed_norm, wm_mean, wm_std, low, high):
        """(S,H,35) LeWM-normalized -> (S,H,5,7) EXACT simulator actions."""
        shape = proposed_norm.shape
        raw = proposed_norm.reshape(*shape[:-1], 5, 7) * wm_std + wm_mean
        raw = torch.maximum(torch.minimum(raw, high), low)
        if self.gripper_mode == "binary":
            raw = raw.clone()
            grip = torch.where(raw[..., -1] >= 0, 1.0, -1.0)
            raw[..., -1] = torch.maximum(torch.minimum(grip, high[-1]), low[-1])
        return raw

    @torch.no_grad()
    def predict_one(self, z_history, action_history, raw_block, wm_mean, wm_std):
        """z_history (3,D); action_history (2,35); raw_block (5,7)."""
        if z_history.shape[0] != 3 or action_history.shape != (2, 35):
            raise ValueError("LeWM requires 3 latent observations and 2 preceding action blocks")
        norm = ((raw_block - wm_mean) / wm_std).reshape(1, 1, 35)
        act = torch.cat([action_history.unsqueeze(0), norm], dim=1)
        feats = self.jepa.action_encoder(act)
        pred = self.jepa.predict(z_history.unsqueeze(0), feats)
        return pred[0, -1]

    @torch.no_grad()
    def rollout_cost(self, z_history, action_history, proposed_norm, z_target,
                     wm_mean, wm_std, low, high):
        """Score candidates while retaining the same physical actions as execution."""
        if z_history.shape[0] != 3 or action_history.shape != (2, 35):
            raise ValueError("Incorrect temporal context shape")
        S, H, A = proposed_norm.shape
        if H != self.H or A != 35:
            raise ValueError(f"expected (S,{self.H},35), got {tuple(proposed_norm.shape)}")
        executed_raw = self._executed_raw(proposed_norm, wm_mean, wm_std, low, high)
        executed_norm = ((executed_raw - wm_mean) / wm_std).reshape(S, H, 35)
        zctx = z_history.unsqueeze(0).expand(S, -1, -1).clone()
        actctx = action_history.unsqueeze(0).expand(S, -1, -1).clone()
        first_pred = None
        for i in range(H):
            a = torch.cat((actctx, executed_norm[:, i:i+1]), dim=1)
            emb = self.jepa.action_encoder(a)
            z_next = self.jepa.predict(zctx, emb)[:, -1]
            if i == 0:
                first_pred = z_next
            zctx = torch.cat((zctx[:, 1:], z_next[:, None]), dim=1)
            actctx = a[:, 1:]
        cost = ((zctx[:, -1] - z_target).square()).mean(-1)
        return cost, executed_norm, executed_raw, first_pred

    @torch.no_grad()
    def plan(self, z_history, action_history, z_target, prior_mu, prior_sigma,
             prior_mean, prior_std, wm_mean, wm_std, low, high,
             mode="pog", sigma_scale=1.0):
        """Return (raw_plan (H,5,7), metrics dict); head outputs prior-normalized."""
        if mode not in self.MODES:
            raise ValueError(mode)
        if prior_mu.shape != (self.H, 35) or prior_sigma.shape != prior_mu.shape:
            raise ValueError(f"Prior shape mismatch {prior_mu.shape}, expected {(self.H,35)}")
        prior_mu_raw = prior_mu.reshape(self.H, 5, 7) * prior_std + prior_mean
        prior_mu_wm = ((prior_mu_raw - wm_mean) / wm_std).reshape(self.H, 35)
        prior_sigma_wm = (prior_sigma.reshape(self.H, 5, 7) *
                          prior_std / wm_std).reshape(self.H, 35) * sigma_scale
        mean = torch.zeros_like(prior_mu_wm)
        std = torch.full_like(mean, self.var_scale)
        if mode == "pog":
            mean, std = pog_fusion(mean, std, prior_mu_wm, prior_sigma_wm, self.sigma_floor)
        elif mode == "warm_start":
            mean = prior_mu_wm.clone()

        best_cost, best_norm = float("inf"), None
        last_avg_cost = float("nan")
        for _ in range(self.n_steps):
            noise = torch.randn((self.S, self.H, 35), generator=self.gen, device=self.device)
            proposal = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            proposal[0] = mean
            costs, clipped_norm, _, _ = self.rollout_cost(
                z_history, action_history, proposal, z_target, wm_mean, wm_std, low, high
            )
            if not torch.isfinite(costs).all():
                raise RuntimeError("nonfinite LeWM planning costs")
            min_value, min_idx = costs.min(dim=0)
            if min_value.item() < best_cost:
                best_cost = float(min_value)
                best_norm = clipped_norm[min_idx].clone()
            last_avg_cost = float(costs.mean())
            if self.topk > 0:
                ids = torch.topk(costs, min(self.topk, self.S), largest=False).indices
                weights = torch.softmax(-(costs[ids] - costs[ids].min()) / self.T, dim=0)
                mean = (clipped_norm[ids] * weights[:, None, None]).sum(0)
            else:
                weights = torch.softmax(-(costs - costs.min()) / self.T, dim=0)
                mean = (clipped_norm * weights[:, None, None]).sum(0)
        mean_cost, mean_norm, _, _ = self.rollout_cost(
            z_history, action_history, mean[None], z_target, wm_mean, wm_std, low, high
        )
        use_mean = float(mean_cost[0]) <= best_cost
        selected_norm = mean_norm[0] if use_mean else best_norm
        raw = self._executed_raw(selected_norm[None], wm_mean, wm_std, low, high)[0]
        chosen_cost = float(mean_cost[0]) if use_mean else best_cost
        return raw, {
            "selected_horizon_cost": chosen_cost,
            "best_sample_cost": best_cost,
            "mean_sample_cost": last_avg_cost,
            "selection": "mean" if use_mean else "best_sample",
            "prior_sigma_mean": float(prior_sigma_wm.mean()),
        }
