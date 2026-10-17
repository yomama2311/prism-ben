"""Stand-alone latent-space MPPI with Product-of-Gaussians (PoG) prior fusion.

Same maths as the repo's `PrismMPPISolver` + `JEPA.get_cost`, but without the
stable-worldmodel `World`/`Policy` plumbing (LIBERO is not a stable-worldmodel env).

    mean_0, std_0   = 0, var_scale                       (planner default, as in LeWM)
    PoG:  tau = 1/std_0^2 + 1/(s*sigma_p)^2
          mean_init = (mean_0/std_0^2 + mu_p/(s*sigma_p)^2) / tau ,   std_init = tau^-1/2
    MPPI: candidates = mean + noise * std (std frozen), top-k elites, softmax(-(c-min)/T)
    cost: || z_hat_{t+H} - z_target ||^2   (JEPA rollout in latent space)
"""
from __future__ import annotations

import torch


def pog_fusion(mean, std, mu_p, sigma_p, sigma_floor: float = 0.05):
    eps = 1e-8
    tau_b, tau_p = 1.0 / (std ** 2 + eps), 1.0 / (sigma_p ** 2 + eps)
    tau = tau_b + tau_p
    return (tau_b * mean + tau_p * mu_p) / tau, (1.0 / tau).sqrt().clamp(min=sigma_floor)


class LatentPoGMPPI:
    VALID_MODES = ("pog", "warm_start", "none")

    def __init__(self, jepa, horizon: int, action_dim: int, num_samples: int = 300,
                 n_steps: int = 30, topk: int = 30, temperature: float = 0.5,
                 var_scale: float = 1.0, sigma_floor: float = 0.05, history_size: int = 3,
                 device: str = "cpu", seed: int = 0):
        self.jepa = jepa.eval()
        self.H, self.A = horizon, action_dim
        self.S, self.n_steps, self.topk, self.T = num_samples, n_steps, topk, temperature
        self.var_scale, self.sigma_floor, self.hs = var_scale, sigma_floor, history_size
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def rollout_cost(self, z0: torch.Tensor, actions: torch.Tensor, z_target: torch.Tensor):
        """z0 (D,), actions (S,H,A) in normalised units, z_target (D,) -> cost (S,)."""
        S, H, _ = actions.shape
        emb = z0.view(1, 1, -1).expand(S, 1, -1).clone()
        for t in range(H):
            act_emb = self.jepa.action_encoder(actions[:, : t + 1])
            pred = self.jepa.predict(emb[:, -self.hs:], act_emb[:, -self.hs:])[:, -1:]
            emb = torch.cat([emb, pred], dim=1)
        return ((emb[:, -1] - z_target.view(1, -1)) ** 2).sum(-1)

    @torch.no_grad()
    def plan(self, z0, z_target, prior_mu=None, prior_sigma=None, mode: str = "pog",
             sigma_scale: float = 1.0):
        """prior_mu / prior_sigma: (H, A). Returns the optimised mean (H, A)."""
        assert mode in self.VALID_MODES
        dev = self.device
        mean = torch.zeros(self.H, self.A, device=dev)
        std = torch.full((self.H, self.A), self.var_scale, device=dev)
        if prior_mu is not None and mode != "none":
            if mode == "pog":
                mean, std = pog_fusion(mean, std, prior_mu.to(dev),
                                       prior_sigma.to(dev) * sigma_scale, self.sigma_floor)
            else:  # warm_start: prior mean, planner std
                mean = prior_mu.to(dev).clone()
        for _ in range(self.n_steps):
            noise = torch.randn(self.S, self.H, self.A, generator=self.gen, device=dev)
            cand = mean.unsqueeze(0) + noise * std.unsqueeze(0)
            cand[0] = mean
            cost = self.rollout_cost(z0, cand, z_target)
            k = min(self.topk, self.S)
            vals, inds = torch.topk(cost, k=k, largest=False)
            w = torch.softmax(-(vals - vals.min()) / self.T, dim=0).view(k, 1, 1)
            mean = (w * cand[inds]).sum(0)
        return mean
