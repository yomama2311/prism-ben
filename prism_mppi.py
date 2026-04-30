"""
PRISM-MPPI: MPPI solver with Product-of-Gaussians fusion of an external action prior.

Subclasses stable_worldmodel.solver.MPPISolver. Adds:
  - set_prior(mu, sigma): externally provide an action-prior Gaussian
  - PoG fusion is applied to (mean, var) right after init, before MPPI iterations

Oracle / VLM agnostic. The "prior provider" is set externally each step.
"""

import torch
from stable_worldmodel.solver import MPPISolver


def pog_fusion(mean, std, mu_prior, sigma_prior, sigma_floor=0.05):
    """Product-of-Gaussians fusion of two Gaussians.

    Args:
        mean:        (..., A) — MPPI initial mean
        std:         (..., A) — MPPI initial std (NOT variance, despite the
                                 stable-worldmodel naming convention)
        mu_prior:    (..., A) — external prior mean
        sigma_prior: (..., A) — external prior std
        sigma_floor: minimum output std

    Returns:
        fused_mean: (..., A)
        fused_std:  (..., A)
    """
    eps = 1e-8
    tau_base  = 1.0 / (std**2 + eps)
    tau_prior = 1.0 / (sigma_prior**2 + eps)
    tau_combined = tau_base + tau_prior

    fused_mean = (tau_base * mean + tau_prior * mu_prior) / tau_combined
    fused_std  = (1.0 / tau_combined).sqrt().clamp(min=sigma_floor)
    return fused_mean, fused_std


class PrismMPPISolver(MPPISolver):
    """MPPI with PoG fusion of an external action prior at init time.

    Identical to MPPISolver if no prior is set. When a prior is set via
    set_prior(), the initial (mean, var) of the action distribution is
    fused with the prior using PoG.

    Args:
        sigma_floor: minimum std after fusion
        prior_sigma_scale: multiplier on sigma_prior. Acts like vlm_sigma_scale
                            in PRISM — larger values reduce prior influence.
    """

    def __init__(
        self,
        *args,
        sigma_floor: float = 0.05,
        prior_sigma_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sigma_floor = sigma_floor
        self.prior_sigma_scale = prior_sigma_scale
        self._prior_mean = None   # (n_envs, horizon, action_dim) tensor
        self._prior_sigma = None  # same shape; absolute units (already scaled)

    def set_prior(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor | None = None,
    ) -> None:
        """Set the action prior used at the next solve() call.

        Args:
            mu:    (n_envs, horizon, action_dim) — prior mean.
            sigma: same shape, OR scalar, OR None (in which case we use
                   prior_sigma_scale * 1.0 broadcast).
        """
        self._prior_mean = mu
        if sigma is None:
            sigma = torch.ones_like(mu) * self.prior_sigma_scale
        elif sigma.dim() == 0:
            sigma = torch.ones_like(mu) * sigma * self.prior_sigma_scale
        else:
            sigma = sigma * self.prior_sigma_scale
        self._prior_sigma = sigma

    def clear_prior(self) -> None:
        self._prior_mean = None
        self._prior_sigma = None

    def init_action_distrib(self, actions: torch.Tensor | None = None):
        """Initialize action distribution; apply PoG fusion if a prior is set."""
        mean, std = super().init_action_distrib(actions)

        if self._prior_mean is not None and self._prior_sigma is not None and not getattr(self, '_disable_fusion', False):
            # Move to same device as mean
            mu_p = self._prior_mean.to(mean.device, dtype=mean.dtype)
            sg_p = self._prior_sigma.to(std.device, dtype=std.dtype)

            # Broadcast / shape-check
            if mu_p.shape != mean.shape:
                # Expand if shapes are broadcastable
                mu_p = mu_p.expand_as(mean)
                sg_p = sg_p.expand_as(std)

            mean, std = pog_fusion(mean, std, mu_p, sg_p, self.sigma_floor)

        return mean, std


def vlm_influence(std: torch.Tensor, sigma_prior: torch.Tensor) -> torch.Tensor:
    """Compute the precision-weighted influence of the prior on the fused mean.

    influence = tau_prior / (tau_base + tau_prior)

    Returns a tensor of shape std.shape — fraction in [0, 1] per element.
    """
    eps = 1e-8
    tau_base  = 1.0 / (std**2 + eps)
    tau_prior = 1.0 / (sigma_prior**2 + eps)
    return tau_prior / (tau_base + tau_prior + eps)
