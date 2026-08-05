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

    Note: upstream MPPISolver.init_action_distrib returns a tensor named
    `var` but it is actually std — sampling is done as `noise * batch_var`
    (i.e. N(mean, std²)), not `noise * sqrt(batch_var)`. We treat that
    output as std throughout this file. PoG fusion below uses
    precision = 1/std².

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

    Identical to MPPISolver if no prior is set OR injection_mode='none'.

    injection_mode dispatch (only relevant when a prior IS set):
      'pog'        — Product-of-Gaussians fusion of N(μ_p, σ_p · prior_sigma_scale)
                     with planner default N(μ_default, σ_default).
      'warm_start' — Replace planner mean with μ_p; KEEP planner default σ.
                     Head's σ_p is intentionally IGNORED here. This is the
                     ablation row that asks "does precision-weighting (PoG)
                     contribute beyond simple mean substitution?". If results
                     match 'pog', PoG isn't earning its keep.
      'none'       — Bypass any fusion (sanity reproduction of upstream).

    Args:
        sigma_floor:       minimum std after fusion
        prior_sigma_scale: multiplier on σ_p (only used by 'pog')
        injection_mode:    one of {'pog', 'warm_start', 'none'}
    """

    VALID_MODES = ('pog', 'warm_start', 'none', 'replace')

    def __init__(
        self,
        *args,
        sigma_floor: float = 0.05,
        prior_sigma_scale: float = 1.0,
        injection_mode: str = 'pog',
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert injection_mode in self.VALID_MODES, \
            f"injection_mode must be one of {self.VALID_MODES}, got {injection_mode!r}"
        self.sigma_floor = sigma_floor
        self.prior_sigma_scale = prior_sigma_scale
        self.injection_mode = injection_mode
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
        """Initialize action distribution; dispatch on injection_mode."""
        mean, std = super().init_action_distrib(actions)

        # Fast bypass: no prior set, OR mode='none' → byte-identical to upstream.
        if self._prior_mean is None or self.injection_mode == 'none':
            return mean, std

        mu_p = self._prior_mean.to(mean.device, dtype=mean.dtype)
        if mu_p.shape != mean.shape:
            mu_p = mu_p.expand_as(mean)

        if self.injection_mode == 'replace':
            # PiJEPA-style policy-guided init (rebuttal hybrid (b)): REPLACE both
            # the planner's init mean and std with the policy-prior moments.
            # No PoG fusion — this is the faithful reproduction of
            # chahe2026pijepa's "empirical mean and (clamped) std as MPPI init".
            assert self._prior_sigma is not None, "replace mode requires sigma"
            sg_p = self._prior_sigma.to(std.device, dtype=std.dtype)
            if sg_p.shape != std.shape:
                sg_p = sg_p.expand_as(std)
            return mu_p, sg_p.clamp(min=self.sigma_floor)

        if self.injection_mode == 'warm_start':
            # ABLATION ROW: head's σ_p is INTENTIONALLY UNUSED here. We replace
            # the planner's initial mean with the prior mean but keep the planner's
            # default std. If the eval result for warm_start ≈ pog, then PoG's
            # precision-weighting contributes nothing beyond simple mean
            # substitution. The head still computed σ — we just throw it away.
            return mu_p, std

        elif self.injection_mode == 'pog':
            assert self._prior_sigma is not None, "pog mode requires σ_p (set_prior with sigma)"
            sg_p = self._prior_sigma.to(std.device, dtype=std.dtype)
            if sg_p.shape != std.shape:
                sg_p = sg_p.expand_as(std)
            mean, std = pog_fusion(mean, std, mu_p, sg_p, self.sigma_floor)
            return mean, std

        # Should be unreachable due to the constructor assertion.
        raise RuntimeError(f"Unknown injection_mode: {self.injection_mode!r}")


def vlm_influence(std: torch.Tensor, sigma_prior: torch.Tensor) -> torch.Tensor:
    """Compute the precision-weighted influence of the prior on the fused mean.

    influence = tau_prior / (tau_base + tau_prior)

    Returns a tensor of shape std.shape — fraction in [0, 1] per element.
    """
    eps = 1e-8
    tau_base  = 1.0 / (std**2 + eps)
    tau_prior = 1.0 / (sigma_prior**2 + eps)
    return tau_prior / (tau_base + tau_prior + eps)
