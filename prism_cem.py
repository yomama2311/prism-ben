"""prism_cem.py — CEMSolver subclass supporting warm-start prior injection.

PiJEPA-style warm-start: at episode start, the planner's INITIAL sampling
mean is replaced with the head's prior μ. CEM's default initial σ is
preserved (`var_scale = 1.0`). All subsequent CEM iterations run as
vanilla CEM — elite-mean updates, σ adapted from elite variance.

We do NOT implement closed-form PoG for CEM. PoG anchors σ at init
relative to MPPI's *fixed* σ; CEM updates its σ from elites, so the
PoG semantics ("anchor the search near the prior") don't carry over
without modifying the σ-update rule, which the experiment spec
explicitly forbids.

Note: upstream CEMSolver returns a tensor named `var` from
`init_action_distrib`, but it is actually std (used as
`mean + noise * std` to sample from N(mean, std²)). Same mislabel as
upstream MPPISolver; see prism_mppi.py header.
"""
import torch
from stable_worldmodel.solver import CEMSolver


class PrismCEMSolver(CEMSolver):
    """CEM with warm-start init-mean injection from an external prior.

    Identical to upstream CEMSolver when no prior is set. When `set_prior(μ)`
    has been called, the first iteration's sampling distribution uses μ as
    the mean. The default initial std `var_scale` is unchanged.

    The σ argument to `set_prior(μ, σ)` is accepted for API compatibility
    with PrismMPPISolver but is INTENTIONALLY IGNORED here — CEM-warm-start
    is the "no precision-anchoring" baseline.
    """

    def __init__(self, *args, verbose: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._prior_mean = None
        self.verbose = verbose
        self._first_init_call = True  # for the dry-run sanity log

    def set_prior(self, mu: torch.Tensor, sigma: torch.Tensor | None = None) -> None:
        """Store the prior mean for the next solve(). σ is ignored by design."""
        self._prior_mean = mu
        # sigma deliberately discarded

    def clear_prior(self) -> None:
        self._prior_mean = None

    def init_action_distrib(self, actions: torch.Tensor | None = None):
        mean, std = super().init_action_distrib(actions)

        if self._prior_mean is None:
            return mean, std

        mu_p = self._prior_mean.to(mean.device, dtype=mean.dtype)
        if mu_p.shape != mean.shape:
            mu_p = mu_p.expand_as(mean)

        if self.verbose and self._first_init_call:
            # Sanity log: prove the prior actually got injected.
            print(f"[PrismCEM] init injection — pre-inject  mean[0,0,:5] = "
                  f"{mean[0, 0, :5].cpu().tolist()}")
            print(f"[PrismCEM] init injection — post-inject mean[0,0,:5] = "
                  f"{mu_p[0, 0, :5].cpu().tolist()}")
            print(f"[PrismCEM] init injection — std unchanged at default "
                  f"(var_scale={self.var_scale}; std[0,0,:5] = "
                  f"{std[0, 0, :5].cpu().tolist()})")
            self._first_init_call = False

        return mu_p, std
