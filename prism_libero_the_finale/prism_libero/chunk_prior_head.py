"""Chunked PRISM prior head with latent anchors (for long-horizon tasks such as LIBERO).

Problem with the original `PriorHead`
-------------------------------------
`PriorHead(z_t, z_g)` is one MLP whose output layer has 2*H*A_block*A_raw units. For a
long horizon the goal image z_g is hundreds of env steps away, so the map
(z_t, z_g) -> whole action sequence is badly conditioned: the output layer grows with H,
every plan-step is predicted from the *same* two latents, and nothing ties the late
steps to what the scene will look like by then.

What this module does instead ("split the latent")
--------------------------------------------------
The long horizon is cut into K chunks of C plan-steps (C * A_block raw env steps each).
Every chunk gets its own *latent anchor*:

    a_0 = z_t                          (observed)
    a_{k+1} = a_k + sigmoid(g_k) * (z_g - a_k) + r_k     (learned stepper, k = 0..K-1;
                                                          g_k, r_k = f(a_k, z_g - a_k, k))

and one SHARED chunk decoder turns the pair of neighbouring anchors into the action
Gaussian of that chunk:

    (mu_k, sigma_k) = g(a_k, a_{k+1} - a_k, z_g - a_k, emb(k))      shape (C, A_block, A_raw)

So each decoder call solves a short-horizon problem (exactly the regime where the
original head works well), the output size no longer depends on K, and the anchors are
trained with hindsight latents z_{t + k*L} taken from the demonstrations.

Interface notes
---------------
* `forward(z_t, z_g, n_chunks=1)` returns (mu, sigma) of shape
  (B, n_chunks*C, A_block, A_raw), the same layout as `PriorHead`, so chunk 0 can be fed
  to the PRISM-MPPI solver unchanged (set `plan_horizon == chunk_steps`).
* `predict_anchors(z_t, z_g, n_chunks)` gives the sub-goal latents; a_1 is the natural
  planning target for the first chunk.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from prior_head import PriorHead  # repo's original head (kept for the flat ablation)


def _mlp(d_in: int, d_hidden: int, d_out: int, n_hidden: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(d_in, d_hidden), nn.GELU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(d_hidden, d_hidden), nn.GELU()]
    layers.append(nn.Linear(d_hidden, d_out))
    return nn.Sequential(*layers)


class ChunkedPriorHead(nn.Module):
    def __init__(
        self,
        z_dim: int,
        A_block: int,
        A_raw: int,
        chunk_steps: int = 5,
        max_chunks: int = 4,
        hidden: int = 512,
        sigma_floor: float = 0.05,
        k_emb: int = 32,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.A_block = A_block
        self.A_raw = A_raw
        self.C = chunk_steps
        self.K = max_chunks
        self.H = chunk_steps                    # PriorHead-compatible: default plan horizon
        self.chunk_len = chunk_steps * A_block  # raw env steps covered by one chunk
        self.sigma_floor = sigma_floor
        self.chunk_dim = chunk_steps * A_block * A_raw

        self.k_embed = nn.Embedding(max_chunks, k_emb)
        # anchor stepper: (a_k, z_g - a_k, k) -> [residual (D), gate logit (1)]
        #   a_{k+1} = a_k + sigmoid(gate) * (z_g - a_k) + residual
        # Zero-init weights + gate bias logit(0.2) => the untrained path is a straight line
        # towards the goal that covers ~20% of the remaining distance per chunk.
        self.anchor_net = _mlp(2 * z_dim + k_emb, hidden, z_dim + 1, n_hidden=2)
        nn.init.zeros_(self.anchor_net[-1].weight)
        nn.init.zeros_(self.anchor_net[-1].bias)
        with torch.no_grad():
            self.anchor_net[-1].bias[-1] = -1.386
        # chunk decoder: (a_k, a_{k+1}-a_k, z_g-a_k, k) -> (mu, raw_sigma)
        self.decoder = _mlp(3 * z_dim + k_emb, hidden, 2 * self.chunk_dim, n_hidden=2)
        # per-dim feature std of the cached latents, used for anchor-noise augmentation
        self.register_buffer("feat_std", torch.ones(z_dim))

    # ------------------------------------------------------------------ anchors
    def step_anchor(self, a_k: torch.Tensor, z_g: torch.Tensor, k: int) -> torch.Tensor:
        ke = self.k_embed.weight[k].expand(a_k.size(0), -1)
        out = self.anchor_net(torch.cat([a_k, z_g - a_k, ke], dim=-1))
        res, gate = out[..., :-1], torch.sigmoid(out[..., -1:])
        return a_k + gate * (z_g - a_k) + res

    def predict_anchors(self, z_t: torch.Tensor, z_g: torch.Tensor, n_chunks: int | None = None):
        """Free-running anchors. Returns (B, n_chunks + 1, D) with a_0 = z_t."""
        n = self.K if n_chunks is None else n_chunks
        assert 1 <= n <= self.K, f"n_chunks must be in [1, {self.K}]"
        anchors = [z_t]
        for k in range(n):
            anchors.append(self.step_anchor(anchors[-1], z_g, k))
        return torch.stack(anchors, dim=1)

    # ------------------------------------------------------------------ decoder
    def decode_chunk(self, a_k, a_next, z_g, k: int):
        """-> mu, sigma each (B, C, A_block, A_raw)."""
        ke = self.k_embed.weight[k].expand(a_k.size(0), -1)
        x = torch.cat([a_k, a_next - a_k, z_g - a_k, ke], dim=-1)
        mu_flat, raw = self.decoder(x).chunk(2, dim=-1)
        sigma_flat = F.softplus(raw) + self.sigma_floor
        shape = (a_k.size(0), self.C, self.A_block, self.A_raw)
        return mu_flat.reshape(shape), sigma_flat.reshape(shape)

    # ------------------------------------------------------------------ inference
    def forward(self, z_t: torch.Tensor, z_g: torch.Tensor, n_chunks: int = 1):
        """Deployment path (free-running anchors). -> (B, n_chunks*C, A_block, A_raw)."""
        anchors = self.predict_anchors(z_t, z_g, n_chunks)
        mus, sigmas = [], []
        for k in range(n_chunks):
            mu, sg = self.decode_chunk(anchors[:, k], anchors[:, k + 1], z_g, k)
            mus.append(mu)
            sigmas.append(sg)
        return torch.cat(mus, dim=1), torch.cat(sigmas, dim=1)

    # ------------------------------------------------------------------ training
    def forward_train(
        self,
        anchors_true: torch.Tensor,  # (B, K+1, D) hindsight latents, [:,0] == z_t
        z_g: torch.Tensor,           # (B, D)
        anchor_noise: float = 0.1,   # in units of per-dim feature std
        p_pred: float = 0.3,         # prob. of feeding the decoder the free-run anchors
    ):
        """Returns dict with mu/sigma (B,K,C,Ab,Ar) and anchor predictions.

        * `anc_tf`   (B,K,D): a_{k+1} predicted from the TRUE a_k   (teacher forcing)
        * `anc_free` (B,K,D): a_{k+1} from the free-running chain   (what deployment sees)
        The decoder is fed noisy true anchors, or (with prob. p_pred per sample) the
        detached free-run anchors, so it learns to tolerate predicted anchors.
        """
        B, Kp1, D = anchors_true.shape
        K = Kp1 - 1
        assert K <= self.K

        # teacher-forced anchor targets
        anc_tf = torch.stack(
            [self.step_anchor(anchors_true[:, k], z_g, k) for k in range(K)], dim=1
        )
        # free-running chain
        chain = [anchors_true[:, 0]]
        for k in range(K):
            chain.append(self.step_anchor(chain[-1], z_g, k))
        free = torch.stack(chain, dim=1)  # (B, K+1, D)

        noisy = anchors_true + anchor_noise * self.feat_std * torch.randn_like(anchors_true)
        noisy = torch.cat([anchors_true[:, :1], noisy[:, 1:]], dim=1)  # keep a_0 exact
        use_pred = (torch.rand(B, 1, 1, device=anchors_true.device) < p_pred)
        dec_in = torch.where(use_pred, free.detach(), noisy)

        mus, sigmas = [], []
        for k in range(K):
            mu, sg = self.decode_chunk(dec_in[:, k], dec_in[:, k + 1], z_g, k)
            mus.append(mu)
            sigmas.append(sg)
        return dict(
            mu=torch.stack(mus, dim=1),
            sigma=torch.stack(sigmas, dim=1),
            anc_tf=anc_tf,
            anc_free=free[:, 1:],
        )


# ---------------------------------------------------------------------- losses
def masked_beta_nll(mu, sigma, target, mask, beta: float = 0.5) -> torch.Tensor:
    """beta-NLL (Seitzer et al. 2022), averaged over elements where mask == 1.

    mu/sigma/target: (..., A_raw);  mask broadcastable to them (e.g. (...,1)).
    """
    var = sigma.pow(2)
    nll = 0.5 * (target - mu).pow(2) / var + sigma.log()
    w = sigma.detach().pow(2 * beta)
    m = mask.expand_as(nll).to(nll.dtype)
    return (w * nll * m).sum() / m.sum().clamp(min=1.0)


# ---------------------------------------------------------------------- io helpers
def head_config(head: ChunkedPriorHead) -> dict:
    return dict(
        arch="chunked", z_dim=head.z_dim, A_block=head.A_block, A_raw=head.A_raw,
        chunk_steps=head.C, max_chunks=head.K, hidden=head.decoder[0].out_features,
        sigma_floor=head.sigma_floor, k_emb=head.k_embed.embedding_dim,
    )


def load_head(path: str, device: str = "cpu"):
    """Load either a ChunkedPriorHead ckpt (this package) or an original PriorHead ckpt."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"])
    if cfg.pop("arch", "flat") == "chunked":
        head = ChunkedPriorHead(**cfg)
    else:
        head = PriorHead(z_dim=cfg["z_dim"], H=cfg["H"], A_block=cfg["A_block"],
                         A_raw=cfg["A_raw"], hidden=cfg["hidden"],
                         sigma_floor=cfg["sigma_floor"])
    head.load_state_dict(ckpt["state_dict"])
    return head.to(device).eval(), ckpt
