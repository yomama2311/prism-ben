"""World-model sanity gate: does the predictor beat the identity baseline?

For H rollout steps we compare
    pred error      = || f_theta(z_t, a_{t:t+H}) - z_{t+H} ||
    identity error  = || z_t - z_{t+H} ||
A ratio near 1 means the predictor has learned nothing beyond "the future looks
like the present", and planning against it is meaningless. Reference values from
this repo: 0.15 at H=5 on the Franka v3 dataset (usable), 0.33 at H=25 (degraded).

Run:  python check_wm_quality.py --task cube_double --h5 ogbench/cube_double_expert
"""
from __future__ import annotations

import argparse
import os

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from sklearn import preprocessing

STABLE = os.environ.get("STABLEWM_HOME", "/home/yuhai/project/prism-jepa/.stable-wm")

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, help="folder under STABLEWM_HOME holding lewm_object.ckpt")
ap.add_argument("--h5", required=True, help="dataset path under STABLEWM_HOME, without .h5")
ap.add_argument("--n", type=int, default=256, help="number of (episode, start) probes")
ap.add_argument("--horizons", default="5,25", help="rollout lengths in env-steps")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
jepa = torch.load(f"{STABLE}/{a.task}/lewm_object.ckpt", map_location=dev, weights_only=False)
jepa.eval()
enc, proj = jepa.encoder, jepa.projector
pred = getattr(jepa, "predictor", None)
act_enc = getattr(jepa, "action_encoder", None)
assert pred is not None, "checkpoint has no predictor"

IM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def embed(frames_uint8):
    x = torch.from_numpy(frames_uint8).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - IM_MEAN) / IM_STD
    with torch.no_grad():
        return proj(enc(x, interpolate_pos_encoding=True).last_hidden_state[:, 0])


f = h5py.File(f"{STABLE}/{a.h5}.h5", "r")
actions = f["action"][:].astype(np.float32)
ep_off, ep_len = f["ep_offset"][:].astype(np.int64), f["ep_len"][:].astype(np.int64)
nan = np.isnan(actions).any(axis=1)
scaler = preprocessing.StandardScaler().fit(actions[~nan])

rng = np.random.default_rng(a.seed)
print(f"[gate] task={a.task}  episodes={len(ep_len)}  probes={a.n}")
for H in [int(h) for h in a.horizons.split(",")]:
    eps = rng.integers(0, len(ep_len), size=a.n)
    starts = np.array([ep_off[e] + rng.integers(0, max(int(ep_len[e]) - H - 1, 1)) for e in eps])
    z0 = embed(f["pixels"][np.sort(starts)])
    zH = embed(f["pixels"][np.sort(starts) + H])
    # roll the predictor forward H env-steps in blocks of the model's action block
    z = z0.clone()
    # predictor/action-encoder take (B, T, .) sequences; one action BLOCK per
    # plan-step (action_block=5 env-steps x A_raw, flattened), rolled out
    # autoregressively the way jepa.rollout does with history_size=1.
    BLOCK = 5
    assert H % BLOCK == 0, f"H={H} must be a multiple of the action block {BLOCK}"
    acts = np.stack([scaler.transform(actions[s:s + H]) for s in np.sort(starts)])   # (N,H,A)
    N, _, A = acts.shape
    acts = acts.reshape(N, H // BLOCK, BLOCK * A)                                    # (N,T,25)
    with torch.no_grad():
        at = torch.from_numpy(acts).to(dev).float()
        act_emb_all = act_enc(at)                                # (N,T,A_emb)
        zseq = z.unsqueeze(1)                                    # (N,1,D)
        for t in range(at.shape[1]):
            nxt = jepa.predict(zseq[:, -1:], act_emb_all[:, t:t + 1])[:, -1:]
            zseq = torch.cat([zseq, nxt], dim=1)
        zpred = zseq[:, -1]
    pe = (zpred - zH).norm(dim=-1).mean().item()
    ie = (z0 - zH).norm(dim=-1).mean().item()
    print(f"  H={H:>3} env-steps   pred={pe:.4f}   identity={ie:.4f}   ratio={pe/ie:.3f}"
          f"   {'OK' if pe/ie < 0.5 else 'WEAK' if pe/ie < 0.8 else 'FAIL'}")
