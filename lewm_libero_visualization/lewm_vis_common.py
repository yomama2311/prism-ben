"""Shared components for a frozen LeWM -> LIBERO latent-to-RGB visualization.

Place this file beside eval_lewm_libero_offline.py. The checkpoint is always
strict-loaded by the verified evaluator; this script does not modify the JEPA.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import eval_lewm_libero_offline as base
except ImportError as exc:
    raise SystemExit(
        'Put lewm_vis_common.py, train_lewm_decoder.py, render_lewm_video.py '
        'beside eval_lewm_libero_offline.py and inspect_checkpoint.py. '
        f'Original import error: {exc}'
    ) from exc


class UpBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(inc, outc, 3, padding=1),
            nn.GroupNorm(min(8, outc), outc),
            nn.SiLU(inplace=True),
            nn.Conv2d(outc, outc, 3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class LatentRGBDecoder(nn.Module):
    """Diagnostic CLS-token decoder; output [B,3,128,128] in [0,1].

    Not the author's official CLSDecoder. Trained from scratch ONLY on LIBERO.
    """
    def __init__(self, latent_dim=192, resolution=128):
        super().__init__()
        if resolution != 128:
            raise ValueError('Only resolution 128 supported in this implementation')
        self.latent_dim = int(latent_dim)
        self.resolution = resolution
        self.project = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.SiLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            UpBlock(256, 128), UpBlock(128, 64),
            UpBlock(64, 32), UpBlock(32, 16),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f'Expected latent [B,{self.latent_dim}], got {tuple(z.shape)}')
        x = self.project(z.float()).reshape(z.shape[0], 256, 8, 8)
        return self.decoder(x)


def frozen_lewm(args, device):
    sys.path.insert(0, str(Path(args.code_root).resolve()))
    cfg = OmegaConf.load(args.config)
    model, _ = base.build_and_load(args, cfg, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    print('Frozen LeWM strict checkpoint load: PASS', flush=True)
    return model, cfg


def list_demos(root: Path, seed=3072, per_task=10, max_demos=100):
    """Return task-balanced shuffled episode records: (filename, path, frames, task)."""
    found = {}
    for path in sorted(set(root.rglob('*.hdf5')) | set(root.rglob('*.h5'))):
        episodes = []
        with h5py.File(path, 'r') as f:
            if 'data' not in f:
                continue
            for name, grp in f['data'].items():
                if not isinstance(grp, h5py.Group):
                    continue
                if 'actions' not in grp or 'obs/agentview_rgb' not in grp:
                    continue
                if grp['actions'].ndim != 2 or grp['actions'].shape[-1] != 7:
                    continue
                n = min(len(grp['actions']), len(grp['obs/agentview_rgb']))
                if n >= 8:
                    episodes.append((str(path), f'data/{name}', int(n), path.stem))
        if episodes:
            found[path.stem] = episodes
    if not found:
        raise RuntimeError('No LIBERO HDF5 episodes found under ' + str(root))
    rng = random.Random(seed)
    for episodes in found.values():
        rng.shuffle(episodes)
    records = []
    for i in range(per_task):
        for task in sorted(found):
            if i < len(found[task]) and len(records) < max_demos:
                records.append(found[task][i])
    return records


def split_episodes(records, val_fraction=0.2, seed=3072):
    """Split at episode level, stratified per task, to avoid decoder data leakage."""
    from collections import defaultdict
    tasks = defaultdict(list)
    for record in records:
        tasks[record[3]].append(record)
    train, val = [], []
    rng = random.Random(seed)
    for _, rr in sorted(tasks.items()):
        rng.shuffle(rr)
        if len(rr) == 1:
            train.extend(rr)
        else:
            nval = max(1, min(len(rr) - 1, round(len(rr) * val_fraction)))
            val.extend(rr[:nval])
            train.extend(rr[nval:])
    if not train or not val:
        raise ValueError('Need >=2 selected demonstrations in at least one task for validation')
    return train, val


def to_rgb_batch(frames, resolution=128):
    """Frames uint8 HWC -> uint8 NCHW, resized with bilinear OpenCV."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f'Unexpected HDF5 image shape: {frames.shape}')
    if frames.shape[1:3] != (resolution, resolution):
        frames = np.stack([cv2.resize(img, (resolution, resolution),
                         interpolation=cv2.INTER_LINEAR) for img in frames], axis=0)
    return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)


def preprocess_uint8(image_nchw, input_size, device, mode='imagenet'):
    """Consistent encoder input across decoder training and inference.

    imagenet: mirrors the previous evaluator's resize224/Imagenet RGB.
    """
    x = image_nchw.to(device, non_blocking=True).float().div_(255.0)
    if mode.startswith('flipud_'):
        x = x.flip(-2)
        mode = mode[len('flipud_'):]
    x = F.interpolate(x, size=(input_size, input_size),
                      mode='bilinear', align_corners=False)
    if mode == 'imagenet':
        mu = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        sd = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        x = (x - mu) / sd
    elif mode == 'minus_one_one':
        x = x * 2 - 1
    elif mode != 'zero_one':
        raise ValueError('Unsupported --image-mode: ' + mode)
    return x


@torch.inference_mode()
def encode_rgb_uint8(model, frames_nchw, img_size, device, encode_batch=32,
                      image_mode='imagenet'):
    zs = []
    for b in range(0, len(frames_nchw), encode_batch):
        x = preprocess_uint8(frames_nchw[b:b+encode_batch], img_size,
                             device, image_mode)
        output = model.encode({'pixels': x[:, None]})
        z = output['emb']
        if z.ndim != 3 or z.shape[1] != 1:
            raise RuntimeError(f'Unexpected encoder output shape: {z.shape}')
        zs.append(z[:, 0].float().cpu())
    return torch.cat(zs)


def read_frames_and_actions(file, group, n=None, indices=None, resolution=128):
    with h5py.File(file, 'r') as f:
        g = f[group]
        im = g['obs/agentview_rgb']
        acts = g['actions']
        if indices is None:
            imgs = im[:n] if n is not None else im[:]
        else:
            imgs = im[indices]
        actions = None if indices is not None else np.asarray(acts[:n] if n is not None else acts[:], dtype=np.float32)
    return to_rgb_batch(imgs, resolution), actions


def load_decoder(path, device):
    obj = torch.load(path, map_location='cpu', weights_only=True)
    decoder = LatentRGBDecoder(latent_dim=int(obj['latent_dim']),
                               resolution=int(obj['resolution']))
    decoder.load_state_dict(obj['decoder_state'], strict=True)
    return decoder.to(device).eval(), obj


@torch.inference_mode()
def decode_chunks(decoder, latents_cpu, device, batch=32):
    out = []
    for i in range(0, len(latents_cpu), batch):
        out.append(decoder(latents_cpu[i:i+batch].to(device)).float().cpu())
    return torch.cat(out)
