#!/usr/bin/env python3
"""
Offline fixed-horizon LeWM evaluation on raw LIBERO demonstration HDF5.

This tests latent next-step and open-loop predictions conditioned on recorded actions.
It does NOT measure robot task success. For scientific validity the model code,
image pipeline, training split, and action normalization must match training.
The included default builder targets prism-ben jepa.py / module.py; strict=True
rejects nonmatching architectures. Supply --builder module:function if the
checkpoint was trained with a different JEPA implementation.
"""
from __future__ import annotations
import argparse
import importlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from inspect_checkpoint import load, get_state


def default_builder(cfg):
    """Matches baseline prism-ben/main train.py, not guaranteed for HF v2."""
    import stable_pretraining as spt
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale, patch_size=cfg.patch_size,
        image_size=cfg.img_size, pretrained=False, use_mask_token=False)
    hidden = int(encoder.config.hidden_size)
    embed = int(cfg.wm.get('embed_dim', hidden))
    pcfg = OmegaConf.to_container(cfg.predictor, resolve=True)
    predictor = ARPredictor(
        num_frames=int(cfg.wm.history_size), input_dim=embed,
        hidden_dim=hidden, output_dim=hidden, **pcfg)
    action_encoder = Embedder(
        input_dim=int(cfg.wm.action_dim) * int(cfg.data.dataset.frameskip),
        emb_dim=embed)
    projector = MLP(input_dim=hidden, output_dim=embed,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden, output_dim=embed,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    return JEPA(encoder, predictor, action_encoder, projector, pred_proj)


def build_and_load(args, cfg, device):
    checkpoint = load(args.checkpoint, args.unsafe_pickle)
    state = get_state(checkpoint)
    if args.builder:
        mod_name, sep, fname = args.builder.partition(':')
        if not sep:
            raise ValueError('--builder must be module:function')
        builder = getattr(importlib.import_module(mod_name), fname)
    else:
        builder = default_builder
    model = builder(cfg)
    if not isinstance(model, torch.nn.Module):
        raise TypeError('Model builder must return torch.nn.Module')
    base_keys = model.state_dict()
    variants = [state]
    for prefix in ('model.', 'module.', 'module.model.'):
        if any(k.startswith(prefix) for k in state):
            variants.append({k[len(prefix):]: v for k, v in state.items()
                             if k.startswith(prefix)})
    good = []
    for sd in variants:
        expected = set(base_keys)
        keys = set(sd)
        matching = sum(base_keys[k].shape == sd[k].shape for k in expected & keys)
        good.append((matching, len(keys & expected), sd))
    _, _, best = max(good, key=lambda x: (x[0], x[1]))
    missing = sorted(set(base_keys) - set(best))
    unexpected = sorted(set(best) - set(base_keys))
    shape_bad = [(k, tuple(base_keys[k].shape), tuple(best[k].shape))
                 for k in (set(base_keys) & set(best))
                 if base_keys[k].shape != best[k].shape]
    if missing or unexpected or shape_bad:
        print('Checkpoint/model mismatch; refusing non-strict loading.', file=sys.stderr)
        print('Missing (first 15):', missing[:15], file=sys.stderr)
        print('Unexpected (first 15):', unexpected[:15], file=sys.stderr)
        print('Mismatched shapes (first 15):', shape_bad[:15], file=sys.stderr)
        raise RuntimeError(
            'HF v2 may require its original model architecture; do NOT use strict=False. '
            'Inspect keys with inspect_checkpoint.py, then provide a matching '
            '--builder module:function from the training repository.')
    model.load_state_dict(best, strict=True)
    model = model.to(device).eval()
    return model, checkpoint


def collect_demos(root, limit):
    paths = sorted(root.rglob('*.hdf5')) + sorted(root.rglob('*.h5'))
    demos = []
    for path in sorted(set(paths)):
        try:
            with h5py.File(path, 'r') as f:
                if 'data' not in f:
                    continue
                for name in f['data']:
                    grp = f['data'][name]
                    if not isinstance(grp, h5py.Group):
                        continue
                    if 'actions' not in grp or 'obs/agentview_rgb' not in grp:
                        continue
                    n = min(len(grp['actions']), len(grp['obs/agentview_rgb']))
                    if n >= 4 and grp['actions'].shape[-1] == 7:
                        demos.append((str(path), 'data/' + name, n))
        except (OSError, KeyError) as e:
            print('Skipping', path, e)
    if not demos:
        raise RuntimeError('No raw LIBERO demos found. Expected *.hdf5 /data/demo_N/actions (7D) '
                           'and /data/demo_N/obs/agentview_rgb. NOT the old 35D converted dataset.')
    # Reproducible selection, independent of filename ordering.
    rng = random.Random(3072)
    rng.shuffle(demos)
    return demos[:limit]


def read_demo(path, group, required):
    with h5py.File(path, 'r') as f:
        g = f[group]
        a = np.asarray(g['actions'][:required], dtype=np.float32)
        im = np.asarray(g['obs/agentview_rgb'][:required], dtype=np.uint8)
    return a, im


def preprocess_image(images, img_size, device):
    """Imagenet / ToImage convention. Ensure orientation matches training."""
    t = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float().to(device) / 255.0
    t = F.interpolate(t, size=(img_size, img_size), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None,:,None,None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None,:,None,None]
    return (t - mean) / std


def get_stats(args, demos):
    if args.action_stats:
        with np.load(args.action_stats) as d:
            mean, std = np.asarray(d['mean'], dtype=np.float32), np.asarray(d['std'], dtype=np.float32)
        source = f'provided: {args.action_stats}'
    elif args.action_norm == 'estimate':
        # Estimation from eval demos does NOT reproduce training preprocessing!
        sums = np.zeros(7, dtype=np.float64)
        sums2 = np.zeros(7, dtype=np.float64)
        total = 0
        for p, g, n in demos:
            with h5py.File(p, 'r') as f:
                a = np.nan_to_num(f[g]['actions'][:], nan=0).astype(np.float64)
            sums += a.sum(axis=0)
            sums2 += (a * a).sum(axis=0)
            total += len(a)
        mean = (sums/total).astype(np.float32)
        std = np.sqrt(np.maximum(sums2/total - (sums/total)**2, 1e-12)).astype(np.float32)
        source = 'estimated from EVAL demos (not necessarily training stats!)'
    else:
        mean = np.zeros(7, dtype=np.float32)
        std = np.ones(7, dtype=np.float32)
        source = 'RAW actions (only valid if training used raw actions)'
    if mean.shape != (7,) or std.shape != (7,):
        raise ValueError(f'Action normalization dimensions must be (7,), got {mean.shape} / {std.shape}')
    std = np.maximum(std, 1e-6)
    print('ACTION NORMALIZATION:', source)
    print('Action mean:', mean.tolist())
    print('Action std:', std.tolist())
    return mean, std, source


@torch.inference_mode()
def encode_sequence(model, images, img_size, device, encode_batch):
    latents = []
    for i in range(0, len(images), encode_batch):
        inp = preprocess_image(images[i:i+encode_batch], img_size, device)
        out = model.encode({'pixels': inp[:, None]})
        if 'emb' not in out:
            raise RuntimeError('model.encode must return a dict containing emb')
        z = out['emb']
        if z.ndim != 3 or z.shape[1] != 1:
            raise RuntimeError(f'Unexpected encoded latent shape: {tuple(z.shape)}')
        latents.append(z[:, 0].float())
    return torch.cat(latents, dim=0)


@torch.inference_mode()
def evaluate_window(model, z, a, history, horizons):
    # z[t] is the encoded observation BEFORE a[t] executes.
    # P(z[t+1]) depends on z[t-history+1:t+1], a[same indices].
    # First rollout prediction at t=history produces z[history].
    max_h = min(max(horizons), len(z) - history)
    if max_h <= 0:
        return {}
    aemb = model.action_encoder(a[None]).float()
    zg = z.float()
    free = zg[:history].clone()
    results = {}
    for step in range(1, max_h+1):
        target_t = history - 1 + step
        act_ctx = aemb[:, target_t-history:target_t]
        # The final action of the history window generates observation target_t.
        if act_ctx.shape[1] != history:
            raise AssertionError(f'Unexpected action window at t={target_t}')
        free_input = free[-history:][None]
        free_pred = model.predict(free_input, act_ctx)[:, -1, :].float()[0]
        teacher_input = zg[target_t-history:target_t][None]
        teacher_pred = model.predict(teacher_input, act_ctx)[:, -1, :].float()[0]
        free = torch.cat((free, free_pred[None]), dim=0)
        if step in horizons:
            gt = zg[target_t]
            start = zg[history-1]
            results[step] = {
                'free_mse': (free_pred-gt).square().mean().item(),
                'teacher_mse': (teacher_pred-gt).square().mean().item(),
                'persistence_mse': (start-gt).square().mean().item(),
            }
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--libero-root', type=Path, required=True,
                   help='Directory containing official LIBERO demo *.hdf5 files')
    p.add_argument('--output', type=Path, default=Path('results/lewm_offline_eval.json'))
    p.add_argument('--code-root', type=Path, default=Path('/workspace/prism-ben'))
    p.add_argument('--builder', default='', help='Optional original code model builder: module:function')
    p.add_argument('--max-demos', type=int, default=10)
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 10, 25, 50, 100, 200])
    p.add_argument('--encode-batch', type=int, default=16)
    p.add_argument('--action-norm', choices=['raw', 'estimate'], default='estimate')
    p.add_argument('--action-stats', type=Path,
                   help='NPZ file with arrays mean[7] and std[7] from original training data')
    p.add_argument('--unsafe-pickle', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.max_demos < 1 or not args.horizons or min(args.horizons) < 1:
        p.error('--max-demos and --horizons must be positive')
    args.horizons = sorted(set(args.horizons))
    sys.path.insert(0, str(args.code_root.resolve()))
    cfg = OmegaConf.load(args.config)
    history = int(cfg.wm.history_size)
    expected_act_dim = int(cfg.wm.action_dim) * int(cfg.data.dataset.frameskip)
    if expected_act_dim != 7 or int(cfg.data.dataset.frameskip) != 1:
        raise ValueError(f'Expected action_dim=7 / frameskip=1; got {expected_act_dim}')
    device = torch.device(args.device)
    demos = collect_demos(args.libero_root, args.max_demos)
    means, stds, norm_label = get_stats(args, demos)
    model, checkpoint = build_and_load(args, cfg, device)
    print('Strict checkpoint load: PASS')
    print(f'Found {len(demos)} demos, predicting horizons {args.horizons}')
    accum = defaultdict(lambda: defaultdict(list))
    details = []
    for i, (file, grp, n) in enumerate(demos, 1):
        max_frames = min(n, history + max(args.horizons))
        raw_action, imgs = read_demo(file, grp, max_frames)
        actions = torch.from_numpy(np.nan_to_num((raw_action - means) / stds)).to(device)
        z = encode_sequence(model, imgs, int(cfg.img_size), device, args.encode_batch)
        metrics = evaluate_window(model, z, actions, history, args.horizons)
        for h, measures in metrics.items():
            for metric, val in measures.items():
                accum[h][metric].append(val)
        details.append({'file': file, 'episode': grp, 'frames': n, 'metrics': metrics})
        print(f'[{i}/{len(demos)}] {Path(file).name}:{grp} frames={n} '
              f'horizons={sorted(metrics)}', flush=True)
    summary = {}
    print('\nFIXED-HORIZON RESULTS (latent per-dimension MSE; lower is better)')
    print('Horizon | N | teacher-forced | free-running | persistence | free/persistence')
    for h in args.horizons:
        vals = accum[h]
        if not vals:
            print(f'{h:7d} | 0 (episodes too short)')
            continue
        s = {k: float(np.mean(v)) for k, v in vals.items()}
        s['n'] = len(vals['free_mse'])
        s['relative_to_persistence'] = s['free_mse'] / max(s['persistence_mse'], 1e-12)
        summary[str(h)] = s
        print(f'{h:7d} | {s["n"]:2d} | {s["teacher_mse"]:.6f} | '
              f'{s["free_mse"]:.6f} | {s["persistence_mse"]:.6f} | '
              f'{s["relative_to_persistence"]:.3f}')
    result = {
        'checkpoint': str(args.checkpoint), 'config': str(args.config),
        'code_root': str(args.code_root), 'dataset_root': str(args.libero_root),
        'action_normalization': norm_label,
        'image_preprocessing': 'resize224/Imagenet RGB; check against original training',
        'test_type': 'offline recorded-action latent prediction; NOT LIBERO success rate',
        'summary': summary, 'episodes': details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
