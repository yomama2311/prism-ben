#!/usr/bin/env python3
"""Render 3-row LIBERO video: Real / Encoder Reconstruction / LeWM Free Rollout.

Requires frozen model checkpoint AND separately trained decoder_best.pt.
This is an offline video of prediction conditioned on recorded demonstration actions,
NOT a simulated policy rollout. Poor decoder quality makes video inconclusive.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image

from lewm_vis_common import (frozen_lewm, load_decoder, decode_chunks,
                             read_frames_and_actions, encode_rgb_uint8)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--decoder', type=Path, required=True)
    p.add_argument('--libero-root', type=Path, required=True)
    p.add_argument('--code-root', type=Path, default=Path('/workspace/prism-ben'))
    p.add_argument('--output', type=Path, default=Path('results/lewm_video/lewm_comparison.mp4'))
    p.add_argument('--builder', default='')
    p.add_argument('--unsafe-pickle', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--task', default='', help='Filename substring (e.g. black_bowl_from_table_center)')
    p.add_argument('--episode', default='', help='Optional HDF5 group name, e.g. demo_26')
    p.add_argument('--split', choices=['val', 'any'], default='val',
                   help='Prefer decoder held-out episode (val)')
    p.add_argument('--start', type=int, default=0, help='First observation index of context')
    p.add_argument('--horizon', type=int, default=50, help='Number of future frames to predict')
    p.add_argument('--encode-batch', type=int, default=32)
    p.add_argument('--decode-batch', type=int, default=32)
    p.add_argument('--fps', type=int, default=8)
    p.add_argument('--display-size', type=int, default=256)
    p.add_argument('--action-stats', type=Path,
                   help='NPZ with ORIGINAL training mean/std (7,), otherwise raw actions')
    args = p.parse_args()
    if args.start < 0 or min(args.horizon, args.fps, args.display_size,
                             args.encode_batch, args.decode_batch) < 1:
        p.error('Nonpositive argument')
    return args


def choose_episode(args, metadata, history):
    allowed = set(metadata.get('validation_episodes', [])) if args.split == 'val' else None
    paths = sorted(set(args.libero_root.rglob('*.hdf5')) | set(args.libero_root.rglob('*.h5')))
    if args.task:
        paths = [p for p in paths if args.task.lower() in p.stem.lower()]
    if not paths:
        raise FileNotFoundError(f'No LIBERO HDF5 file matches --task={args.task!r}')
    candidates = []
    for path in paths:
        with h5py.File(path, 'r') as f:
            if 'data' not in f:
                continue
            for name in sorted(f['data']):
                if args.episode and args.episode not in (name, 'data/' + name):
                    continue
                group = f['data'][name]
                if 'actions' not in group or 'obs/agentview_rgb' not in group:
                    continue
                n = min(len(group['actions']), len(group['obs/agentview_rgb']))
                if n < args.start + history + 1:
                    continue
                key = f'{path}::data/{name}'
                if allowed is None or key in allowed:
                    candidates.append((str(path), 'data/' + name, n))
    if not candidates:
        if args.split == 'val':
            raise ValueError('No decoder-validation demo matches; try --split any or another --task')
        raise ValueError('No matching episode long enough for given --start')
    # Prefer episodes with enough frames for the full horizon.
    full = [c for c in candidates if c[2] >= args.start + history + args.horizon]
    return sorted(full if full else candidates, key=lambda x: (x[0], x[1]))[0]


def normalize_actions(raw, stats, device):
    if stats is None:
        print('Actions: RAW (training normalization not verified)', flush=True)
        return torch.from_numpy(np.nan_to_num(raw, nan=0, posinf=0, neginf=0)).float().to(device)
    with np.load(stats) as npz:
        mean = np.asarray(npz['mean'], np.float32)
        std = np.asarray(npz['std'], np.float32)
    if mean.shape != (7,) or std.shape != (7,):
        raise ValueError('--action-stats must contain mean[7], std[7]')
    print('Actions normalized with:', stats, flush=True)
    return torch.from_numpy(np.nan_to_num((raw - mean) / np.maximum(std, 1e-6),
                                           nan=0, posinf=0, neginf=0)).float().to(device)


@torch.inference_mode()
def make_latent_rollout(model, encoded, raw_actions, history, horizon,
                        device, stats):
    acts = normalize_actions(raw_actions, stats, device)
    aemb = model.action_encoder(acts[None]).float()
    ground = encoded.to(device).float()
    context = ground[:history][None].clone()
    preds = []
    for step in range(1, horizon + 1):
        target_t = history - 1 + step
        ac = aemb[:, target_t - history:target_t]
        if ac.shape[1] != history:
            raise RuntimeError('Action alignment window incorrect')
        one = model.predict(context, ac)
        if one.ndim != 3:
            raise RuntimeError(f'Unexpected predictor output shape: {one.shape}')
        z = one[:, -1, :].float()
        if not torch.isfinite(z).all():
            raise FloatingPointError('Nonfinite predicted embedding at horizon ' + str(step))
        preds.append(z[0].cpu())
        context = torch.cat((context[:, 1:, :], z[:, None, :]), dim=1)
    return torch.cat((encoded[:history], torch.stack(preds)), dim=0)


def to_numpy_images(chw_tensor):
    x = chw_tensor.float().permute(0, 2, 3, 1).numpy()
    return np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)


def label_rgb(frame, text, size, band_height=32):
    thumb = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    # Draw on BGR internally (OpenCV VideoWriter expects BGR).
    bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
    out = np.zeros((size + band_height, size, 3), dtype=np.uint8)
    out[:band_height] = (35, 35, 35)
    out[band_height:] = bgr
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.56, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, cfg = frozen_lewm(args, device)
    decoder, metadata = load_decoder(args.decoder, device)
    if metadata.get('decoder_type') != 'diagnostic_encoder_cls':
        raise ValueError('Incompatible decoder format')
    if Path(metadata['model_checkpoint']).resolve() != args.checkpoint.resolve():
        raise ValueError('Decoder was trained for another LeWM checkpoint')
    if int(metadata['latent_dim']) != int(cfg.wm.embed_dim):
        raise ValueError('Decoder latent dimension incompatible with LeWM config')
    image_mode = metadata['image_mode']
    h = int(cfg.wm.history_size)
    if int(cfg.data.dataset.frameskip) != 1 or int(cfg.wm.action_dim) != 7:
        raise ValueError('Expected LIBERO action_dim=7 and frameskip=1')
    file, group, n = choose_episode(args, metadata, h)
    horizon = min(args.horizon, n - args.start - h)
    n_take = h + horizon
    print(f'Selected {Path(file).name} / {group}; start={args.start}, '
          f'context={h}, predicting {horizon} frames.', flush=True)
    # Read just the necessary contiguous slice (one HDF5 read).
    with h5py.File(file, 'r') as f:
        g = f[group]
        frames = g['obs/agentview_rgb'][args.start:args.start+n_take]
        raw_actions = np.asarray(g['actions'][args.start:args.start+n_take], np.float32)
    from lewm_vis_common import to_rgb_batch
    gt_uint8 = to_rgb_batch(frames, resolution=128)
    real_z = encode_rgb_uint8(model, gt_uint8, int(cfg.img_size), device,
                              args.encode_batch, image_mode)
    imagined_z = make_latent_rollout(model, real_z, raw_actions, h, horizon,
                                     device, args.action_stats)
    real_rec = decode_chunks(decoder, real_z, device, args.decode_batch)
    imagined = decode_chunks(decoder, imagined_z, device, args.decode_batch)
    gt = (gt_uint8.float() / 255.0).cpu()
    rgb_gt = to_numpy_images(gt)
    rgb_rec = to_numpy_images(real_rec)
    rgb_pred = to_numpy_images(imagined)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'),
                            args.fps, (args.display_size, 3*(args.display_size+32)))
    if not video.isOpened():
        raise RuntimeError('OpenCV could not open MP4 writer; check codec support')
    stats = []
    selected = []
    samples = set([0, h-1] + [h-1+k for k in (1, 5, 10, 25, 50, 100) if k <= horizon])
    for t in range(n_take):
        state = 'CONTEXT' if t < h else f'PREDICT +{t-h+1}'
        img1 = label_rgb(rgb_gt[t], f'Ground Truth   frame {args.start+t}', args.display_size)
        img2 = label_rgb(rgb_rec[t], 'Encoder reconstruction', args.display_size)
        img3 = label_rgb(rgb_pred[t], 'LeWM imagined  ' + state, args.display_size)
        canvas = np.concatenate((img1, img2, img3), axis=0)
        video.write(canvas)
        if t in samples:
            # For PNG, convert BGR to RGB before PIL.
            selected.append(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
        mse_rec = (real_rec[t]-gt[t]).square().mean().item()
        mse_pred = (imagined[t]-gt[t]).square().mean().item()
        latent_mse = (imagined_z[t]-real_z[t]).square().mean().item()
        stats.append({'frame': args.start+t, 'horizon': max(0, t-h+1),
                      'is_context': t < h, 'encoder_recon_rgb_mse': mse_rec,
                      'imagined_rgb_mse': mse_pred, 'imagined_latent_mse': latent_mse,
                      'imagined_psnr_db': (float('inf') if mse_pred == 0
                                           else -10*math.log10(mse_pred))})
    video.release()
    montage = args.output.with_suffix('.png')
    if selected:
        w, hh = selected[0].size
        sheet = Image.new('RGB', (w*len(selected), hh), (255, 255, 255))
        for j, im in enumerate(selected):
            sheet.paste(im, (j*w, 0))
        sheet.save(montage)
    report = {
        'model_checkpoint': str(args.checkpoint),
        'decoder_checkpoint': str(args.decoder),
        'dataset_file': file, 'episode': group, 'start': args.start,
        'horizon': horizon, 'history': h, 'image_mode': image_mode,
        'actions': str(args.action_stats) if args.action_stats else 'RAW',
        'video': str(args.output), 'montage': str(montage),
        'results': stats,
        'warning': 'Diagnostic decoder may reconstruct images poorly, especially from free-running '
                   'out-of-distribution latents. Video is NOT simulator rollout or task success. '
                   'Data preprocessing and action semantics are not verified against original training.'
    }
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved video:', args.output)
    print('Saved montage:', montage)
    print('Saved metrics:', args.output.with_suffix('.json'))
    print('IMPORTANT: Judge row 2 reconstruction quality before interpreting row 3.')


if __name__ == '__main__':
    main()
