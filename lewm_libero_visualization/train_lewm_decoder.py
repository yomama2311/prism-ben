#!/usr/bin/env python3
"""Train a diagnostic RGB decoder on top of FROZEN LeWM encoder [CLS] latents.

Uses the strict checkpoint loader and image convention from the earlier
LIBERO evaluator. This does NOT fine-tune LeWM, and does NOT measure task success.
It trains a NEW image decoder (not included in the HF LeWM checkpoint).

Place next to lewm_vis_common.py and eval_lewm_libero_offline.py.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageDraw

from lewm_vis_common import (LatentRGBDecoder, frozen_lewm, list_demos,
                             split_episodes, read_frames_and_actions,
                             encode_rgb_uint8)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--libero-root', type=Path, required=True)
    p.add_argument('--code-root', type=Path, default=Path('/workspace/prism-ben'))
    p.add_argument('--output-dir', type=Path, default=Path('results/lewm_decoder'))
    p.add_argument('--builder', default='')
    p.add_argument('--unsafe-pickle', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--image-mode', choices=['imagenet', 'zero_one', 'minus_one_one',
                      'flipud_imagenet'], default='imagenet')
    p.add_argument('--max-demos', type=int, default=100)
    p.add_argument('--per-task', type=int, default=10)
    p.add_argument('--frames-per-demo', type=int, default=40)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--encode-batch', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--seed', type=int, default=3072)
    p.add_argument('--amp', action='store_true', help='BF16 mixed precision on GPU')
    args = p.parse_args()
    if min(args.max_demos, args.per_task, args.frames_per_demo, args.epochs,
           args.encode_batch, args.batch_size) < 1 or not (0 < args.val_fraction < 1):
        p.error('Invalid counts or val fraction')
    return args


def prepare_cache(records, model, img_size, args, device, split):
    """One encoder forward per selected image. Cache small NCHW uint8 and latents on CPU."""
    all_z, all_rgb = [], []
    for i, (file, group, n, task) in enumerate(records, 1):
        indices = np.unique(np.linspace(0, n - 1, min(n, args.frames_per_demo),
                                        dtype=np.int64))
        frames, _ = read_frames_and_actions(file, group, indices=indices, resolution=128)
        z = encode_rgb_uint8(model, frames, img_size, device, args.encode_batch,
                             args.image_mode)
        if z.shape[0] != frames.shape[0]:
            raise RuntimeError('Latent / RGB frame count mismatch')
        all_z.append(z)
        all_rgb.append(frames)
        if i % 10 == 0 or i == len(records):
            print(f'Cached {split}: {i}/{len(records)} episodes, '
                  f'{sum(len(v) for v in all_z)} frames', flush=True)
    return TensorDataset(torch.cat(all_z), torch.cat(all_rgb))


def preview(path, decoder, z, rgb, device, n=5):
    n = min(n, len(z))
    with torch.no_grad():
        rec = decoder(z[:n].to(device)).float().cpu().clamp(0, 1)
    real = rgb[:n].float().div(255).clamp(0, 1)
    real = real.permute(0, 2, 3, 1).numpy()
    recons = rec.permute(0, 2, 3, 1).numpy()
    canvas = Image.new('RGB', (n * 128, 2 * (128 + 22)), 'white')
    d = ImageDraw.Draw(canvas)
    for i in range(n):
        for j, frame in enumerate((real[i], recons[i])):
            x, y = i * 128, j * 150
            d.text((x + 3, y + 3), 'GT' if j == 0 else 'Encoder recon', fill='black')
            tile = Image.fromarray((frame * 255).round().astype(np.uint8))
            canvas.paste(tile, (x, y + 22))
    canvas.save(path)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print('IMPORTANT: decoder-only train; LeWM encoder/predictor FROZEN.')
    print('Image convention is NOT verified against original HF checkpoint training.')
    model, cfg = frozen_lewm(args, device)
    records = list_demos(args.libero_root, args.seed, args.per_task, args.max_demos)
    train_rec, val_rec = split_episodes(records, args.val_fraction, args.seed)
    print(f'Selected {len(train_rec)} train / {len(val_rec)} validation demos '
          '(disjoint for DECODER only, NOT verified against LeWM pretraining).')
    train_data = prepare_cache(train_rec, model, int(cfg.img_size), args, device, 'train')
    val_data = prepare_cache(val_rec, model, int(cfg.img_size), args, device, 'val')
    dim = train_data.tensors[0].shape[-1]
    decoder = LatentRGBDecoder(dim, resolution=128).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.0001)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == 'cuda')

    def score(data_loader, training):
        decoder.train(training)
        total, tot_mse, n = 0., 0., 0
        for z, rgb in data_loader:
            z = z.to(device, non_blocking=True)
            rgb = rgb.to(device, non_blocking=True).float().div_(255)
            with torch.set_grad_enabled(training):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(args.amp and device.type == 'cuda')):
                    decoded = decoder(z)
                    # L1 sharpness term and MSE, same GT pixels for both.
                    mse = F.mse_loss(decoded.float(), rgb)
                    loss = mse + 0.1 * F.l1_loss(decoded.float(), rgb)
                if training:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                    opt.step()
            batch_n = len(z)
            n += batch_n
            total += loss.item() * batch_n
            tot_mse += mse.item() * batch_n
        return total / n, tot_mse / n

    best = float('inf')
    log_path = args.output_dir / 'decoder_metrics.csv'
    with log_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_pixel_mse',
                         'val_loss', 'val_pixel_mse'])
        for epoch in range(1, args.epochs + 1):
            train_loss, train_mse = score(loader, True)
            val_loss, val_mse = score(val_loader, False)
            writer.writerow([epoch, train_loss, train_mse, val_loss, val_mse])
            f.flush()
            print(f'[{epoch:03d}/{args.epochs}] train MSE={train_mse:.6f} '
                  f'val MSE={val_mse:.6f} val loss={val_loss:.6f}', flush=True)
            if val_mse < best:
                best = val_mse
                state = {
                    'decoder_state': {k: v.detach().cpu() for k, v in decoder.state_dict().items()},
                    'latent_dim': dim,
                    'resolution': 128,
                    'image_mode': args.image_mode,
                    'model_checkpoint': str(args.checkpoint.resolve()),
                    'model_config': str(args.config.resolve()),
                    'decoder_type': 'diagnostic_encoder_cls',
                    'epoch': epoch,
                    'val_pixel_mse': best,
                    'training_episodes': [f'{r[0]}::{r[1]}' for r in train_rec],
                    'validation_episodes': [f'{r[0]}::{r[1]}' for r in val_rec],
                }
                torch.save(state, args.output_dir / 'decoder_best.pt')
                preview(args.output_dir / 'encoder_reconstruction_preview.png',
                        decoder, val_data.tensors[0], val_data.tensors[1], device)
    (args.output_dir / 'run_args.json').write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()}, indent=2), encoding='utf-8')
    print(f'Finished. Best val pixel MSE={best:.6f}')
    print(f'Decoder: {args.output_dir / "decoder_best.pt"}')
    print(f'Preview: {args.output_dir / "encoder_reconstruction_preview.png"}')
    print('Judge encoder reconstruction FIRST. If blurry, imagined videos are inconclusive.')


if __name__ == '__main__':
    main()
