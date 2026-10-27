#!/usr/bin/env python3
"""Inspect LeWM Lightning checkpoint safely (does not need model source)."""
import argparse
from pathlib import Path
import torch


def load(path, unsafe=False):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        if not unsafe:
            raise RuntimeError(
                f"Safe loading failed: {exc}\n"
                "If you trust the checkpoint publisher, rerun with --unsafe-pickle."
            ) from exc
        return torch.load(path, map_location='cpu', weights_only=False)


def get_state(ckpt):
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.state_dict()
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            return ckpt['state_dict']
        if 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
            return ckpt['model_state_dict']
        if 'model' in ckpt and isinstance(ckpt['model'], torch.nn.Module):
            return ckpt['model'].state_dict()
        if ckpt and all(isinstance(x, torch.Tensor) for x in ckpt.values()):
            return ckpt
    raise ValueError(f"Unrecognized checkpoint format: {type(ckpt).__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--unsafe-pickle', action='store_true')
    parser.add_argument('--limit', type=int, default=60)
    args = parser.parse_args()
    c = load(args.checkpoint, args.unsafe_pickle)
    print(f'CHECKPOINT: {args.checkpoint}; size={args.checkpoint.stat().st_size:,} bytes')
    print('Container type:', type(c).__name__)
    if isinstance(c, dict):
        print('Root keys:', list(c.keys())[:50])
        for k in ('epoch', 'global_step', 'pytorch-lightning_version'):
            if k in c:
                print(f'{k}: {c[k]}')
        if 'hyper_parameters' in c:
            print('hyper_parameters:', str(c['hyper_parameters'])[:3000])
    s = get_state(c)
    print(f'State tensors: {len(s)}')
    for i, (k, v) in enumerate(s.items()):
        if i >= args.limit:
            break
        print(f'  {k}: {tuple(v.shape) if hasattr(v,"shape") else type(v).__name__}')
    print('Key prefixes (sample):', sorted(set(k.split('.')[0] for k in s))[:25])

if __name__ == '__main__':
    main()
