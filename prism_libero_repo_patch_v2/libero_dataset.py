"""Direct LIBERO reader. No conversion or simulator installation required."""
import bisect
from collections import OrderedDict
from pathlib import Path
import h5py
try:
    import hdf5plugin  # register optional HDF5 compression filters
except ImportError:
    pass
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def make_split(root, history=3, frameskip=5, val_fraction=.1, seed=3072, camera='agentview_rgb'):
    if history < 1 or frameskip < 1 or not 0 < val_fraction < 1:
        raise ValueError('Invalid history, frameskip, or val fraction')
    rng = np.random.default_rng(seed)
    train, val = [], []
    files = sorted(set(Path(root).rglob('*.hdf5')) | set(Path(root).rglob('*.h5')))
    if not files:
        raise ValueError(f'No HDF5 files under {root}')
    for path in files:
        episodes = []
        with h5py.File(path, 'r') as f:
            if 'data' not in f:
                raise ValueError(f'{path}: expected raw LIBERO data/demo_*/obs layout')
            for name in sorted(f['data']):
                g = f['data'][name]
                if not isinstance(g, h5py.Group) or 'actions' not in g:
                    continue
                a, images = g['actions'], g[f'obs/{camera}']
                if a.ndim != 2 or a.shape[1] != 7 or images.ndim != 4 or images.shape[-1] != 3:
                    raise ValueError(f'Unexpected shape in {path}/{name}')
                if len(a) != len(images):
                    raise ValueError(f'Action/image length mismatch in {path}/{name}')
                # Observation at start + history*frameskip must exist.
                count = len(images) - history * frameskip
                if count > 0:
                    episodes.append(dict(file=str(path.resolve()), demo=name, clips=count))
        if len(episodes) < 2:
            raise ValueError(f'{path}: need >=2 eligible demos for per-task split')
        rng.shuffle(episodes)
        n = max(1, min(len(episodes)-1, round(len(episodes)*val_fraction)))
        val.extend(episodes[:n]); train.extend(episodes[n:])
    return train, val


def action_stats(episodes):
    total = np.zeros(7, dtype=np.float64); squares = total.copy(); count = 0
    for e in episodes:
        with h5py.File(e['file'], 'r') as f:
            a = np.asarray(f[f"data/{e['demo']}/actions"], dtype=np.float64)
        if not np.isfinite(a).all():
            raise ValueError(f'Nonfinite actions: {e}')
        total += a.sum(0); squares += (a*a).sum(0); count += len(a)
    mean = total/count
    std = np.sqrt(np.maximum(squares/count-mean*mean, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


class LiberoClips(Dataset):
    def __init__(self, episodes, mean, std, history=3, frameskip=5,
                 camera='agentview_rgb', image_size=224, flip_vertical=False):
        self.episodes, self.history, self.frameskip = episodes, history, frameskip
        self.camera, self.image_size, self.flip_vertical = camera, image_size, flip_vertical
        self.mean, self.std = mean, std
        self.ends = np.cumsum([e['clips'] for e in episodes]).tolist()
        self.handles = OrderedDict()

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getstate__(self):
        state = self.__dict__.copy(); state['handles'] = OrderedDict(); return state

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        i = bisect.bisect_right(self.ends, index)
        start = index - (self.ends[i-1] if i else 0)
        e = self.episodes[i]; path = e['file']
        if path not in self.handles:
            self.handles[path] = h5py.File(path, 'r')
        self.handles.move_to_end(path)
        while len(self.handles) > 8:
            self.handles.popitem(last=False)[1].close()
        g = self.handles[path][f"data/{e['demo']}"]
        h, k = self.history, self.frameskip
        times = start + np.arange(h+1)*k
        pixels = np.asarray(g[f'obs/{self.camera}'][times])
        if self.flip_vertical:
            pixels = pixels[:, ::-1].copy()
        pixels = torch.from_numpy(pixels).permute(0,3,1,2).float()/255
        pixels = F.interpolate(pixels, size=(self.image_size,self.image_size), mode='bilinear', align_corners=False, antialias=True)
        pixels = (pixels-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
        actions = np.asarray(g['actions'][start:start+h*k], dtype=np.float32)
        actions = ((actions-self.mean)/self.std).reshape(h,k*7)
        # JEPA.encode expects one action entry for each image; last is unused.
        actions = np.concatenate([actions,np.zeros((1,k*7),np.float32)])
        return {'pixels':pixels, 'action':torch.from_numpy(actions)}


def build_libero_datasets(cfg, run_dir):
    """Build episode-disjoint splits and persist training-only action statistics."""
    import json
    from omegaconf import open_dict
    if cfg.wm.num_preds != 1:
        raise ValueError('LIBERO adapter requires wm.num_preds=1 (upstream shifted loss).')
    settings = cfg.data.dataset
    train, val = make_split(settings.root, cfg.wm.history_size, settings.frameskip,
                            cfg.data.val_fraction, cfg.seed, settings.camera)
    mean, std = action_stats(train)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir/'split.json').write_text(json.dumps({'train':train,'val':val},indent=2))
    np.savez(run_dir/'action_stats.npz',mean=mean,std=std)
    with open_dict(cfg):
        cfg.wm.action_dim = 7
    datasets = [LiberoClips(s,mean,std,cfg.wm.history_size,settings.frameskip,
                           settings.camera,cfg.img_size,settings.flip_vertical) for s in (train,val)]
    print({name:{'episodes':len(s),'clips':len(d)} for name,s,d in zip(['train','val'],(train,val),datasets)},flush=True)
    if len(datasets[0]) < cfg.loader.batch_size:
        raise ValueError('Not enough train clips for batch size; reduce loader.batch_size')
    return datasets
