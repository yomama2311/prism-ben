"""Step 1 — convert LIBERO demo files (robomimic-style .hdf5) into the flat HDF5 layout that
stable-worldmodel's `HDF5Dataset` (and therefore LeWM `train.py`) reads.

Output  <out>.h5:
    pixels    (N, S, S, 3) uint8     agent-view RGB (upright if --flip, the default)
    action    (N, 7) float32         env actions; the LAST row of every episode is NaN
                                     (stable-worldmodel convention, train.py nan_to_num's it)
    proprio   (N, 8) float32         ee_pos(3) + ee_ori(3) + gripper_states(2)
    ep_len    (n_ep,) int32
    ep_offset (n_ep,) int64
Sidecar <out>.meta.json: task names, per-episode task id, image size, flip flag, ...

Usage:
    python prism_libero/convert_libero_to_swm.py \
        --libero-dir /path/to/LIBERO/libero/datasets/libero_spatial \
        --out $STABLEWM_HOME/libero/libero_spatial.h5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _demo_keys(f: h5py.File) -> list[str]:
    keys = list(f["data"].keys())
    return sorted(keys, key=lambda k: int(k.split("_")[-1]))


def _prep_images(imgs: np.ndarray, flip: bool, size: int) -> np.ndarray:
    if flip:  # LIBERO renders images rotated by 180 degrees
        imgs = imgs[:, ::-1, ::-1]
    if size and imgs.shape[1] != size:
        import cv2  # only needed when resizing
        interp = cv2.INTER_AREA if size < imgs.shape[1] else cv2.INTER_LINEAR
        imgs = np.stack([cv2.resize(np.ascontiguousarray(i), (size, size), interpolation=interp)
                         for i in imgs])
    return np.ascontiguousarray(imgs, dtype=np.uint8)


class FlatWriter:
    """Append complete episodes to an SWM-compatible HDF5 and metadata sidecar."""
    def __init__(self, out, camera, flip):
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.out, "w")
        self.ds = {}
        self.ptr = 0
        self.camera, self.flip = camera, flip
        self.ep_len, self.ep_offset, self.ep_task = [], [], []
        self.task_names = []
        self.stored_size, self.action_dim = None, None

    def append(self, tid, pix, acts, proprio):
        length = len(acts)
        if length != len(pix) or length != len(proprio) or length < 2:
            raise ValueError("Inconsistent/too short LIBERO demo")
        if not self.ds:
            size = int(pix.shape[1]); actdim = int(acts.shape[1])
            self.stored_size, self.action_dim = size, actdim
            self.ds["pixels"] = self.f.create_dataset(
                "pixels", (0, size, size, 3), maxshape=(None, size, size, 3),
                dtype=np.uint8, chunks=(16, size, size, 3), compression="lzf")
            for key, arr in (("action", acts), ("proprio", proprio)):
                dim = arr.shape[1]
                self.ds[key] = self.f.create_dataset(key, (0, dim), maxshape=(None, dim),
                                                       dtype=np.float32, chunks=(1024, dim))
        for key, arr in (("pixels", pix), ("action", acts), ("proprio", proprio)):
            ds = self.ds[key]
            ds.resize(self.ptr + length, axis=0)
            ds[self.ptr:self.ptr+length] = arr
        self.ep_offset.append(self.ptr)
        self.ep_len.append(length)
        self.ep_task.append(int(tid))
        self.ptr += length

    def close(self):
        if not self.ds:
            self.f.close()
            raise ValueError(f"No episodes saved to {self.out}")
        self.f.create_dataset("ep_len", data=np.asarray(self.ep_len, dtype=np.int32))
        self.f.create_dataset("ep_offset", data=np.asarray(self.ep_offset, dtype=np.int64))
        self.f.close()
        payload = dict(task_names=self.task_names, ep_task_id=self.ep_task,
                       n_frames=self.ptr, n_episodes=len(self.ep_len),
                       camera=self.camera, flip=self.flip,
                       stored_img_size=self.stored_size, action_dim=self.action_dim)
        Path(str(self.out) + ".meta.json").write_text(json.dumps(payload, indent=2))
        print(f"[convert] wrote {self.out}: episodes={len(self.ep_len)} frames={self.ptr}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--libero-dir", required=True)
    ap.add_argument("--out", required=True, help="training/complete .h5")
    ap.add_argument("--val-out", default=None, help="optional episode-disjoint val .h5")
    ap.add_argument("--val-frac", type=float, default=0., help="0 disables split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--camera", default="agentview_rgb", choices=["agentview_rgb", "eye_in_hand_rgb"])
    ap.add_argument("--img-size", type=int, default=0)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--max-demos-per-task", type=int, default=0)
    ap.add_argument("--max-tasks", type=int, default=0)
    args = ap.parse_args()
    if not 0 <= args.val_frac < 1:
        ap.error("--val-frac must be [0,1)")
    if args.val_frac and not args.val_out:
        ap.error("--val-out required if --val-frac > 0")
    if args.val_out and Path(args.val_out).resolve() == Path(args.out).resolve():
        ap.error("train and validation HDF5 paths must differ")
    flip = not args.no_flip
    files = sorted(Path(args.libero_dir).glob("*_demo.hdf5"))
    if args.max_tasks:
        files = files[:args.max_tasks]
    if not files:
        raise FileNotFoundError(f"No *_demo.hdf5 in {args.libero_dir}")
    rng = np.random.default_rng(args.seed)
    writers = [FlatWriter(args.out, args.camera, flip)]
    if args.val_frac:
        writers.append(FlatWriter(args.val_out, args.camera, flip))
    for tid, fp in enumerate(files):
        name = fp.name[:-len("_demo.hdf5")]
        for writer in writers:
            writer.task_names.append(name)
        with h5py.File(fp, "r") as fi:
            keys = _demo_keys(fi)
            if args.max_demos_per_task:
                keys = keys[:args.max_demos_per_task]
            heldout = set()
            if args.val_frac and len(keys) > 1:
                n_val = max(1, int(round(len(keys)*args.val_frac)))
                n_val = min(n_val, len(keys)-1)
                heldout = set(rng.choice(len(keys), size=n_val, replace=False).tolist())
            for i, dk in enumerate(keys):
                g = fi["data"][dk]
                acts = g["actions"][:].astype(np.float32)
                obs = g["obs"]
                pix = _prep_images(obs[args.camera][:], flip, args.img_size)
                proprio = np.concatenate(
                    [obs["ee_pos"][:], obs["ee_ori"][:], obs["gripper_states"][:]], axis=1
                ).astype(np.float32)
                if len(acts) < 2 or len(pix) != len(acts) or len(proprio) != len(acts):
                    raise ValueError(f"Bad trajectory {fp}/{dk}")
                acts[-1] = np.nan  # no transition after terminal frame
                w = writers[1] if i in heldout else writers[0]
                w.append(tid, pix, acts, proprio)
        print(f"[convert] task={tid} {name} demos={len(keys)} val={len(heldout)}", flush=True)
    for writer in writers:
        writer.close()
    if len(writers) > 1:
        assert set(writers[0].ep_task) == set(writers[1].ep_task), \
            "Episode-disjoint split needs >1 demo per task for every task"
        print("[convert] episode-disjoint train/val HDF5 files written", flush=True)


if __name__ == "__main__":
    main()
