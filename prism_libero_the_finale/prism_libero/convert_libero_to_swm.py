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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero-dir", required=True, help="folder with <task>_demo.hdf5 files")
    ap.add_argument("--out", required=True, help="output .h5 path")
    ap.add_argument("--camera", default="agentview_rgb",
                    help="obs key to use as pixels (agentview_rgb | eye_in_hand_rgb)")
    ap.add_argument("--img-size", type=int, default=0,
                    help="resize to SxS when storing; 0 keeps the native 128 (LeWM resizes at load)")
    ap.add_argument("--no-flip", action="store_true", help="do not rotate images by 180 degrees")
    ap.add_argument("--max-demos-per-task", type=int, default=0, help="0 = all demos")
    ap.add_argument("--max-tasks", type=int, default=0, help="0 = all tasks (debug)")
    args = ap.parse_args()
    flip = not args.no_flip

    files = sorted(Path(args.libero_dir).glob("*_demo.hdf5"))
    if args.max_tasks:
        files = files[: args.max_tasks]
    assert files, f"no *_demo.hdf5 found in {args.libero_dir}"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    task_names, ep_task, ep_len, ep_offset = [], [], [], []
    ptr = 0
    ds = {}

    with h5py.File(out, "w") as fo:
        for tid, fp in enumerate(files):
            name = fp.name[: -len("_demo.hdf5")]
            task_names.append(name)
            with h5py.File(fp, "r") as fi:
                keys = _demo_keys(fi)
                if args.max_demos_per_task:
                    keys = keys[: args.max_demos_per_task]
                for dk in keys:
                    g = fi["data"][dk]
                    acts = g["actions"][:].astype(np.float32)
                    T = len(acts)
                    obs = g["obs"]
                    pix = _prep_images(obs[args.camera][:], flip, args.img_size)
                    proprio = np.concatenate(
                        [obs["ee_pos"][:], obs["ee_ori"][:], obs["gripper_states"][:]], axis=1
                    ).astype(np.float32)
                    assert len(pix) == T == len(proprio), f"{fp.name}/{dk}: length mismatch"
                    acts[-1] = np.nan  # stable-worldmodel: last action has no successor

                    if not ds:  # first episode defines the schema
                        S = pix.shape[1]
                        stored_size, action_dim = int(S), int(acts.shape[1])
                        ds["pixels"] = fo.create_dataset(
                            "pixels", shape=(0, S, S, 3), maxshape=(None, S, S, 3),
                            dtype=np.uint8, chunks=(16, S, S, 3), compression="lzf")
                        ds["action"] = fo.create_dataset(
                            "action", shape=(0, acts.shape[1]), maxshape=(None, acts.shape[1]),
                            dtype=np.float32, chunks=(1024, acts.shape[1]))
                        ds["proprio"] = fo.create_dataset(
                            "proprio", shape=(0, proprio.shape[1]), maxshape=(None, proprio.shape[1]),
                            dtype=np.float32, chunks=(1024, proprio.shape[1]))
                    for key, arr in (("pixels", pix), ("action", acts), ("proprio", proprio)):
                        ds[key].resize(ptr + T, axis=0)
                        ds[key][ptr: ptr + T] = arr
                    ep_task.append(tid)
                    ep_len.append(T)
                    ep_offset.append(ptr)
                    ptr += T
            print(f"[convert] task {tid:>3d} {name}: {len(keys)} demos  (total frames {ptr})")

        fo.create_dataset("ep_len", data=np.asarray(ep_len, dtype=np.int32))
        fo.create_dataset("ep_offset", data=np.asarray(ep_offset, dtype=np.int64))

    meta = dict(
        task_names=task_names, ep_task_id=ep_task, n_frames=ptr, n_episodes=len(ep_len),
        camera=args.camera, flip=flip, stored_img_size=stored_size,
        action_dim=action_dim,
    )
    with open(str(out) + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[convert] wrote {out}  ({len(ep_len)} episodes, {ptr} frames, {len(task_names)} tasks)")


if __name__ == "__main__":
    main()
