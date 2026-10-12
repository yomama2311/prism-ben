"""Chuyển dữ liệu LIBERO (hdf5, data/demo_i/...) sang định dạng stable-worldmodel mà PRISM-JEPA / LeWM đọc.

Định dạng đích (theo stable_worldmodel.data.formats.hdf5.HDF5Dataset):
    một file .h5 phẳng với các cột nối liền mọi episode:
      pixels  (N, H, W, 3) uint8      action (N, 7) float32      proprio (N, 7) float32
      ep_len (n_ep,)                  ep_offset (n_ep,)          <- độ dài và vị trí bắt đầu của từng episode

Chia train/val THEO DEMO (không theo clip) để dùng chung một tập giữ lại khi so với V-JEPA 2-AC.
Danh sách demo của mỗi phần được lưu trong split.json.

Ví dụ:
    python libero_to_swm.py --src "E:\\path\\to\\libero_folder" --out_dir "E:\\stablewm\\datasets\\libero"
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np

from libero_clip_dataset import build_states  # cùng một hàm tạo state 7 chiều như bên V-JEPA


def scan(src, seed, val_frac):
    files = sorted(glob.glob(f"{src}/**/*.hdf5", recursive=True))
    assert files, f"Không thấy file .hdf5 trong {src}"
    rng = np.random.default_rng(seed)
    split = {"train": [], "val": []}
    for path in files:
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys())
            lens = {d: int(f["data"][d]["actions"].shape[0]) for d in demos}
        n_val = int(round(len(demos) * val_frac))
        val = set(rng.permutation(len(demos))[:n_val].tolist())
        for i, d in enumerate(demos):
            split["val" if i in val else "train"].append((path, d, lens[d]))
    return split


def write_h5(out_path, items, camera_key, flip, gripper_mode, max_width):
    total = sum(t for _, _, t in items)
    with h5py.File(out_path, "w", libver="latest") as out:  # libver=latest để mở được bằng swmr=True
        pix = ac = pr = None
        ptr, lens, offs = 0, [], []
        for path, demo, T in items:
            with h5py.File(path, "r") as f:
                g = f["data"][demo]
                img = g[camera_key][:]
                if flip:
                    img = img[:, ::-1, ::-1]
                img = np.ascontiguousarray(img)
                act = g["actions"][:].astype(np.float32)
                st = build_states(g["obs/ee_states"][:], g["obs/gripper_states"][:],
                                  gripper_mode=gripper_mode, max_width=max_width)
            if pix is None:
                H, W = img.shape[1:3]
                pix = out.create_dataset("pixels", (total, H, W, 3), "uint8", chunks=(1, H, W, 3), compression="lzf")
                ac = out.create_dataset("action", (total, act.shape[1]), "float32")
                pr = out.create_dataset("proprio", (total, st.shape[1]), "float32")
            pix[ptr:ptr + T], ac[ptr:ptr + T], pr[ptr:ptr + T] = img, act, st
            offs.append(ptr); lens.append(T); ptr += T
        out.create_dataset("ep_len", data=np.array(lens, dtype=np.int64))
        out.create_dataset("ep_offset", data=np.array(offs, dtype=np.int64))


def verify(path, frameskip=5, num_steps=4):
    """Kiểm tra theo đúng cách HDF5Dataset đọc: mở swmr, cột thẳng hàng, đếm clip, lát cắt mẫu."""
    with h5py.File(path, "r", swmr=True) as f:
        ep_len, ep_off = f["ep_len"][:], f["ep_offset"][:]
        N = int(ep_len.sum())
        assert f["pixels"].shape[0] == f["action"].shape[0] == f["proprio"].shape[0] == N
        assert (ep_off[1:] == np.cumsum(ep_len)[:-1]).all() and ep_off[0] == 0
        span = num_steps * frameskip
        n_clips = int(sum(max(0, l - span + 1) for l in ep_len))
        s = int(ep_off[0]); a = f["action"][s:s + span]
        print(f"{os.path.basename(path)}: {len(ep_len)} episode | {N} frame | {n_clips} clip "
              f"(frameskip={frameskip}, num_steps={num_steps}) | pixels {f['pixels'].shape} | "
              f"action reshape -> {a.reshape(num_steps, -1).shape}")
        assert not np.isnan(f["action"][:]).any() and not np.isnan(f["proprio"][:]).any()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--camera_key", default="obs/eye_in_hand_rgb", help="hoặc obs/agentview_rgb")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_flip", action="store_true", help="không xoay ảnh 180 độ")
    p.add_argument("--gripper_mode", default="closure")
    p.add_argument("--max_width", type=float, default=0.08)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    split = scan(a.src, a.seed, a.val_frac)
    for name in ("train", "val"):
        items = split[name]
        if not items:
            continue
        out = os.path.join(a.out_dir, f"libero_{name}.h5")
        print(f"ghi {out}: {len(items)} demo", flush=True)
        write_h5(out, items, a.camera_key, not a.no_flip, a.gripper_mode, a.max_width)
        verify(out)
    with open(os.path.join(a.out_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump({k: [[p, d] for p, d, _ in v] for k, v in split.items()}, f, indent=1)
    print("đã lưu split.json")


if __name__ == "__main__":
    main()
