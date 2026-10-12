import glob
import json
import os

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


def build_states(ee_states, gripper_states, ori_mode="euler", gripper_mode="closure", max_width=0.08,
                 unwrap_euler=True):
    """(T,6) ee_states + (T,2) gripper_states -> (T,7) state.

    ori_mode    : "euler" đổi ee_ori từ axis-angle sang Euler ngoại (extrinsic xyz, radian),
                  "raw" giữ nguyên ee_ori của LIBERO.
    gripper_mode: "closure" -> 1 - width/max_width (0 = mở, 1 = đóng, dải [0,1]);
                  "width"   -> khoảng cách 2 ngón (mét).
    unwrap_euler: khi gripper hướng xuống, roll ~ +-pi nên bị nhảy dấu giữa 2 timestep (bước nhảy ~2pi).
                  True -> làm liên tục theo thời gian (np.unwrap) rồi dịch cả demo bởi bội của 2pi để
                  roll nằm quanh +pi (cùng một quy ước cho mọi demo), pitch/yaw quanh 0.
    """
    pos, ori = ee_states[:, :3], ee_states[:, 3:6]
    if ori_mode == "euler":
        ori = R.from_rotvec(ori).as_euler("xyz")  # chữ thường = extrinsic
        if unwrap_euler:
            ori = np.unwrap(ori, axis=0)
            centers = np.array([np.pi, 0.0, 0.0])
            shift = 2 * np.pi * np.round((np.median(ori, axis=0) - centers) / (2 * np.pi))
            ori = ori - shift
    width = np.abs(gripper_states[:, 0] - gripper_states[:, 1])[:, None]  # (T, 1)
    grip = 1.0 - np.clip(width / max_width, 0.0, 1.0) if gripper_mode == "closure" else width
    return np.concatenate([pos, ori, grip], axis=1).astype(np.float32)


def state_diffs(st):
    """st: (F, 7) = [xyz, euler xyz, gripper] -> (F-1, 7) giống poses_to_diffs của repo V-JEPA 2
    (app/vjepa_droid/droid.py): hiệu vị trí, xoay tương đối R_{t+1} @ R_t^T đổi sang Euler xyz, hiệu gripper.
    """
    st = st.astype(np.float64)
    rots = R.from_euler("xyz", st[:, 3:6])
    dpos = st[1:, :3] - st[:-1, :3]
    drot = (rots[1:] * rots[:-1].inv()).as_euler("xyz")
    dgrip = st[1:, 6:7] - st[:-1, 6:7]
    return np.concatenate([dpos, drot, dgrip], axis=1).astype(np.float32)


def load_split(split_json, part):
    """Đọc split.json do libero_to_swm.py ghi ra, trả về set (tên_file, demo) của phần "train" hoặc "val",
    để LiberoClipDataset chỉ đọc đúng các demo đó (khớp với libero_train.h5 / libero_val.h5 bên PRISM)."""
    with open(split_json, encoding="utf-8") as f:
        s = json.load(f)
    return {(os.path.basename(p), d) for p, d in s[part]}


class LiberoClipDataset(Dataset):
    """Mỗi mẫu = 1 clip gồm `num_frames` frame cách nhau `stride` timestep.

    images  : (F, 3, 128, 128) uint8   <- obs/eye_in_hand_rgb
    states  : (F, 7) float32           <- [x,y,z, 3 góc xoay, gripper], cùng dạng (T,7) như DROID
    actions : (F-1, 7) float32         <- action nối frame k với frame k+1

    action_mode="state_diff": giống poses_to_diffs của V-JEPA 2-AC (hiệu vị trí, xoay tương đối, hiệu gripper).
    action_mode="sum"       : gộp action gốc của LIBERO (tổng 6 chiều delta, gripper lấy giá trị cuối).
    Clip không bao giờ vượt qua ranh giới giữa hai demo.
    """

    def __init__(self, folder, num_frames=8, stride=5, start_step=1, flip_image=True,
                 ori_mode="euler", gripper_mode="closure", max_width=0.08, action_mode="state_diff",
                 camera_key="obs/eye_in_hand_rgb", demos=None):
        # camera_key: "obs/eye_in_hand_rgb" (camera cổ tay) hoặc "obs/agentview_rgb" (camera tĩnh bên ngoài).
        # Đổi tham số này để so sánh camera nào khớp hơn với predictor pretrained, không cần sửa code.
        # demos: set (tên_file, demo) từ load_split(...), để chỉ dùng đúng tập train/val giống PRISM.
        #        Để None thì dùng mọi demo trong folder (cách cũ, KHÔNG khớp split khi so sánh với PRISM).
        self.files = sorted(glob.glob(f"{folder}/**/*.hdf5", recursive=True))
        assert self.files, f"Không tìm thấy file .hdf5 nào trong {folder}"
        self.num_frames, self.stride, self.flip_image = num_frames, stride, flip_image
        self.ori_mode, self.action_mode, self.camera_key = ori_mode, action_mode, camera_key
        self.span = (num_frames - 1) * stride
        self.index, self.lowdim, self._handles = [], {}, {}

        for fi, path in enumerate(self.files):
            fname = os.path.basename(path)
            with h5py.File(path, "r") as f:
                for demo in f["data"].keys():
                    if demos is not None and (fname, demo) not in demos:
                        continue
                    g = f["data"][demo]
                    actions = g["actions"][:].astype(np.float32)
                    states = build_states(g["obs/ee_states"][:], g["obs/gripper_states"][:],
                                          ori_mode, gripper_mode, max_width)
                    self.lowdim[(fi, demo)] = (actions, states)
                    self.index += [(fi, demo, s) for s in range(0, len(actions) - self.span, start_step)]

    def __len__(self):
        return len(self.index)

    def _file(self, fi):
        if fi not in self._handles:
            self._handles[fi] = h5py.File(self.files[fi], "r")
        return self._handles[fi]

    def __getitem__(self, i):
        fi, demo, s = self.index[i]
        e = s + self.span
        img = self._file(fi)["data"][demo][self.camera_key][s : e + 1 : self.stride]
        if self.flip_image:
            img = img[:, ::-1, ::-1]
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(0, 3, 1, 2)

        raw_actions, states = self.lowdim[(fi, demo)]
        st = states[s : e + 1 : self.stride]  # (F, 7)

        if self.action_mode == "state_diff":
            assert self.ori_mode == "euler", "state_diff cần ori_mode='euler'"
            acts = state_diffs(st)
        else:
            acts = np.empty((self.num_frames - 1, 7), dtype=np.float32)
            for k in range(self.num_frames - 1):
                seg = raw_actions[s + k * self.stride : s + (k + 1) * self.stride]
                acts[k, :6] = seg[:, :6].sum(0)
                acts[k, 6] = seg[-1, 6]

        return {"images": img, "states": torch.from_numpy(st), "actions": torch.from_numpy(np.ascontiguousarray(acts))}
