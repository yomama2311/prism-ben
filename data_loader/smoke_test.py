r"""Kiểm tra CODE đúng logic trên máy yếu (RAM thấp, không cần GPU mạnh), KHÔNG tải checkpoint 11,8 GB
và KHÔNG dựng ViT-g thật. Thay vào đó dùng một encoder/predictor giả rất nhỏ, cùng chữ ký gọi hàm như
thật, để chạy hết một vòng: đọc dữ liệu -> encode -> dự đoán -> loss -> backward -> lưu checkpoint.

Nếu script này chạy xong không lỗi, nghĩa là logic của libero_clip_dataset.py và vjepa2_ac_libero.py
đúng. Nó KHÔNG kiểm tra việc nạp checkpoint thật hay kiến trúc ViT-g thật có khớp không - việc đó bắt
buộc phải chạy trên server đủ RAM/VRAM.

Chạy:
    python smoke_test.py --data "E:\thesis\Libero_dataset" --split_json "E:\thesis\Prism_ouput\split.json"
"""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import vjepa2_ac_libero as V
from libero_clip_dataset import LiberoClipDataset, load_split


class FakeEncoder(nn.Module):
    """Thay ViT-g: nhận x (B*F, 3, 2, RES, RES) như encoder thật, trả về (B*F, N, D) NHỎ tuỳ ý."""
    def forward(self, x):
        b = x.shape[0]
        return torch.randn(b, V.N, D_FAKE)


class FakePredictor(nn.Module):
    """Thay predictor 2-AC: cùng chữ ký gọi (z, actions, states, extrinsics) như thật, có tham số
    thật (nn.Linear) để kiểm tra backward() và optimizer.step() chạy được, không chỉ forward()."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D_FAKE, D_FAKE)

    def forward(self, z, actions, states, extrinsics):
        return self.proj(z)


D_FAKE = 8  # số chiều mỗi token của model giả, cố tình để rất nhỏ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--split_json", default=None)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--stride", type=int, default=5)
    a = p.parse_args()

    # Thu nhỏ ảnh và ép chạy CPU để không cần RAM/VRAM lớn. Đây là biến TOÀN CỤC trong module thật
    # (vjepa2_ac_libero.py dùng RES/PATCH/N/device làm biến module-level), nên override trước khi gọi
    # encode/evaluate/finetune là đủ, không cần sửa file gốc.
    V.RES, V.PATCH = 32, 16
    V.N = (V.RES // V.PATCH) ** 2
    V.device = "cpu"
    V.MEAN, V.STD = V.MEAN.cpu(), V.STD.cpu()  # MEAN/STD được tạo lúc import theo device cũ (cuda nếu máy có GPU)

    print(f"RES={V.RES} N={V.N} D_FAKE={D_FAKE} device={V.device}", flush=True)

    train_demos = load_split(a.split_json, "train") if a.split_json else None
    val_demos = load_split(a.split_json, "val") if a.split_json else None

    ds = LiberoClipDataset(a.data, num_frames=a.num_frames, stride=a.stride,
                           start_step=200, demos=val_demos or train_demos)  # start_step lớn -> ít clip, đọc nhanh
    print("số clip (đã giới hạn cho nhanh):", len(ds), flush=True)
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)

    encoder, predictor = FakeEncoder(), FakePredictor()

    print("--- evaluate() với model giả ---", flush=True)
    V.evaluate(encoder, predictor, loader, max_batches=2)

    print("--- finetune() 1 bước với model giả ---", flush=True)
    V.finetune(encoder, predictor, loader, epochs=1, save_path="smoke_predictor.pt")

    sd = torch.load("smoke_predictor.pt", map_location="cpu", weights_only=True)
    print("đã lưu và nạp lại checkpoint giả, số khoá:", len(sd), flush=True)
    print("\n=== TOÀN BỘ PIPELINE CHẠY ĐÚNG, không lỗi ===")


if __name__ == "__main__":
    main()
