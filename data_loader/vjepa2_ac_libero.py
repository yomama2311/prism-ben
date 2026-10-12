"""V-JEPA 2-AC pretrained (ViT-g) trên clip LIBERO.

Cách dùng trong notebook:
    from vjepa2_ac_libero import *
    encoder, predictor = load_model()
    ds = LiberoClipDataset(r"E:\\path\\to\\folder", num_frames=8, stride=5)
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    evaluate(encoder, predictor, loader, max_batches=20)      # 1) đo zero-shot trước
    finetune(encoder, predictor, loader, epochs=1)            # 2) rồi mới fine-tune predictor

Logic encode / predict / loss bám theo app/vjepa_droid/train.py của repo facebookresearch/vjepa2.
"""
import zipfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from libero_clip_dataset import LiberoClipDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
RES, PATCH = 256, 16
N = (RES // PATCH) ** 2  # số token mỗi frame = 256
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)  # ImageNet
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
NORMALIZE_REPS = True  # tương ứng loss.normalize_reps trong config droid-256px-8f.yaml, hãy đối chiếu


def _clean_keys(sd):
    """Bỏ tiền tố 'module.' và 'backbone.' giống _clean_backbone_key của repo V-JEPA 2."""
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}


def _load_ckpt(path):
    """Nạp checkpoint lên CPU. KHÔNG dùng mmap=True: trên Windows, torch.load(mmap=True) với file lớn
    (checkpoint này ~11,8 GB) có thể gây access violation (crash cứng, không phải lỗi Python bắt được) -
    đây là lỗi đã biết của PyTorch trên Windows với file dung lượng lớn, không liên quan đến RAM máy."""
    last = None
    for kw in ({"weights_only": True}, {"weights_only": False}):
        try:
            return torch.load(path, map_location="cpu", **kw)
        except Exception as e:
            last = e
            print(f"torch.load({kw}) lỗi: {type(e).__name__}: {str(e)[:120]}", flush=True)
    raise last


def load_model(repo=None, ckpt=None):
    """repo: thư mục vjepa2 trên máy (có hubconf.py); ckpt: file vjepa2-ac-vitg.pt tải sẵn.

    Không truyền gì -> tải qua torch.hub, hiện KHÔNG chạy được vì src/hub/backbones.py của repo
    đang trỏ VJEPA_BASE_URL tới http://localhost:8300 (dòng 'for testing'). Vì vậy dùng repo + ckpt cục bộ:
    dựng mô hình với pretrained=False rồi nạp trọng số bằng đúng các bước của _make_vjepa2_ac_model.
    """
    if repo is None and ckpt is None:
        encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
    else:
        assert repo and ckpt, "cần truyền cả repo và ckpt"
        encoder, predictor = torch.hub.load(repo, "vjepa2_ac_vit_giant", source="local", pretrained=False)
        if not zipfile.is_zipfile(ckpt):
            raise RuntimeError(
                f"{ckpt} không phải file checkpoint PyTorch hợp lệ (thường do tải chưa đủ hoặc bị hỏng). "
                "Kiểm tra dung lượng so với Content-Length của máy chủ rồi tải lại."
            )
        sd = _load_ckpt(ckpt)
        msg = encoder.load_state_dict(_clean_keys(sd["encoder"]), strict=False)  # repo gốc cũng dùng strict=False
        print(f"encoder: thiếu {len(msg.missing_keys)} khoá, thừa {len(msg.unexpected_keys)} khoá", flush=True)
        if msg.missing_keys or msg.unexpected_keys:
            print("  ví dụ thiếu:", msg.missing_keys[:3], "| ví dụ thừa:", msg.unexpected_keys[:3], flush=True)
        predictor.load_state_dict(_clean_keys(sd["predictor"]), strict=True)
        del sd
    encoder.to(device).eval().requires_grad_(False)
    predictor.to(device)
    return encoder, predictor


@torch.no_grad()
def encode(encoder, images_u8):
    """(B, F, 3, 128, 128) uint8 -> (B, F*N, D). Mỗi frame được encode riêng: nhân đôi theo trục thời gian
    để khớp tubelet_size=2, giống forward_target trong train.py của repo."""
    B, Fr = images_u8.shape[:2]
    x = images_u8.flatten(0, 1).float() / 255.0
    x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
    x = ((x - MEAN) / STD).unsqueeze(2).repeat(1, 1, 2, 1, 1)  # (B*F, 3, 2, H, W)
    h = encoder(x)                                             # (B*F, N, D)
    h = h.view(B, Fr, -1, h.size(-1)).flatten(1, 2)            # (B, F*N, D)
    return F.layer_norm(h, (h.size(-1),)) if NORMALIZE_REPS else h


def _step(predictor, z, a, s):
    z = predictor(z, a, s, None)  # train.py luôn truyền tham số thứ 4 (extrinsics); ở đây không dùng
    return F.layer_norm(z, (z.size(-1),)) if NORMALIZE_REPS else z


def forward_predictions(predictor, h, actions, states, auto_steps=2):
    """h: (B, F*N, D); actions: (B, F-1, 7); states: (B, F, 7).
    Trả về (z_tf, z_ar): dự đoán teacher-forcing 1 bước và rollout tự hồi quy."""
    z_tf = _step(predictor, h[:, :-N], actions, states[:, :-1])
    z = torch.cat([h[:, :N], z_tf[:, :N]], dim=1)
    for n in range(1, auto_steps):
        z_next = _step(predictor, z, actions[:, : n + 1], states[:, : n + 1])[:, -N:]
        z = torch.cat([z, z_next], dim=1)
    return z_tf, z[:, N:]


def l1(z, h):
    return (z - h[:, N : z.size(1) + N]).abs().mean()


def _batch(batch):
    return (batch["images"].to(device), batch["states"].to(device).float(), batch["actions"].to(device).float())


@torch.no_grad()
def evaluate(encoder, predictor, loader, max_batches=20, auto_steps=2):
    """L1 zero-shot của predictor so với baseline 'chép frame trước'. Nếu predictor không thấp hơn
    baseline rõ rệt thì đầu vào (state/action/camera) đang lệch với DROID."""
    predictor.eval()
    tf, ar, copy = [], [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        images, states, actions = _batch(batch)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            h = encode(encoder, images)
            z_tf, z_ar = forward_predictions(predictor, h, actions, states, auto_steps)
            tf.append(l1(z_tf, h).item()); ar.append(l1(z_ar, h).item()); copy.append(l1(h[:, :-N], h).item())
    m = lambda v: sum(v) / len(v)
    print(f"L1 teacher-forcing: {m(tf):.4f} | rollout: {m(ar):.4f} | baseline chép frame trước: {m(copy):.4f}")


def finetune(encoder, predictor, loader, epochs=1, lr=1e-5, auto_steps=2, save_path=None):
    predictor.train()
    opt = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=0.04)
    for epoch in range(epochs):
        for it, batch in enumerate(loader):
            images, states, actions = _batch(batch)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                h = encode(encoder, images)
                z_tf, z_ar = forward_predictions(predictor, h, actions, states, auto_steps)
                loss = l1(z_tf, h) + l1(z_ar, h)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if it % 20 == 0:
                print(f"epoch {epoch} it {it}/{len(loader)} loss {loss.item():.4f}", flush=True)
        if save_path:  # lưu trọng số predictor sau mỗi epoch
            torch.save(predictor.state_dict(), save_path)
            print("đã lưu:", save_path, flush=True)
