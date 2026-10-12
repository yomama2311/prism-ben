r"""Chạy từ command prompt, ví dụ:

    python run_train.py --data "E:\path\to\folder" --split_json "E:\thesis\Prism_ouput\split.json" --mode eval --repo ... --ckpt ...
    python run_train.py --data "E:\path\to\folder" --split_json "E:\thesis\Prism_ouput\split.json" --mode train --epochs 1 --batch_size 2 --repo ... --ckpt ...
    python run_train.py --data "E:\path\to\folder" --split_json "E:\thesis\Prism_ouput\split.json" --mode eval --load predictor_finetuned.pt --repo ... --ckpt ...

--split_json trỏ tới split.json do libero_to_swm.py ghi ra: dùng đúng tập train để fine-tune và
đúng tập val để đánh giá, khớp với libero_train.h5 / libero_val.h5 mà PRISM-JEPA dùng, để so sánh
hai model công bằng trên cùng một cách chia dữ liệu. Bỏ --split_json thì dùng toàn bộ --data
(cách cũ, KHÔNG khớp split khi so sánh với PRISM).
"""
import argparse

import torch
from torch.utils.data import DataLoader

from libero_clip_dataset import LiberoClipDataset, load_split
from vjepa2_ac_libero import evaluate, finetune, load_model


def make_loader(a, demos, batch_size, shuffle):
    ds = LiberoClipDataset(a.data, num_frames=a.num_frames, stride=a.stride,
                           start_step=a.start_step, demos=demos)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=a.num_workers,
                          drop_last=True, pin_memory=torch.cuda.is_available())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder chứa các file .hdf5")
    p.add_argument("--split_json", default=None, help="split.json do libero_to_swm.py ghi ra")
    p.add_argument("--mode", choices=["eval", "train"], default="eval")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--start_step", type=int, default=5, help="bước nhảy giữa hai điểm bắt đầu clip")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save", default="predictor_finetuned.pt")
    p.add_argument("--load", default=None, help="nạp trọng số predictor đã fine-tune")
    p.add_argument("--repo", default=None, help="thư mục repo vjepa2 trên máy, ví dụ E:\\thesis\\vjepa2-main")
    p.add_argument("--ckpt", default=None, help="file checkpoint V-JEPA 2-AC, ví dụ E:\\vjepa2-ac-vitg.pt")
    a = p.parse_args()

    print("cuda:", torch.cuda.is_available(), flush=True)

    train_demos = load_split(a.split_json, "train") if a.split_json else None
    val_demos = load_split(a.split_json, "val") if a.split_json else None

    ds_val, loader_val = make_loader(a, val_demos, a.batch_size, shuffle=True)
    print(f"số clip val{' (theo split.json)' if val_demos else ''}: {len(ds_val)}", flush=True)

    encoder, predictor = load_model(a.repo, a.ckpt)
    if a.load:
        predictor.load_state_dict(torch.load(a.load, map_location="cpu"))

    print("--- đánh giá trên tập val, trước khi train ---", flush=True)
    evaluate(encoder, predictor, loader_val, max_batches=a.eval_batches)

    if a.mode == "train":
        ds_tr, loader_tr = make_loader(a, train_demos, a.batch_size, shuffle=True)
        print(f"số clip train{' (theo split.json)' if train_demos else ''}: {len(ds_tr)}", flush=True)
        finetune(encoder, predictor, loader_tr, epochs=a.epochs, lr=a.lr, save_path=a.save)
        print("--- đánh giá trên tập val, sau khi train ---", flush=True)
        evaluate(encoder, predictor, loader_val, max_batches=a.eval_batches)


if __name__ == "__main__":  # bắt buộc trên Windows khi num_workers > 0
    main()
