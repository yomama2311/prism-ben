"""
Smoke-test Qwen2.5-VL-3B on a handful of Cube frames before committing to a
full feature-extraction run. Reports:
  - Hidden-state shape and dtype
  - Total token count, language-token slice
  - Per-frame and batch latency
  - Approximate VRAM footprint

Run:  python vlm_smoke.py
First invocation downloads ~6 GB to ~/.cache/huggingface.
"""
import os
import time
import hdf5plugin  # noqa: F401  — registers Blosc codec for h5py before File()
import h5py
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
CUBE_H5 = os.path.join(
    os.environ.get("STABLEWM_HOME", "/home/yuhai/project/prism-jepa/.stable-wm"),
    "ogbench/cube_single_expert.h5",
)
TASK_DESC = "Move the red cube to the green target position."
SAMPLE_FRAME_IDS = [0, 100, 1000, 100_000]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device} (bf16)…")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    print(f"  model loaded in {time.time()-t0:.1f}s")
    if device == "cuda":
        print(f"  vram alloc: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print(f"\nLoading {len(SAMPLE_FRAME_IDS)} frames from {CUBE_H5}…")
    with h5py.File(CUBE_H5, "r") as f:
        frames = f["pixels"][SAMPLE_FRAME_IDS]
    pil_frames = [Image.fromarray(x) for x in frames]

    print("\n--- Per-frame forward (batch=1) ---")
    for i, img in enumerate(pil_frames):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": TASK_DESC},
            ],
        }]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)

        # Warm-up + timed pass
        with torch.no_grad():
            if i == 0:
                _ = model(**inputs, output_hidden_states=True, return_dict=True)
                if device == "cuda":
                    torch.cuda.synchronize()
            t0 = time.time()
            out = model(**inputs, output_hidden_states=True, return_dict=True)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0

        hs = out.hidden_states[-1]  # (1, T, hidden)
        n_tok = inputs.input_ids.shape[1]
        print(f"  frame_id={SAMPLE_FRAME_IDS[i]:>7d}  "
              f"tokens={n_tok:>4d}  "
              f"hidden={tuple(hs.shape)}  "
              f"dtype={hs.dtype}  "
              f"latency={elapsed*1000:>5.0f} ms")

        if i == 0:
            ids = inputs.input_ids[0].tolist()
            decoded = [processor.tokenizer.decode([t]) for t in ids]
            print(f"\n    First 5 tokens : {decoded[:5]}")
            print(f"    Last 12 tokens : {decoded[-12:]}")
            print(f"    (we'll likely read out the language-token slice "
                  f"covering the prompt text)")

    # Batched pass to estimate cache throughput
    print("\n--- Batched forward (batch=4) ---")
    msgs_batch = [
        [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": TASK_DESC},
            ],
        }]
        for img in pil_frames
    ]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in msgs_batch]
    images, videos = process_vision_info(msgs_batch)
    inputs = processor(text=texts, images=images, videos=videos,
                       padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        t0 = time.time()
        out = model(**inputs, output_hidden_states=True, return_dict=True)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
    hs = out.hidden_states[-1]
    print(f"  batch_size=4  hidden={tuple(hs.shape)}  "
          f"latency={elapsed*1000:.0f} ms  "
          f"per-frame={elapsed*1000/4:.0f} ms")

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  peak vram during run: {peak:.2f} GB")
        # Throughput projection
        per_frame_s = elapsed / 4
        total_frames = 2_010_000
        est_hours = per_frame_s * total_frames / 3600
        print(f"\nProjected full-cube cache time at this batch size: "
              f"{est_hours:.1f} hours ({total_frames:,} frames)")

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
