# PRISM-JEPA trên LIBERO — pipeline đầy đủ + tách latent theo chunk

Thư mục này đặt **trong gốc repo `prism-jepa`** (cạnh `train.py`, `prior_head.py`):

```
prism-jepa/
├── prism_libero/            ← thư mục này
└── ...
```

## 1. Vì sao `PriorHead` gốc bị giới hạn với chuỗi dài

`PriorHead(z_t, z_g)` là **một MLP duy nhất** với lớp ra `2·H·A_block·A_raw`. Với LIBERO (7 chiều action,
frameskip 5, episode 100–500 bước, goal ở rất xa):

- lớp ra phình theo `H`, mọi bước phải đoán từ đúng hai latent `(z_t, z_g)`, không có gì ràng buộc các bước muộn với cảnh *tại thời điểm đó*;
- repo gốc được huấn luyện và kiểm chứng ở `H=5` (hoặc `H=3`), tức 15–25 bước env; các biến thể chuỗi dài trong `train_prior_head.py` chủ yếu dùng "offset-mode" (goal giả cách `t` đúng `OFFSET` bước), nhưng lúc triển khai không có goal ở khoảng cách đó.

## 2. Cách tách latent (`chunk_prior_head.py`)

Chia chuỗi thành `K` chunk, mỗi chunk `C` plan-step (`C·A_block` bước env). Mỗi chunk có một **latent mốc (anchor)**:

```
a_0 = z_t                                              (quan sát được)
a_{k+1} = a_k + sigmoid(g_k)·(z_g − a_k) + r_k         (bộ bước học được; g_k, r_k = f(a_k, z_g−a_k, k))
(μ_k, σ_k) = decoder(a_k, a_{k+1}−a_k, z_g−a_k, emb(k))   → (C, A_block, A_raw)
```

- **Một decoder dùng chung** cho mọi chunk → kích thước đầu ra không phụ thuộc `K`; mỗi lần gọi là bài toán ngắn (đúng vùng head gốc làm tốt).
- Anchor được huấn luyện bằng **latent hindsight** `z_{t+k·L}` lấy từ demo (L = `C·A_block`); anchor sau cuối episode được kẹp về frame cuối (= "đứng ở goal").
- Decoder thấy anchor có nhiễu hoặc anchor do chính mô hình dự đoán (prob. `--p-pred`) để chịu được sai số lúc triển khai.
- `z_g` huấn luyện có thể bị hoán sang demo khác cùng task (`--goal-swap-prob`), mô phỏng goal bank lúc eval.
- Chọn checkpoint theo **NLL val với anchor chạy tự do** (đúng điều planner nhìn thấy).

**Planner** (`latent_mppi.py`): PoG-MPPI giữ nguyên toán của repo, nhưng nhận **sub-goal `a_1`** làm đích của chunk đầu (`--goal-mode subgoal`) thay vì ảnh goal ở xa; `--goal-mode final` giữ hành vi PRISM gốc để làm ablation.

## 3. Chạy

```bash
export STABLEWM_HOME=$HOME/.stable-wm
bash prism_libero/run_libero_pipeline.sh libero_spatial /path/to/LIBERO/libero/datasets
```

| Bước | File | Ghi chú |
|---|---|---|
| 1 | `convert_libero_to_swm.py` | hdf5 robomimic → h5 của stable-worldmodel (`pixels, action, proprio, ep_len, ep_offset`) + `*.meta.json`. Ảnh xoay 180° (cờ `--no-flip` để tắt). Action cuối mỗi episode = NaN (đúng quy ước swm). |
| 2 | `train.py data=libero ...` | cấu hình `config/train/data/libero.yaml` (frameskip 5, `proprio`). Script tự copy vào `config/`. |
| 3 | `cache_features_libero.py` | `features.npy`, `goals.npy`, `task_goals.npy` (goal bank), `action_stats.json`. |
| 4 | `train_chunk_prior_head.py` | `--arch chunked` (đề xuất) hoặc `--arch flat` (head gốc, ablation). |
| 5 | `eval_libero_prism.py` | vòng kín trên LIBERO: `pog`, `warm_start`, `none`, `bc_only`. |

Test không cần LIBERO/GPU: `python prism_libero/tests/test_pipeline.py`

## 4. Đọc báo cáo val ở bước 4

```
action MSE per chunk k=0..K-1 : ...     ← MSE tăng mạnh theo k là bình thường; so với --arch flat
anchor MSE learned / hold / linear      ← "learned" phải thấp hơn "hold" và (tốt nhất) cả "linear"
sigma ... frac_near_floor               ← σ thoái hóa sát sàn => PoG vô nghĩa (cổng chặn lưu)
```

Nếu `anchor learned` không thua `linear` thì lợi ích của việc tách latent chủ yếu đến từ decoder chia chunk, không phải đường đi học được — vẫn hợp lệ, nhưng hãy báo cáo đúng như vậy.

## 5. Giả định và giới hạn (đọc trước khi tin kết quả)

1. **LIBERO không có ảnh goal.** Goal của task = *medoid* latent frame cuối của các demo cùng task (`task_goals.npy`). Task có vị trí vật thể ngẫu nhiên nhiều thì goal này lệch; có thể thay bằng điều kiện ngôn ngữ (CLIP, repo có `clip_text_cache.py`).
2. **Phần đã kiểm thử:** converter, cache, trainer (chunked + flat), PoG-MPPI trên **dữ liệu giả + JEPA giả** (shape, mask, loss giảm, ckpt nạp lại được). Trên dữ liệu giả chunked đạt MSE val 0.38 so với flat 0.77 — đây chỉ là sanity check, **không phải bằng chứng cho LIBERO thật**.
3. **Chưa chạy được ở đây:** huấn luyện JEPA thật, và `eval_libero_prism.py` (cần GPU + LIBERO + robosuite/mujoco). Script dựa trên API chuẩn của LIBERO (`OffScreenRenderEnv`, `set_init_state`, 10 bước dummy) nên có thể cần chỉnh nhỏ theo phiên bản bạn cài.
4. Gripper được nhị phân hóa (`sign`) khi gửi vào env; tắt bằng `--no-binarize-gripper`.
5. `--receding 2` (thay vì 5) vì thao tác LIBERO cần phản hồi dày hơn; chi phí replan tăng tương ứng.

## 6. Bảng ablation nên chạy (cùng seed, ≥10 episode/task)

| Cấu hình | Lệnh |
|---|---|
| Vanilla MPPI | `--mode none` |
| BC-only (chỉ head) | `--mode bc_only` |
| PRISM, head flat, goal cuối | flat ckpt, `--mode pog --goal-mode final` |
| PRISM, head chunk, goal cuối | `--mode pog --goal-mode final` |
| **PRISM, head chunk, sub-goal** | `--mode pog --goal-mode subgoal` |
| Quét độ tin prior | `--sigma-scale 0.3 / 1 / 3` |
