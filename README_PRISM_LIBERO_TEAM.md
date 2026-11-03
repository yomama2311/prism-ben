# PRISM-JEPA × LIBERO-Object — Hướng dẫn dữ liệu, huấn luyện và bàn giao thí nghiệm

> **Dành cho teammate / mentor** · Cập nhật: **17/10/2026**  
> **Mục tiêu:** Giải thích chính xác baseline mà nhóm đã thực hiện, các lỗi/giới hạn đã phát hiện, và giúp người tiếp nhận thiết kế **một hướng huấn luyện khác**, tránh lặp lại thí nghiệm đã chạy.  
> **Phạm vi:** LIBERO-Object; ví dụ simulator task 0: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`.
>
> **Trạng thái:** Đây là **baseline nghiên cứu đang debug**, **chưa phải** hệ thống PRISM đạt task success trên LIBERO. Một số giá trị đánh giá chỉ từ **1 episode** hoặc **512 mẫu training**; không coi là kết quả benchmark tổng quát.

## Đọc nhanh trong 2 phút

| Thành phần | Cách làm **đã chạy** | Trạng thái / kết quả |
|---|---|---|
| Nguồn | Demonstrations LIBERO-Object `.hdf5` | Converter chia demo theo episode |
| Chuyển đổi | `data_loader/libero_to_swm.py` → flat HDF5 | Train: **5.850 eps / 906.476 frames**; Val: **650 eps / 101.142 frames** |
| Visual input | Camera `eye_in_hand`, RGB 128×128, xoay 180° theo mặc định converter | Train transform: ImageNet ToImage → resize 224×224 |
| Action | Raw controller action 7 chiều; gom **5 raw actions/block** | LeWM `action_encoder` nhận vector 35 chiều/block |
| **LeWM (world model)** | **Train từ đầu trên LIBERO**, history 3, predict 1 block, **100 epochs** | Checkpoint `stablewm/lewm_epoch_100_object.ckpt`; cần xác minh lại chất lượng dynamics |
| Latent cache | Encoder+projector LeWM epoch 100 → cache `(N,192)` | `features.npy` `(906476,192)`, `goals.npy` `(5850,192)`; cache/checkpoint khớp |
| **Gaussian Prior Head** | Train từ đầu trên **frozen cache**; input `(z_t,z_goal)` → Gaussian action plan | Best epoch **4**; normalized val action MSE **0.52204** |
| Planner | PoG prior fusion + **σ-frozen MPPI**, H=5 blocks × 5 actions | K=32/K=128 đều chưa thành công ở các episode đã thử |
| Simulator task 0 | 300 raw steps, một initial state | BC-only đến **9,21 cm**; MPPI K=32 gần nhất **26,59 cm**; cả hai **0/1 success** |

**Điểm cần kiểm chứng trước khi tuyên bố LeWM “học yếu”:** offline prediction MSE sau train chỉ hơn baseline giữ nguyên latent ≈ **1,50%**, nhưng `train.py` dùng cơ chế chia validation theo clip; cần so sánh lại bằng **cùng sampling, target alignment và scaler trong training** trên **episode validation độc lập**. Không dùng riêng train loss để suy ra task success.

---

## 1. Kiến trúc pipeline đang dùng

```text
LIBERO raw dataset (demonstration .hdf5)
      │
      ├─ data_loader/libero_to_swm.py
      │     ├─ libero_train.h5  ──► train.py ──► LeWM epoch 100
      │     ├─ libero_val.h5         (pixel encoder + projector + action encoder + predictor)
      │     └─ split.json                  │
      │                                    ▼
      │                            jepa_cache_libero.py
      │                                    │
      │                         features.npy / goals.npy
      │                                    │
      │                                    ▼
      └────────────────────────► train_prior_head_libero.py
                                           │
                                    Gaussian Prior Head
                                    μ, σ | z_t, z_goal
                                           │
                                           ▼
                               PoG fusion + σ-frozen MPPI
                                           │
                          LeWM rollout + terminal goal latent MSE
                                           │
                                           ▼
                                 LIBERO simulator actions
```

**Hai khối được train khác nhau, không phải hai lần train chung một model:**

- **LeWM / JEPA world model:** train **end-to-end** encoder, projector, action encoder, predictor để dự đoán **latent tương lai**. Không train một action policy ở bước này.
- **Prior Head:** train **sau LeWM**, sử dụng embedding LeWM **đóng băng** để dự đoán phân phối Gaussian `μ, σ` của **chuỗi action**. Nó là một *goal-conditioned imitation/action prior*; không phải RL policy được tối ưu bằng reward.
- **MPPI:** không train bằng gradient trong lần chạy này; nó là thuật toán **planning tại inference**, dùng LeWM dự đoán latent tương lai và một hàm cost để chọn action.

Đây là bản **LIBERO-adapted** của PRISM-MPPI (`eval_lib/eval_libero_prism_mppi.py`), **không phải** `stable_worldmodel.solver.MPPISolver` nguyên bản. Khi báo cáo, cần ghi rõ khác biệt này.

## 2. Repo, môi trường và vị trí file

### 2.1. Đường dẫn đã dùng trên Vast.ai

```bash
ROOT=/workspace/prism-ben
LIBERO_SRC=/workspace/FA26_RE/LIBERO
RAW_DATA=/workspace/FA26_RE/datasets/libero_object
DATA_DIR=/workspace/prism-ben/prism-output/dataset_lib
STABLEWM_HOME=/workspace/prism-ben/stablewm
```

Code fork: <https://github.com/yomama2311/prism-ben>  
Code PRISM upstream: <https://github.com/YuhaiW/prism-jepa>

Ví dụ môi trường phiên chạy: Vast.ai **NVIDIA RTX 5090 (~32 GiB VRAM)**, `(prism)` Python 3.10. Trước khi chạy, teammate phải kiểm tra **phiên bản thư viện thực tế** và không tự thay toàn bộ environment đang dùng.

```bash
cd /workspace/prism-ben
# Kích hoạt môi trường 'prism' hiện có; đường dẫn thực tế ở lần chạy trước là /venv/prism/bin/python
export STABLEWM_HOME=/workspace/prism-ben/stablewm
export PYTHONPATH="/workspace/FA26_RE/LIBERO:/workspace/prism-ben:${PYTHONPATH:-}"
export MUJOCO_GL=egl
# Chỉ khi cần load init-state pickle từ bộ LIBERO tin cậy:
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> Chỉ `torch.load(..., weights_only=False)` với checkpoint do nhóm tạo/tin cậy. `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` cũng có tác động về bảo mật; không dùng với file không rõ nguồn gốc.

### 2.2. Bản đồ file

| File | Vai trò |
|---|---|
| `data_loader/libero_to_swm.py` | Đọc raw LIBERO, tạo `libero_train.h5`, `libero_val.h5`, `split.json` |
| `data_loader/libero_clip_dataset.py` | Hàm `build_states(...)` tạo proprio 7D cho converter; các chế độ action `state_diff/sum` khác trong file này **không phải** action dùng để train LeWM |
| `config/train/data/libero.yaml` | HDF5 path, `frameskip=5`, `num_steps=history_size+num_preds` |
| `config/train/lewm.yaml` | Hyperparameters LeWM/Hydra; **kiểm tra file và config snapshot của run** trước khi tái lập |
| `train.py` | Train LeWM: `pred_loss + λ·sigreg_loss` |
| `jepa.py`, `module.py`, `utils.py` | JEPA, predictor, action encoder, image transform/normalizer |
| `jepa_cache_libero.py` | Encode toàn bộ frame + endframe goal bằng LeWM đã train |
| `prior_head.py` | MLP Gaussian action prior và β-NLL loss |
| `train_prior_head_libero.py` | Train Prior Head, episode-level internal split, checkpoint/early stopping |
| `eval_goal_shuffle_libero.py` | Offline ablation về vai trò của goal |
| `eval_lib/eval_libero_prior_bc.py` | Simulator baseline chỉ dùng action mean từ Prior Head (**không** có MPPI) |
| `eval_lib/eval_libero_prism_mppi.py` | Full model: prior + PoG + MPPI + LeWM |
| `check_lewm_libero.py`, `check_lewm_action_ablation.py` | Offline LeWM sanity test và action shuffle |
| `check_libero_goal_trajectory.py` | Goal-cost trên expert demonstrations |

> **Lưu ý đường dẫn:** Một số bản script cũ nằm ở **root**, các bản đánh giá nằm trong `eval_lib/`; dùng `find . -name '<filename>'` để xác nhận. Script có default **hard-code** `/workspace/prism-ben`, nên **cần đổi path** nếu teammate có workspace riêng.

---

## 3. Xử lý dữ liệu LIBERO: raw → stable-worldmodel HDF5

### 3.1. Nguồn và split

Raw LIBERO-Object chứa file theo task, ví dụ:

```text
.../libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
└── data/
    ├── demo_0/
    │   ├── actions
    │   └── obs/
    │       ├── eye_in_hand_rgb
    │       ├── ee_states
    │       └── gripper_states
    └── demo_1/ ...
```

Converter `scan()` duyệt `.hdf5` theo task, dùng `--seed 0`, chọn khoảng **10% demos mỗi source file** vào validation và phần còn lại vào train. Split được ghi thành `split.json` dưới dạng `[path_nguồn, tên_demo]`. Đây là **episode split**, không phải frame split.

Kết quả đã quan sát:

| Phần | Episode | Frame | Clips reported (`frameskip=5`, `num_steps=4`) |
|---|---:|---:|---:|
| Train | 5.850 | 906.476 | 795.326 |
| Validation riêng từ converter | 650 | 101.142 | 88.792 |

**Không nhầm `libero_val.h5` với validation bên trong `train.py`**: ở baseline hiện tại, `train.py` lấy `libero_train.h5` rồi gọi `random_split(dataset, ...)` trên **các clips**, nên có thể xảy ra *overlap leakage* giữa clip train/val (cùng episode và các cửa sổ kế cận). File `libero_val.h5` do converter tạo **không tự được dùng** bởi lệnh `python train.py data=libero` thông thường.

### 3.2. Nội dung mỗi frame sau conversion

| Key | Shape train | Ý nghĩa |
|---|---|---|
| `pixels` | `(906476,128,128,3)` `uint8` | RGB camera `obs/eye_in_hand_rgb` mặc định |
| `action` | `(906476,7)` `float32` | **Raw** LIBERO `g['actions']`; converter **không** đổi sang `state_diff` |
| `proprio` | `(906476,7)` `float32` | `build_states(ee_states,gripper_states, gripper_mode='closure')` |
| `ep_len` | `(5850,)` | Độ dài của từng episode |
| `ep_offset` | `(5850,)` | Offset bắt đầu episode trong flat arrays |

- Converter mặc định **xoay camera 180°** (`img[:, ::-1, ::-1]`). Khi simulator đánh giá cũng cần `--camera eye_in_hand` và `--flip180` (đang là mặc định) để nhất quán. Hãy kiểm tra ảnh thực tế trước khi chắc chắn các flag đã dùng.
- Images sau conversion vẫn là 128×128. Khi **train LeWM**, `utils.get_img_preprocessor` áp dụng **ImageNet ToImage → Resize(224)**. Cache phải dùng cùng transform này.
- `proprio` có trong data và được normalize khi load theo `train.py`, nhưng model `JEPA.encode()` trong file hiện có đọc **`pixels` và `action`**; không thấy proprio đưa trực tiếp vào predictor. **Không khẳng định** LeWM hiện tại có proprio-conditioned dynamics chỉ vì dataset chứa key `proprio`.

### 3.3. Ví dụ tạo dataset (chỉ dùng nếu cần tạo lại)

```bash
cd /workspace/prism-ben
python data_loader/libero_to_swm.py \
  --src /workspace/FA26_RE/datasets/libero_object \
  --out_dir /workspace/prism-ben/prism-output/dataset_lib_NEW \
  --camera_key obs/eye_in_hand_rgb \
  --val_frac 0.1 --seed 0
```

**Dùng `dataset_lib_NEW` để không ghi đè dataset cũ.** Với camera `agentview`, tạo **dataset riêng** và cache riêng; không trộn hai camera trong cùng run. Kiểm tra script đang nằm ở `data_loader/` trước khi chạy.

### 3.4. Temporal contract: `frameskip=5`, `history_size=3`, `num_preds=1`

Một LeWM training clip dài **4 mốc quan sát**, với **5 raw actions giữa hai mốc liên tiếp**:

```text
frame z0 ── actions[0:5]  ──► frame z1
frame z1 ── actions[5:10] ──► frame z2
frame z2 ── actions[10:15]──► frame z3
                              (có thêm block data để tạo đủ clip theo loader)

Context: [z0, z1, z2] + [a0(35D), a1(35D), a2(35D)]
Target:  [z1, z2, z3]
MPPI one planning step = 1 block = 5 raw environment steps.
```

`train.py`: `ctx_emb=emb[:,:3]`, `ctx_act=act_emb[:,:3]`, `tgt_emb=emb[:,1:]`; predictor tạo chuỗi latent dự đoán và loss MSE so với target. Đây là **mô tả indexing từ source code**; khi sửa frameskip cần verify lại implementation chính xác của `HDF5Dataset` và timestamps.

---

## 4. Hướng A — Cách LeWM đã được train từ đầu

### 4.1. Model và loss

- Khởi tạo ViT encoder `pretrained=False` trong `train.py` (không fine-tune từ checkpoint PushT/Cube).
- Encoder → CLS token → projector (`z` kích thước **192** trong run đã dùng).
- Action raw 7D được normalize rồi gom 5 bước thành 35D → action encoder.
- Predictor `ARPredictor` được cấu hình `num_frames=3`, kết hợp latent history và action embeddings.
- Loss:

  ```text
  pred_loss   = mean((pred_emb - target_emb)^2)
  sigreg_loss = SIGReg(emb)
  total_loss  = pred_loss + cfg.loss.sigreg.weight * sigreg_loss
  ```

- `utils.get_column_normalizer` tính `torch.mean` và `torch.std` (sample/unbiased std) từ action gốc qua `HDF5Dataset.get_col_data('action')`, lọc NaN. **Action normalizer LeWM khác normalizer của Prior Head** (head dùng sklearn StandardScaler fit chỉ trên train episodes).
- Checkpoint model **object** xuất bằng callback `ModelObjectCallBack`, tên phụ thuộc `output_model_name` và `subdir`. Checkpoint cuối đã dùng cho baseline được đặt tại `stablewm/lewm_epoch_100_object.ckpt`.

### 4.2. Cấu hình dữ liệu đã dùng

`config/train/data/libero.yaml` trên remote:

```yaml
dataset:
  num_steps: ${eval:'${wm.num_preds} + ${wm.history_size}'}
  frameskip: 5
  path: /workspace/prism-ben/prism-output/dataset_lib/libero_train.h5
  keys_to_load:
    - pixels
    - action
    - proprio
  keys_to_cache:
    - action
    - proprio
```

`history_size=3`, `num_preds=1`, `img_size=224` là thông số của **run được báo cáo**; các tham số khác (LR, λ SIGReg, batch, scheduler, optimizer, actual seed) **phải lấy từ config snapshot của run hoặc log, không suy đoán chỉ từ README này**.

### 4.3. Cách chạy tương đương (mẫu, không cam kết trùng mọi flag với run 100 epoch)

```bash
cd /workspace/prism-ben
export STABLEWM_HOME=/workspace/prism-ben/stablewm
# Xem config trước, đừng vô tình ghi đè checkpoint cùng tên
python train.py data=libero \
  output_model_name=lewm_libero_alt \
  subdir=lewm_libero_alt \
  trainer.max_epochs=100 trainer.devices=1 \
  wandb.enabled=False
```

**Giữ `output_model_name` và `subdir` khác run gốc** nếu teammate thử hướng mới. Kiểm tra `python train.py --help` và Hydra config thực tế trước khi dành nhiều giờ GPU. Không ghi đè `/workspace/prism-ben/stablewm/lewm_epoch_100_object.ckpt`.

### 4.4. Bằng chứng huấn luyện và điều chưa xác nhận

- Nhóm đã huấn luyện **100 epochs** từ LIBERO. LeWM checkpoint object cuối ≈ **18,03 triệu parameters** (theo phiên đã kiểm tra).
- Một **sanity run sớm** với 15 train batches trong 1 epoch báo `fit/loss=2.3369`, `fit/pred_loss=0.3212`, `fit/sigreg_loss=22.375`, `validate/loss=3.0889`, `validate/pred_loss=0.1538`, `validate/sigreg_loss=32.5995`. **Đây không phải kết quả epoch 100, cũng không phải held-out episode validation**; không đặt ngang với các đánh giá offline ở phần 7.
- Chưa có bảng learning curve đầy đủ 100 epochs được xác minh trong bộ tài liệu này. Teammate muốn so sánh loss train phải lấy log/CSV và resolved Hydra config của **đúng run**.

---

## 5. Tạo latent cache để train Prior Head

Sau khi LeWM train xong, dùng encoder **đóng băng** để encode **mọi frame train** và frame cuối của **mỗi episode** thành goal embedding.

```bash
cd /workspace/prism-ben
python jepa_cache_libero.py --dry-run --batch 16 \
  --h5 prism-output/dataset_lib/libero_train.h5 \
  --ckpt stablewm/lewm_epoch_100_object.ckpt

# Chỉ chạy full nếu bạn thực sự cần cache mới; KHÔNG dùng --force lên cache gốc.
python jepa_cache_libero.py --batch 64 \
  --h5 prism-output/dataset_lib/libero_train.h5 \
  --ckpt stablewm/lewm_epoch_100_object.ckpt \
  --out stablewm/jepa_cache/libero_alt
```

Cache gốc đã có tại `stablewm/jepa_cache/libero/`:

```text
features.npy  (906476, 192), float16   # 1 embedding / frame
 goals.npy    (5850, 192), float16     # embedding frame cuối từng episode
 goal_meta.json, meta.json             # metadata về goal/cache/checkpoint
```

Đã xác minh cache với model epoch 100 trên 16 frame ngẫu nhiên: **mean MSE 0.00000550, mean cosine 0.999994**. Đây là sanity check tính nhất quán, không phải metric task success.

**Cực kỳ quan trọng:** Đổi LeWM checkpoint, camera hoặc preprocessing → **phải tạo latent cache mới**. Không dùng `features.npy` từ encoder A để train/evaluate predictor B.

---

## 6. Hướng A — Cách Gaussian Prior Head đã được train

### 6.1. Input, output, objective

`prior_head.py` nhận:

```text
Input : z_t (192D) + z_goal (192D) → concatenate 384D
Model : MLP hidden=512, GELU
Output: μ và σ cho (H=5, action_block=5, action_dim=7)
        = 25 raw actions cho mỗi lần lên plan
Loss  : β-NLL (beta=0.5), Gaussian normalized action target
```

- `z_goal` = **latent frame cuối cùng của chính demonstration**. Đây là **endframe visual goal**, không phải text instruction hoặc task ID embedding.
- Mỗi training example lấy latent ở thời điểm đầu của một action window 25 steps và lấy toàn bộ 25 raw actions phía sau làm target.
- `StandardScaler` cho target action được fit **chỉ trên train episodes** bên trong `libero_train.h5`. `σ` đầu ra có `softplus + sigma_floor`.
- Split Prior Head: episode-random `val_frac=0.1`, seed mặc định `42`, **nằm bên trong 5.850 episodes của `libero_train.h5`**. Đây **không phải** `libero_val.h5` độc lập của converter.
- Mỗi epoch đánh giá `val beta-NLL` và `val MSE` trên **normalized action units**. Best checkpoint chọn theo **val beta-NLL**, nên epoch tốt nhất theo MSE có thể không trùng.

### 6.2. Lệnh mẫu

```bash
cd /workspace/prism-ben
python train_prior_head_libero.py --dry-run --batch 256

# Ví dụ cho teammate: đổi checkpoint output để không ghi đè run gốc
python -u train_prior_head_libero.py \
  --h5 prism-output/dataset_lib/libero_train.h5 \
  --cache-dir stablewm/jepa_cache/libero \
  --epochs 50 --patience 10 --batch 512 \
  --H 5 --action-block 5 --save-every 5 \
  --out stablewm/prior_head_teammate_experiment.pt \
  --ckpt-dir stablewm/prior_head_teammate_checkpoints
```

Nếu muốn resume, script lưu `last.pt` và các mốc `epoch_XXX.pt`. **Khi resume**, các tham số ảnh hưởng training (epochs, LR, batch, seed, split…) phải khớp checkpoint. Không sử dụng đường dẫn checkpoint chung giữa các thành viên.

### 6.3. Kết quả đã có

| Đánh giá | Kết quả | Diễn giải đúng |
|---|---:|---|
| Best epoch | **4** | Theo beta-NLL trên split val của Prior Head |
| Val action MSE | **0.52204** | MSE trên action **đã normalize**; không phải success rate |
| Global goal shuffle MSE | **≈0.988** | Tráo goal giữa nhiều task làm kết quả xấu đi |
| Same-task goal shuffle MSE | **≈0.544** | Demo cùng task có goal gần tương đương; không kết luận model bỏ qua goal |
| BC-only simulator (task 0, 300 steps) | **0/1 success** | EEF–object gần nhất **0.0921 m** tại step 85; không nhấc vật |

**Phân biệt rõ:** Prior Head loss tốt không đảm bảo policy điều khiển robot gắp thành công; MSE của normalized action cũng không trực tiếp phản ánh sai số vị trí EEF.

---

## 7. Planner, evaluation và quá trình debug đã thực hiện

### 7.1. PRISM-style MPPI hiện tại

- Prior Head sinh `μ,σ` cho 5 action blocks; biến đổi **Prior scaler → raw action → LeWM scaler** trước khi MPPI sử dụng.
- PoG (`pog_fusion`) kết hợp base Gaussian với Gaussian Prior; MPPI cập nhật **mean** theo softmax trọng số và **giữ σ cố định** giữa các iterations.
- `K=32` hoặc `128`, `iters=3`, horizon `H=5` blocks = **25 raw simulator steps** theo mặc định.
- World-model rollout dự đoán latent cuối; objective hiện tại là **terminal latent goal MSE** `mean((z_pred_final-z_goal)^2)`.
- Mỗi replan chỉ thực thi **block đầu (5 raw actions)**, nhận ảnh thực tế mới, rồi lên plan lại.
- LIBERO simulator reward chỉ được dùng để kiểm tra success/done; không phải reward mà MPPI tối ưu.

```bash
cd /workspace/prism-ben
export MUJOCO_GL=egl
export PYTHONPATH="/workspace/FA26_RE/LIBERO:/workspace/prism-ben:${PYTHONPATH:-}"
# Một trial chẩn đoán ngắn; tùy phiên script có in [GRASP], [WM-DIAG], [MPPI-COST].
python eval_lib/eval_libero_prism_mppi.py \
  --suite libero_object --task-id 0 --camera eye_in_hand \
  --trials 1 --max-steps 100 --samples 32 --iters 3 \
  --injection pog --out stablewm/prism_teammate_test.json
```

Để ghi `[MPPI-COST]`, dùng phiên script **đã thêm goal diagnostics** `eval_lib/eval_libero_prism_mppi_goal_diag.py` nếu file đó thực sự có trên workspace.

### 7.2. Kết quả simulator (mang tính chẩn đoán, **không phải benchmark**)

| So sánh trên task 0 | Prior Head mean, không MPPI | PRISM-MPPI K=32 |
|---|---:|---:|
| Horizon simulator | 300 raw steps | 300 raw steps |
| Closest EEF–object | **0.0921 m** | **0.2659 m** |
| End EEF–object | **0.1313 m** | **0.5027 m** ở episode 300 steps đã log |
| Object lifted | Không | Không |
| Task success | **0/1** | **0/1** |

K=128 cũng có **trial thất bại**, nhưng số trial hạn chế và chưa đủ thống kê để kết luận xu hướng K. Tốc độ MPPI K32 từng đo khoảng **35–40 ms/replan**; không coi là phép đo hiệu năng phần cứng chuẩn hóa.

### 7.3. Các kiểm tra đã hoàn tất

| Thí nghiệm | Kết quả | Giới hạn diễn giải |
|---|---|---|
| Latent cache vs encoder epoch 100 | Mean cosine **0.999994** (16 frames) | Loại trừ phần lớn cache mismatch |
| Offline one-step LeWM, 512 mẫu `libero_train.h5` | Prediction MSE **0.276888** vs Persistence **0.281096** → tốt hơn **≈1.50%** | Các target gồm t+5,t+10,t+15, teacher-forced; **train data**, không phải held-out task |
| LeWM action-shuffle | Correct **0.276888** vs shuffled **0.290172** → shuffled tệ hơn **4.80%** | Action có ảnh hưởng nhưng chưa khẳng định dynamics đủ tốt |
| Online LeWM/goal-cost trên 100 steps | 20 blocks, **10/20 lần đúng dấu delta**; 9/20 lần dự đoán goal cost giảm nhưng thực tế tăng | Một trajectory, prediction gap có thể lớn |
| Goal-cost trên 12 other expert demos cùng task | **12/12** có end MSE < start MSE; mean **1.6469 → 0.1339** | Có global progress signal; không bảo đảm giảm từng bước |
| MPPI simulator step 90 | Goal MSE **0.2681** nhưng EEF–object **0.3848 m** và vật chưa nhấc | Cost thấp **không đồng nghĩa task success** |

**Tóm tắt chẩn đoán:** (1) predictive dynamics chưa vượt Persistence nhiều trong phép đo hiện tại; (2) LeWM có sử dụng action nhưng sai số cost sau rollout cao trên simulator; (3) terminal visual goal cost phù hợp với xu hướng expert tổng thể nhưng có thể bị tối ưu tới những trạng thái không hữu ích ngoài expert distribution; (4) chính BC Prior cũng chưa hoàn thành grasp. **Không được khẳng định một nguyên nhân duy nhất đã được chứng minh.**


## 8. Các giới hạn **bắt buộc giải quyết** để có benchmark nghiêm túc

1. **Validation leakage của LeWM:** `train.py` random-split **clip-level** từ `libero_train.h5`; các clip lân cận cùng episode có thể xuất hiện ở train/val. Mọi thử nghiệm mới nên **train trên `libero_train.h5`, validate/test trên `libero_val.h5`** hoặc sử dụng sampler tách **theo episode**, không leak. Giữ cố định tập train/val khi so sánh.
2. **Training loss vs offline eval:** `fit/pred_loss` trong Lightning và offline prediction MSE `0.276888` chưa được chứng minh dùng đúng cùng sampling/metric split. Ưu tiên viết **teacher-forced evaluation trên `libero_val.h5`** với transform, alignment, scaler và mask **khớp training**. Không suy từ các số này rằng epoch 100 đã overfit.
3. **Task success metric:** cần nhiều initial states và nhiều tasks, report `success rate ± uncertainty`; không dùng 0/1 làm kết luận tổng quát.
4. **Goal design:** goal cache dùng **endframe từ training demonstration**, phù hợp với visual-goal evaluation nhưng là thông tin privileged nếu bài toán đích không cấp goal image. So sánh các model phải dùng **cùng cơ chế cấp goal**.
5. **Không lẫn metric:** offline action MSE, world-model latent MSE, goal MSE, MPPI planned cost, and binary simulator success là **5 đại lượng khác nhau**; ghi unit và horizon.
6. **Công bằng về compute:** báo cáo GPU-hours, số gradient steps, số training frames, K/iterations/horizon, số simulator steps, seed và camera.
7. **Data provenance:** dataset HDF5 lớn và checkpoint có thể không được push Git; dùng path/manifest/checksum, tránh để teammate nhận nhầm run.

---

## 9. Hướng khác cho teammate — **không trùng baseline đã chạy**

Baseline hiện tại đã bao gồm: **LeWM scratch 1-step với eye-in-hand + endframe-goal Gaussian MLP Prior Head + PoG/σ-frozen MPPI + terminal latent MSE**. **Đừng chỉ đổi seed hoặc tăng K rồi gọi là hướng train mới.**

| Mã nhánh | Ý tưởng khác baseline | Thay đổi cụ thể | Giữ cố định để so sánh | Câu hỏi nghiên cứu |
|---|---|---|---|---|
| **B1 – Multi-step LeWM** | Huấn luyện world model có supervision nhiều bước / autoregressive rollout | Sửa `num_preds`, indexing/rollout loss và config; thử horizon 2–3 blocks **sau khi kiểm tra shape** | Cùng data split, camera, eval suite, compute budget | Dự đoán nhiều bước giúp MPPI giảm model exploitation không? |
| **B2 – Multi-view / agentview** | Thay hoặc phối hợp camera `agentview` | Convert dataset mới (camera riêng), thay encoder input/goal; tạo cache riêng | Raw demos, controller action và episode split | Observation viewpoint có quyết định khả năng nhìn thấy object/grasp không? |
| **B3 – Goal-conditioned action head khác** | Mạnh hóa action prior thay vì chỉ Gaussian MLP | Train chunked BC Transformer / diffusion-style policy hoặc phase-aware prior; vẫn dùng LeWM latent | Cùng LeWM checkpoint+cache và action targets | Policy tốt hơn có tăng grasp rate ngay cả khi bỏ MPPI? |
| **B4 – Improved world-model target** | Bổ sung object/state-aware auxiliary loss, action-delta loss hoặc task-aware sampling | Thêm supervised auxiliary targets từ proprio/object state (nếu được phép theo protocol) | Cùng ảnh, raw action, evaluation, cùng output latent dimension khi có thể | Thông tin vật lý rõ ràng có làm predictor action-sensitive hơn? |
| **B5 – Memory/subgoal training** | Retrieve intermediate expert states làm goal có điều kiện theo phase | Thiết kế retrieval và subgoal selector, huấn luyện thêm subgoal head nếu cần | Cùng LeWM/Prior baseline, cùng memory access/budget | Episodic memory có giúp goal cost bớt lệch ở giai đoạn grasp? |

**Phân công tránh chồng chéo (đề xuất):**

- **Người đang sở hữu baseline:** duy trì LeWM 100 epoch + Prior Head hiện tại, đối chiếu training/offline loss, giữ evaluation script và main results.
- **Teammate 1 (ưu tiên):** nhận **B1 (multi-step LeWM)**; **không chỉnh trực tiếp checkpoint baseline**. Đây là hướng thay đổi thực chất phần **train world model** và kiểm tra giả thuyết dynamics yếu.
- **Teammate 2 (nếu có):** nhận **B2 (camera/multi-view)**; ownership toàn bộ converter camera mới + cache + eval; không sửa dataset `dataset_lib` gốc.
- **Teammate 3 (nếu có):** nhận **B3 (Prior Head alternative)**; freeze LeWM baseline và thay action policy, có BC-only và MPPI head ablation.
- **B5:** gắn với hướng Memory-Grounded JEPA của luận văn; cần phối hợp riêng để không trùng với người đang xây dựng episodic retrieval.

**Khuyến nghị thực dụng:** teammate nên chọn **B1 hoặc B3**. B1 đánh thẳng vấn đề *dynamics quality*; B3 giúp xác minh giới hạn nằm ở *action prior/grasp*. Đừng đổi LeWM, camera, prior, cost và MPPI **đồng thời** vì sẽ không biết cải thiện đến từ đâu.

### 9.1. Mẫu thiết kế một experiment độc lập

```text
Experiment ID: B1_multistep_2pred_seed42
Owner:
Giả thuyết:
Baseline đối chứng: lewm_epoch_100_object.ckpt + prior_head_libero.pt
Code commit / branch:
Data manifest/split hash:
Train config (resolved):
Một biến chính đã thay đổi:
Train time / GPU hours:
Best val latent MSE (held-out episodes):
Persistence latent MSE:
Action-shuffle delta:
BC-only success / grasp / min EEF-object:
MPPI success K=32/K=128 (paired init states):
Predicted-vs-actual goal cost MAE:
Kết luận có bằng chứng:
Hạn chế / cần kiểm tra tiếp:
```

### 9.2. Quy ước Git/branch và tránh ghi đè

```bash
cd /workspace/prism-ben
git status
git fetch origin
# Teammate làm việc trên nhánh riêng, từ commit đã thống nhất của nhóm:
git switch -c exp/B1-multistep-lewm
```

Quy ước thư mục output:

```text
stablewm/experiments/B1_multistep_seed42/
stablewm/experiments/B2_agentview_seed42/
stablewm/experiments/B3_prior_transformer_seed42/
prism-output/dataset_lib/              # GỐC, chỉ đọc
prism-output/dataset_lib_agentview/    # Dữ liệu thử nghiệm mới
```

**Không** push HDF5 lớn, cache/ckpt nhạy tài nguyên lên Git nếu chưa có Git LFS và đồng thuận nhóm. Đảm bảo `git status` không vô tình track raw data/checkpoints. Dùng `git pull --ff-only` khi working tree sạch; khi sửa local hãy commit/stash hoặc sao lưu trước. Không dùng `git reset --hard`/`git clean -fd` nếu chưa xác minh nội dung sẽ mất.

---

## 10. Checklist trước khi teammate bắt đầu GPU training

- [ ] Chạy `git status`, lưu Git commit và snapshot resolved Hydra config.
- [ ] Xác nhận camera trong converter, sample RGB và hướng xoay 180°.
- [ ] Verify `ep_len/ep_offset`, split episode train/val, không cross-boundary trong sampler.
- [ ] Xác nhận raw `action` 7D; frameskip=5 ⇒ encoder input 35D.
- [ ] Xác nhận `history_size`, `num_preds`, output latent shape và alignment target bằng **1 batch** trước khi train.
- [ ] Tính normalization chỉ trên tập training; ghi mean/std để tái lập.
- [ ] Dùng checkpoint/cache/output folder **riêng**; không dùng `--force` trên cache gốc.
- [ ] Đánh giá LeWM trên **held-out episodes** bằng đúng training preprocessing.
- [ ] Chạy BC-only và MPPI với **paired initial states**, same task/goal/camera/step budget.
- [ ] Báo cáo cả thất bại, sai số và giới hạn; không báo `val MSE` như `task success`.

## 11. Các lệnh chẩn đoán có thể tái sử dụng

```bash
cd /workspace/prism-ben

# Xem HDF5/cached artifacts tồn tại
ls -lh prism-output/dataset_lib/{libero_train.h5,libero_val.h5,split.json}
ls -lh stablewm/lewm_epoch_100_object.ckpt stablewm/prior_head_libero.pt

# Offline LeWM/Persistence (trên train HDF5 mặc định)
python check_lewm_libero.py --n 512 --batch-size 64

# LeWM action sensitivity
python check_lewm_action_ablation.py --n 512 --batch-size 64

# Goal-cost trên đúng task, ref demo_0 là ep 4950
python check_libero_goal_trajectory.py --goal-episode-id 4950 --n-cross 12 --stride 5

# Goal-shuffle prior; script nên chạy từ root theo phiên hiện có
python eval_goal_shuffle_libero.py \
  --split-json prism-output/dataset_lib/split.json --batch 512

# Simulator BC-only
python eval_lib/eval_libero_prior_bc.py \
  --suite libero_object --task-id 0 --camera eye_in_hand \
  --trials 1 --max-steps 300 --seed 42 --modes bc_only
```

## 12. Những điều hiện **chưa có bằng chứng**

1. Chưa có **success rate thống kê** trên toàn bộ LIBERO-Object hoặc nhiều task.
2. Chưa chứng minh goal MSE thấp là do **camera viewpoint** (đó vẫn là giả thuyết).
3. Chưa chứng minh LeWM **không thể cải thiện** khi train lại; cần đo lại với đúng held-out protocol.
4. Chưa có so sánh **PRISM baseline vs Memory-Grounded JEPA** theo cùng dataset, compute và initial states.
5. Chưa xác minh end-to-end LIBERO action conventions cho mọi controller/task; cần kiểm tra trước khi đổi policy architecture.

---

## 13. Nguồn code cần đọc trước khi thiết kế hướng mới

- `data_loader/libero_to_swm.py`: `scan`, `write_h5`, `verify`, camera/flip/proprio/action.
- `config/train/data/libero.yaml`: HDF5 loader, frame skip.
- `train.py`: `lejepa_forward`, `random_split`, normalization, SIGReg/prediction loss.
- `jepa.py`: `JEPA.encode`, `predict`, `rollout`.
- `utils.py`: `get_img_preprocessor`, `get_column_normalizer`.
- `jepa_cache_libero.py`: feature/goal generation.
- `prior_head.py`, `train_prior_head_libero.py`: Gaussian prior, β-NLL, episode-level head split, resume/early-stop.
- `eval_lib/eval_libero_prior_bc.py`: action mean BC-only.
- `eval_lib/eval_libero_prism_mppi.py`: PoG + frozen-σ MPPI + world-model terminal goal cost.

**Nếu teammate chỉ đọc một câu:** Baseline của nhóm đã train **LeWM scratch (pixel → latent dynamics)** rồi **Gaussian Prior Head (latent + endframe goal → 25 actions)** và dùng **PoG + MPPI** để plan; trên một trial LIBERO-Object cả BC và MPPI chưa thành công, vì thế hướng mới nên thay **một biến khoa học rõ ràng** và đánh giá lại trên **episode held-out + paired simulator initial states**, không chỉ tăng epochs/K.
