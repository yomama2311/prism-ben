# PRISM-JEPA + LIBERO — bản sửa pipeline Chunked/Flat Prior Head

**Phiên bản sửa ngày 17-10-2026.** Giữ ý tưởng gốc **LeWM world model + Prior Head Gaussian (Flat hoặc Chunked) + learned latent anchors + PoG/MPPI (sigma cố định)**. Sửa các lỗi temporal context, action consistency, định nghĩa success, kiểm soát validation, checkpoint và thu thập log khi test.

> **Trạng thái xác minh:** Đã qua kiểm tra cú pháp, synthetic end-to-end converter → cache → cả hai Prior Head → evaluator giả lập; callback lưu checkpoint LeWM được kiểm thử bằng Lightning CPU. **Chưa thực sự train hoặc test trên LIBERO/MuJoCo + RTX 5090 trong môi trường của bạn**. Bạn cần chạy smoke test và benchmark LIBERO thật trước khi dùng số liệu nghiên cứu.

## 1. Cài vào repo

Giải nén thư mục `prism_libero/` vào **gốc repo** `/workspace/prism-ben` (cùng cấp `train.py`, `jepa.py`, `prior_head.py`, `module.py`). Nên backup bản cũ:

```bash
cd /workspace/prism-ben
cp -a prism_libero "prism_libero_backup_$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
# Sau đó giải nén ZIP sao cho xuất hiện /workspace/prism-ben/prism_libero/
conda activate prism     # hoặc kích hoạt đúng môi trường đã dùng để train
export STABLEWM_HOME=/workspace/stable-wm
export MUJOCO_GL=egl
export PYTHONPATH="/workspace/FA26_RE/LIBERO:/workspace/prism-ben:${PYTHONPATH:-}"
```

Yêu cầu có các thư viện từ repo gốc (`stable-worldmodel[train]`, stable_pretraining, Hydra, Lightning), PyTorch, h5py, numpy, LIBERO + robosuite + MuJoCo, `imageio` và `imageio-ffmpeg` (ghi video). Không nên thay đổi package versions của môi trường đang chạy chỉ để chạy script.

## 2. Lệnh chạy MỚI: train → lưu ckpt → test và ghi log

```bash
cd /workspace/prism-ben
export STABLEWM_HOME=/workspace/stable-wm
export MUJOCO_GL=egl
export PYTHONPATH="/workspace/FA26_RE/LIBERO:/workspace/prism-ben:${PYTHONPATH:-}"

# Quản lý lưu ckpt và đánh giá sau khi train.
EPOCHS_JEPA=50 EPOCHS_HEAD=50 CHECKPOINT_EVERY=10 \
LEWM_PATIENCE=15 HEAD_PATIENCE=8 \
EVAL_TASKS=0 EVAL_EPISODES=3 EVAL_SAMPLES=32 EVAL_ITERATIONS=3 EVAL_VIDEO=1 \
bash prism_libero/run_libero_pipeline.sh libero_spatial /workspace/FA26_RE/datasets
```

Đổi `libero_spatial` thành `libero_object` hoặc suite khác có dữ liệu tương ứng. `EVAL_TASKS=0` giúp smoke test một task; khi chạy được hãy đổi `EVAL_TASKS=all`. Theo converter mặc định, **camera `agentview_rgb`**; eval tự lấy đúng `agentview_image` và xoay ảnh tương ứng. Nếu train lại với eye-in-hand, dùng `TRAIN_CAMERA=eye_in_hand_rgb` **từ bước converter**, không trộn hai camera trong cùng thử nghiệm.

Pipeline gồm 5 giai đoạn:

1. `convert_libero_to_swm.py` đọc `<task>_demo.hdf5`, tách riêng `libero_<suite>_train.h5` và `..._val.h5` theo **nguyên episode** (10% val), tạo sidecar `.meta.json`. `pixels` uint8 128×128, `action` 7D (action cuối episode NaN theo SWM), `proprio` 8D.
2. `train_lewm_managed.py` train **cùng predictor JEPA/LeWM và loss với repo gốc**, nhưng dùng val HDF5 từ các episodes độc lập. Theo dõi `validate/pred_loss` hoặc alias `val/mse`. Lưu best, last và checkpoint mỗi 10 epochs.
3. `cache_features_libero.py` encode toàn bộ **train frames bằng chính best checkpoint** từ giai đoạn 2, tạo `features.npy` fp16, `goals.npy`, `task_goals.npy` (medoid mỗi task), `action_stats.json` và `meta.json`.
4. `train_chunk_prior_head.py` train **Chunked** và **Flat** head. Mỗi head lưu best, last và epoch_010, epoch_020,...; early stopping khi validation NLL ngừng cải thiện. Ưu tiên **full-window** 100 raw steps, không học sai các anchor được padding cuối episode. Action scaler của head fit trên **head training episodes**, không dùng head validation episodes.
5. `eval_libero_prism.py` chạy 5 biến thể trên cùng task/seed/init states: `chunked_bc`, `chunked_pog_final`, `chunked_pog_subgoal`, `flat_bc`, `flat_pog_final`. Tạo summary JSON/CSV, JSONL từng action/block và MP4 nếu `EVAL_VIDEO=1`, rồi tổng hợp `compare.csv`.

**Lưu ý:** `--patience` có thể khiến training dừng trước epoch 10/20/...; chỉ checkpoint những mốc mà quá trình train thực sự đi qua. Muốn chạy đủ 50 epochs để lưu 10,20,30,40,50, dùng `HEAD_PATIENCE=0 LEWM_PATIENCE=0`.

## 3. Nơi lưu file

Giả sử `STABLEWM_HOME=/workspace/stable-wm`, suite `libero_spatial`:

```text
/workspace/stable-wm/
├── libero/
│   ├── libero_spatial_train.h5
│   ├── libero_spatial_train.h5.meta.json
│   ├── libero_spatial_val.h5
│   └── libero_spatial_val.h5.meta.json
├── libero_libero_spatial/
│   ├── best_object.ckpt                   # torch.load => JEPA, dùng tạo cache + eval
│   ├── last_object.ckpt
│   ├── epoch_010_object.ckpt, epoch_020_object.ckpt,...
│   ├── config.yaml, summary.json, history.jsonl, history.csv
│   └── checkpoints/
│       ├── best.ckpt                      # FULL Lightning trainer+optimizer
│       ├── last.ckpt                      # resume
│       └── epoch_010.ckpt,...
├── jepa_cache/libero_spatial/
│   ├── features.npy, goals.npy, task_goals.npy
│   ├── action_stats.json, meta.json
├── prior_checkpoints/libero_spatial/
│   ├── chunked_best.pt, flat_best.pt      # heads được dùng khi eval
│   ├── chunked/
│   │   ├── best.pt, last.pt, epoch_010.pt,...
│   │   ├── history.jsonl, history.csv, summary.json, split.json
│   └── flat/
│       ├── best.pt, last.pt, epoch_010.pt,...
├── logs/libero_spatial/
│   ├── convert.log, lewm.log, cache.log
│   ├── prior_chunked.log, prior_flat.log
│   └── eval_*.log
└── evaluations/libero_spatial/<UTC timestamp>/
    ├── chunked_bc.json, chunked_bc.csv
    ├── chunked_bc_diagnostics/
    │   ├── task00_trial000.jsonl          # mọi action + mỗi 5-step block
    │   └── task00_trial000.mp4            # chỉ khi --video
    ├── chunked_pog_final.json / ...
    ├── chunked_pog_subgoal.json / ...
    ├── flat_bc.json / ...
    ├── flat_pog_final.json / ...
    ├── compare.csv
    └── compare.txt
```

Checkpoint **best LeWM và best head** là *hai mô hình khác nhau*. Không load `best.pt` (Prior Head) vào `torch.load` của LeWM. `last.pt` của head có optimizer + global step để resume; `best.pt` để eval, không để resume optimizer. `*_object.ckpt` LeWM là JEPA object; `.ckpt` trong `checkpoints/` là trạng thái Lightning.

## 4. Chỉ test lại model đã train (KHÔNG train lại)

Nếu checkpoint/cache đã tạo đầy đủ ở đúng vị trí:

```bash
cd /workspace/prism-ben
export STABLEWM_HOME=/workspace/stable-wm
SKIP="1 2 3 4" EVAL_TASKS=0 EVAL_EPISODES=3 EVAL_VIDEO=1 \
bash prism_libero/run_libero_pipeline.sh libero_spatial /workspace/FA26_RE/datasets
```

Để đổi `--mode`, `--goal-mode`, sample K hoặc chỉ chọn một checkpoint:

```bash
python -u prism_libero/eval_libero_prism.py \
  --suite libero_spatial \
  --h5 /workspace/stable-wm/libero/libero_spatial_train.h5 \
  --cache-name libero_spatial \
  --jepa-ckpt /workspace/stable-wm/libero_libero_spatial/best_object.ckpt \
  --head /workspace/stable-wm/prior_checkpoints/libero_spatial/chunked_best.pt \
  --tasks 0 --episodes 3 --mode pog --goal-mode subgoal \
  --num-samples 32 --n-steps 3 --receding 1 \
  --gripper-mode continuous --video \
  --out /workspace/stable-wm/evaluations/manual_chunked_subgoal.json \
  2>&1 | tee /workspace/stable-wm/evaluations/manual_chunked_subgoal.log
```

`--gripper-mode binary` là **ablation**: nhị phân hóa ở cả bước MPPI chấm điểm lẫn thực thi; mặc định `continuous` để tránh làm lệch phân phối action huấn luyện. `--receding 1` thực hiện 5 simulator actions rồi quan sát mới. Gripper sign switching có thể được phân tích từ JSONL.

## 5. Cách đọc log test để tìm nguyên nhân thất bại

Mỗi file `task00_trial000.jsonl` có các bản ghi:

- `type=meta`: tên task, initial state, keys quan sát khả dụng.
- `type=plan`: thời gian planning, predicted horizon cost, `current_goal_mse`, `current_target_mse`.
- `type=action`: **từng raw action 7 chiều đã thực thi**, `gripper_action`, robot EEF, vật thể (nếu xác định được `object_key`), `done`, `success`.
- `type=block`: so sánh sau **5 simulator steps** `prediction_mse`, `persistence_mse`, `predicted_goal_mse`, `actual_goal_mse`, `prediction_gap`, tương tự với **subgoal** (`predicted_target_mse`, `actual_target_mse`), object_z, EEF-object distance. Nếu episode kết thúc giữa block, `complete_5_step_block=false`, và các metric 5-step prediction là `null` (không dùng để tổng hợp).
- `type=summary`: task success dựa trên `env.check_success()`; `done` chỉ là kết thúc episode.

`eef_object_distance` là `null` nếu không phát hiện rõ một object key duy nhất; truyền `--object-key alphabet_soup_1_pos` nếu muốn. Không đoán sai đối tượng từ task text trong những scene có nhiều vật thể.

Phân tích một trajectory:

```bash
python prism_libero/analyze_trace.py \
  --trace /workspace/stable-wm/evaluations/libero_spatial/RUN/chunked_pog_subgoal_diagnostics/task00_trial000.jsonl \
  --plot
```

Tạo `.actions.csv`, `.blocks.csv`, `.diagnostics.json` và `.diagnostics.png`. Các tín hiệu đặc biệt quan trọng là: khoảng cách EEF-object tối thiểu; vật thể có tăng độ cao hay không; số lần planner dự đoán goal cost giảm nhưng cost thực tế tăng; gripper action đảo dấu; mean prediction MSE so với Persistence; và actual task success.

## 6. Chạy smoke test trước khi thuê GPU dài hạn

```bash
cd /workspace/prism-ben
python prism_libero/tests/test_pipeline.py
bash -n prism_libero/run_libero_pipeline.sh
python -m compileall -q prism_libero
```

Synthetic test xác nhận chuyển đổi/tách HDF5, cache, học Flat/Chunked, tạo `epoch_010.pt`, test causal planner, đúng shape, action clamp, JSONL closed-loop và check success trên **fake LIBERO**. Fake LIBERO chỉ dành cho test code, **không dùng để báo cáo hiệu năng robot thật**.

## 7. Resume khi rớt SSH / tmux

Prior Head (sử dụng cùng lệnh train như trước, thêm `--resume`):

```bash
python prism_libero/train_chunk_prior_head.py \
  --h5 /workspace/stable-wm/libero/libero_spatial_train.h5 \
  --name libero_spatial --arch chunked --epochs 50 --max-chunks 4 --chunk-steps 5 \
  --checkpoint-dir /workspace/stable-wm/prior_checkpoints/libero_spatial/chunked \
  --out /workspace/stable-wm/prior_checkpoints/libero_spatial/chunked_best.pt \
  --resume /workspace/stable-wm/prior_checkpoints/libero_spatial/chunked/last.pt
```

LeWM Lightning: dùng `+resume_ckpt=/workspace/stable-wm/libero_libero_spatial/checkpoints/last.ckpt` khi gọi `train_lewm_managed.py` (các override Hydra khác giữ nguyên). **Không resume từ `best_object.ckpt`** vì file đó chỉ chứa mô hình, không chứa optimizer.

## 8. Khác biệt so với phiên bản ban đầu và hạn chế

| Vấn đề ở bản cũ | Bản sửa |
|---|---|
| LeWM rollout dùng 1 latent lúc bắt đầu | Lịch sử 3 latent + 2 action blocks thực được cập nhật sau mỗi 5 actions |
| MPPI chấm điểm action chưa clamp, simulator clamp/gripper sign | Cả dự đoán và thực thi dùng cùng raw action đã qua clamp/gripper policy |
| `done` được cộng là success | `env.check_success()` quyết định success; giữ `done` riêng |
| `GATE FAIL` làm mất head/không có checkpoint để test | Lưu best/last/periodic **bất kể gate**, có `quality_gate.pass` + cảnh báo |
| Validation LeWM random clips từ cùng episode | Pipeline mới tạo **episode-disjoint HDF5** và tái sử dụng transform fit từ train |
| Chỉ có log loss cuối cùng | Log train epoch + best/last/10-epoch, JSONL từng simulator action + MPPI cost, CSV/MP4 |
| Anchor target cuối window bị clamp làm số liệu dễ hơn | Mặc định `--window-policy full`, không dùng padded anchors |
| Stats Prior Head dùng cả head val episodes | Stats Prior Head fit trên các head training episodes |

**Không thể kết luận performance được cải thiện chỉ vì code đã sửa.** Cần train lại, đo val loss + LeWM offline/Persistence (trên tập val episode độc lập), rồi thử cùng initial states, cùng camera và cùng số MPPI samples. Dataset mới có 500 episodes khác xa thử nghiệm trước với 5.850 episodes; không so sánh các loss train như thể cùng benchmark.

**Các hạn chế còn lại:** Không có chứng minh latent anchors là các subgoals có ý nghĩa thao tác; final-goal latent có thể bị ảnh hưởng viewpoint; Gaussian head có thể underfit experts đa phương thức; LeWM vẫn có thể yếu khi dự đoán multi-step; PoG/MPPI vẫn không đảm bảo grasp. Early stopping có thể kết thúc trước các mốc 10 epochs kế tiếp. Mô hình/val checkpoint có thể chịu khác biệt giữa phiên bản thư viện trên Vast.ai và máy thử ở đây.

**Quy tắc cho team:** Mỗi thử nghiệm dùng thư mục `STABLEWM_HOME` hoặc `LIBERO_LEWM_RUN_DIR` riêng, không ghi đè checkpoint cũ; ghi lại seed, dataset split, camera, image size, frameskip, H, C, K và action mode. So sánh `bc_only`, `flat_pog_final`, `chunked_pog_final`, `chunked_pog_subgoal` trên **cùng initial states** và nêu số episode mẫu.
