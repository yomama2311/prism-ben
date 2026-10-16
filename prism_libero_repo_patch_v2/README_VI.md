# Các file chép vào repo PRISM-JEPA

Gói này dành cho TRAIN TỪ ĐẦU world model trên LIBERO gốc, không nạp checkpoint.
Dựa trên upstream commit baa0eb95efb812196b68796c258b1f0cf10b7625.
Repo: https://github.com/YuhaiW/prism-jepa

## File cần chép

| File trong gói | Vị trí trong repo | Thao tác |
|---|---|---|
| train.py | train.py | Backup file hiện tại rồi thay thế |
| libero_dataset.py | libero_dataset.py | Thêm mới |
| training_callbacks.py | training_callbacks.py | Thêm mới |
| config/train/data/libero.yaml | config/train/data/libero.yaml | Thêm/thay config LIBERO |
| config/train/lewm_libero.yaml | config/train/lewm_libero.yaml | Thêm mới |

Giữ nguyên jepa.py, module.py, utils.py, config/train/lewm.yaml của upstream.
Nếu repo của bạn đã sửa các file này, việc giữ nguyên chúng KHÔNG bảo đảm chúng
còn giống upstream. Gói có train_upstream.diff để review thay đổi train.py so
với commit nêu trên; không phải patch cho mọi phiên bản fork.
Không chép code từ ZIP prism_libero_training.zip cũ vào repo.

## CHỈ HAI PATH CẦN CHỈNH

1. config/train/data/libero.yaml -> dataset.root
   Mặc định: /workspace/FA26_RE/datasets/libero_object
   Đây là thư mục chứa *.hdf5 gốc với data/demo_*/actions và obs/agentview_rgb.
   Quét đệ quy. Không dùng file libero_train.h5 đã convert sang stable-worldmodel.
   Có thể chọn suite khác; nếu trỏ thư mục cha nhiều suite thì tất cả file sẽ
   được dùng. Nên chọn rõ một suite cho thí nghiệm đầu tiên.

2. config/train/lewm_libero.yaml -> output_dir
   Mặc định: /workspace/prism-ben/stablewm/lewm_libero_object_scratch
   Dùng folder mới cho mỗi lần train. Code từ chối ghi đè run có config.yaml.
   Không cần sửa STABLEWM_HOME cho đầu ra của nhánh LIBERO này.

Không có path hardcode cần chỉnh trong hai file Python.

## Cài thêm dependency trong env prism hiện tại

Repo cần các dependency upstream đã cài. Loader dùng thêm h5py/numpy:

```bash
conda activate prism
cd /workspace/prism-ben
python -m pip install h5py numpy
```

Nếu dữ liệu dùng compression plugin thì cài hdf5plugin; loader tự import khi có.
Không cài lại torch nếu môi trường hiện tại đã nhận CUDA và train upstream được.

## Backup và chép

Giải nén ZIP ở một thư mục tạm. Trước khi thay train.py và config LIBERO hiện tại,
backup chúng sang tên khác. Chép 5 file theo đúng cấu trúc bảng trên.

## Kiểm tra config và chạy thử (RTX 5090)

```bash
python train.py --config-name=lewm_libero --cfg job --resolve

python train.py --config-name=lewm_libero \
  output_dir=/workspace/prism-ben/stablewm/libero_smoke_01 \
  loader.batch_size=8 num_workers=0 \
  trainer.max_epochs=1 \
  +trainer.limit_train_batches=2 +trainer.limit_val_batches=2
```

Đổi hậu tố output_dir nếu đã chạy smoke trước. Smoke vẫn quét metadata và đọc
action để tính stats trên train, không chỉ đọc 2 batch ngay từ đầu.

Train đầy đủ (default batch=128, 4 workers, BF16, 100 epochs):

```bash
python train.py --config-name=lewm_libero
```

Đây là điểm khởi đầu chưa đo hiệu năng, không đảm bảo batch 128 vừa GPU khi còn
process khác dùng VRAM. Nếu OOM giảm loader.batch_size=64 và dùng output_dir mới.
Không dự đoán thời gian train chỉ từ tên GPU; đo epoch thực tế.

## Những gì giữ nguyên và thay đổi

Giữ nguyên phần model/optimizer/module construction và các phép tính loss;
lejepa_forward chỉ thay phần logging để tổng hợp theo epoch:
- encoder do spt.backbone.utils.vit_hf(... pretrained=False) tạo;
- JEPA, ARPredictor, Embedder, MLP, SIGReg, trọng số loss, optimizer/scheduler upstream.

Thay đổi data pipeline: demo split trong từng task (90/10), action stats chỉ
trên train, đọc trực tiếp HDF5. Input ảnh vẫn RGB resize + ImageNet normalize,
nhưng thực hiện bằng torch interpolate bilinear antialias trong adapter, không
qua spt get_img_preprocessor. Đây là khác biệt preprocessing cần báo cáo; không
khẳng định byte-for-byte bằng resize upstream.

history=3, frameskip=5:
- 4 ảnh tại t, t+5, t+10, t+15.
- 3 action chunks [t:t+5], [t+5:t+10], [t+10:t+15], mỗi chunk 35D.
- Action cuối pad zero để khớp encode, không được dùng bởi predictor/loss.
- Mỗi clip nằm trong một demo, không nối episode.
- wm.num_preds phải là 1; adapter kiểm tra rõ điều này.

Không thêm proprio, reward, simulator state, language hoặc fusion camera.
Có thể đổi camera sang eye_in_hand_rgb nhưng model vẫn chỉ dùng một camera.
Không tự flip ảnh; nếu bật flip_vertical thì evaluation/goal phải dùng cùng quy ước.
Train/val là demo khác nhau của cùng task, không phải đánh giá task chưa thấy.

## Logging, checkpoints và early stopping (v2)

Mỗi epoch in [EPOCH] và lưu epoch_metrics.csv / epoch_metrics.jsonl, một dòng
mỗi epoch: epoch (đếm từ 1), seconds, train/loss, train/mse, train/sigreg_loss,
val/loss, val/mse, val/sigreg_loss, lr.
MSE = pred_loss = sai số latent prediction, không phải MSE ảnh hoặc success rate.
Loss = MSE + weight * SIGReg. Loss/MSE được tổng hợp theo batch_size;
SIGReg bản thân phụ thuộc batch, nên so sánh giữ cùng batch size/validation setup.
CSVLogger của Lightning còn lưu log gốc trong logs/version_*/metrics.csv.

Output:
- config.yaml, split.json, action_stats.npz: như phiên bản trước.
- checkpoints/best/best.ckpt: full Lightning checkpoint có val/mse thấp nhất.
- checkpoints/best/last.ckpt: full checkpoint lần validation gần nhất (mỗi epoch).
- checkpoints/periodic/epoch-009.ckpt: epoch thứ 10; 019 là epoch thứ 20.
  Tên checkpoint dùng epoch zero-based của Lightning, bảng CSV đếm từ 1.
- lewm_libero_object_epoch_10_object.ckpt: model object mỗi 10 epoch và epoch
  max_epochs, theo callback upstream.
- last_object.ckpt: model object cuối cùng khi fit kết thúc bình thường/early stop.
- best_checkpoint.json: đường dẫn và điểm checkpoint tốt nhất khi kết thúc fit.

best.ckpt là full checkpoint, không phải JEPA object. Không đưa trực tiếp nó
vào evaluator đang yêu cầu torch.load(...) trả về JEPA. last_object.ckpt là model
cuối cùng, KHÔNG mặc định là best. File model object không kèm normalization:
evaluator phải dùng action_stats.npz và preprocessing giống lúc train.

Default: validation mỗi epoch; early stop sau 15 lần validation liên tiếp không
cải thiện val/mse ít nhất 0.0001. Best checkpoint xét mọi mức giảm; min_delta chỉ
áp dụng early stop. Nonfinite monitor cũng gây dừng. Không đánh giá early stop
trên sanity check. Epoch checkpoints có từ epoch 10; nếu stop trước đó vẫn có
best/last. Checkpoint cuối không đồng nghĩa checkpoint có success rate tốt nhất.

Đổi thông số trên command line:

```bash
python train.py --config-name=lewm_libero \
  checkpoint.every_n_epochs=10 \
  early_stopping.patience=20 early_stopping.min_delta=0.0001
```

Tắt early stop: early_stopping.enabled=false.
Đổi tiêu chí best + early stop cùng lúc: checkpoint.monitor=val/loss.
MSE thấp có thể đi kèm latent collapse; theo dõi thêm SIGReg và kiểm tra closed-loop
trước khi kết luận chất lượng robot. Các callback không thay đổi mục tiêu tối ưu.

Entry point vẫn train mới: manager ckpt_path=None; chưa thêm resume. Full
checkpoints lưu trạng thái optimizer/scheduler để có thể bổ sung resume sau.

## Kiểm chứng

Đã chạy compileall, đối chiếu AST các phép tính loss và đối chiếu nguyên văn toàn bộ khối
tạo model/optimizer với upstream: không thay đổi. Đã kiểm tra độc lập biên clip
và độ dài action. Chưa chạy Hydra/PyTorch forward/backward trong môi trường
soạn gói vì chưa có dependency này và HDF5 LIBERO thật. Lệnh smoke ở trên là
bước bắt buộc để kiểm tra tương thích stable_pretraining/stable_worldmodel đang
cài trên Vast của bạn trước khi train dài.

Train world model xong vẫn cần train prior LIBERO và đánh giá PRISM-MPPI riêng.
