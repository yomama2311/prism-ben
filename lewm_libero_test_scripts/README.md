# Test pretrained LeWM v2 checkpoint on LIBERO (offline latent prediction)

Model: https://huggingface.co/QinJialong/lewm_libero_v2_200/tree/main

## On Vast.ai in the `(prism)` conda environment

```bash
cd /workspace/prism-ben
conda activate prism
python -m pip install -U huggingface_hub pyyaml omegaconf h5py numpy opencv-python
mkdir -p tools/lewm_libero_eval
# Copy the 3 scripts from this ZIP to tools/lewm_libero_eval
bash tools/lewm_libero_eval/download_model.sh
python tools/lewm_libero_eval/inspect_checkpoint.py \
  --checkpoint weights/lewm_libero_v2_200/last.ckpt \
  | tee weights/lewm_libero_v2_200/checkpoint_keys.txt
```

Find raw LIBERO `.hdf5` data (7D actions):

```bash
find /workspace/FA26_RE/datasets -type f \( -name '*.hdf5' -o -name '*.h5' \) | head -20
```

Then test offline (examples, the `--libero-root` argument is an example: **change it**):

```bash
python tools/lewm_libero_eval/eval_lewm_libero_offline.py \
  --checkpoint weights/lewm_libero_v2_200/last.ckpt \
  --config weights/lewm_libero_v2_200/config.yaml \
  --libero-root /workspace/FA26_RE/datasets/libero_spatial \
  --code-root /workspace/prism-ben \
  --max-demos 10 \
  --horizons 1 5 10 25 50 100 200 \
  --output results/lewm_last_libero_spatial.json
```

**IMPORTANT:** The Hugging Face repository publishes the YAML and checkpoints but not the matching training/inference source implementation. `prism-ben/main` is the closest inspected example, *not a verified equivalent*. The evaluator uses **strict model loading**: if it rejects unmatched tensors you **must obtain the exact source architecture**. Do **not** switch to `strict=False` because the numbers could become invalid. You may add a custom `--builder my_libero_model:build_model` function in the original training source.

The demo loader expects raw LIBERO HDF5: `/data/demo_N/actions` (shape Nx7) and `/data/demo_N/obs/agentview_rgb` (NxHxWx3). The previously converted `libero_train.h5` with action dimension 35 and frameskip=5 is **NOT compatible** with the target config (7D / frameskip=1). You will also need to confirm camera orientation / normalization and provide the original training action mean/std as `--action-stats path.npz` (two arrays `mean` and `std`, each shape `[7]`) for reproducible results. Default `--action-norm estimate` calculates approximate stats from the evaluated demos, **not** an exact replica of training-time stats; treat the numbers as diagnostic only. `--action-norm raw` is available if training used unnormalized actions.

The script reports per-horizon teacher-forced MSE, free-running MSE and a persistence baseline; no LIBERO **task success rate**. To measure success rate you additionally need a trained policy / action prior + action selection and a closed-loop task runner; the checkpoint alone cannot infer robot actions. Also ensure validation tasks/episodes are **held out** from training.

## Files
- `config.yaml`: copy of the Hugging Face training config (download script obtains the upstream file too).
- `download_model.sh`: downloads `last.ckpt` and `config.yaml` from Hugging Face. Optional `DOWNLOAD_EPOCH58=1` also downloads epoch58.
- `inspect_checkpoint.py`: verifies checkpoint metadata and prints tensor names and shapes without instantiating any model.
- `eval_lewm_libero_offline.py`: offline latent prediction benchmark on original LIBERO demonstrations, strict reconstruction check.
