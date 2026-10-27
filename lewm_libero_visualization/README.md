# LeWM LIBERO Visualization — Frozen World Model + New Decoder

**Purpose:** create a side-by-side 3-row MP4: LIBERO ground-truth RGB, decoder reconstruction of ground-truth encoder latent, and decoder output of LeWM free-running predicted latent.

This is an **offline demonstration-conditioned world-model prediction**, not interactive LIBERO simulation, robot policy rollout, or task success evaluation.

## Contents

- `train_lewm_decoder.py` — train a new RGB decoder from frozen encoder [CLS] latents. Trains **decoder only**, never updates LeWM.
- `render_lewm_video.py` — strict-load HF checkpoint, read a demo, compute free-running latent rollout, decode to MP4 + PNG + JSON.
- `lewm_vis_common.py` — shared model loading, dataset access, preprocessing and decoder definition.
- `eval_lewm_libero_offline.py` and `inspect_checkpoint.py` — compatible evaluator and checkpoint inspection utility reused for STRICT model loading (copied from the previous evaluation scripts; no changes to the original weights).

## Set up in Vast.ai

Assumes:

- Repository root: `/workspace/prism-ben`
- Conda/venv: `prism`
- LeWM checkpoint: `weights/lewm_libero_v2_200/last.ckpt`
- Config: `weights/lewm_libero_v2_200/config.yaml`
- Original LIBERO demos: `/workspace/datasets/libero_spatial/*.hdf5`

Upload `lewm_libero_visualization.zip` to `/workspace/prism-ben/`.

```bash
conda activate prism
cd /workspace/prism-ben
mkdir -p lewm_libero_test_scripts
unzip -o lewm_libero_visualization.zip -d lewm_libero_test_scripts
python -m pip install 'pillow>=9' 'opencv-python>=4.5' h5py numpy omegaconf
```

**Avoid `pip install -U` on existing transformers/huggingface_hub/omegaconf**: earlier environment had dependency conflicts. You probably have these libraries already. Use `python -m pip check` to verify compatibility.

## 1. Smoke test: train decoder on small sample

```bash
python lewm_libero_test_scripts/train_lewm_decoder.py \
  --checkpoint weights/lewm_libero_v2_200/last.ckpt \
  --config weights/lewm_libero_v2_200/config.yaml \
  --libero-root /workspace/datasets/libero_spatial \
  --code-root /workspace/prism-ben \
  --max-demos 20 --per-task 2 --frames-per-demo 20 \
  --epochs 2 --batch-size 16 \
  --output-dir results/lewm_decoder_smoke
```

Read `results/lewm_decoder_smoke/encoder_reconstruction_preview.png` to make sure decoder inference works. This is NOT sufficient training for meaningful image quality.

## 2. Train decoder for visualization

```bash
python lewm_libero_test_scripts/train_lewm_decoder.py \
  --checkpoint weights/lewm_libero_v2_200/last.ckpt \
  --config weights/lewm_libero_v2_200/config.yaml \
  --libero-root /workspace/datasets/libero_spatial \
  --code-root /workspace/prism-ben \
  --max-demos 100 --per-task 10 \
  --frames-per-demo 40 \
  --epochs 40 --batch-size 32 --encode-batch 32 \
  --amp \
  --output-dir results/lewm_decoder \
  2>&1 | tee results_lewm_decoder_train.log
```

Decoder outputs:

- `results/lewm_decoder/decoder_best.pt` — NEW decoder weights, separate from LeWM (contains architecture metadata and decoder-only split)
- `results/lewm_decoder/decoder_metrics.csv` — train/validation pixel MSE each epoch
- `results/lewm_decoder/encoder_reconstruction_preview.png` — qualitative decoder reconstruction validation
- `results/lewm_decoder/run_args.json`

The decoder train/val split is **episode-disjoint for training this decoder**, but not proven independent of the LeWM world-model training split. Latent representations are always generated using the frozen LeWM encoder. Frame sampling covers the full chosen episodes, rather than only first few frames.

## 3. Produce an MP4

```bash
python lewm_libero_test_scripts/render_lewm_video.py \
  --checkpoint weights/lewm_libero_v2_200/last.ckpt \
  --config weights/lewm_libero_v2_200/config.yaml \
  --decoder results/lewm_decoder/decoder_best.pt \
  --libero-root /workspace/datasets/libero_spatial \
  --code-root /workspace/prism-ben \
  --task black_bowl_from_table_center \
  --start 15 --horizon 50 --fps 8 \
  --output results/lewm_video/black_bowl_comparison.mp4
```

Default `--split val` tries an episode held out while training the decoder. Use `--split any` to choose any demo. `--episode demo_8` chooses a specific demo within matching task files. Start/horizon must fit within the selected episode (the renderer clips horizon to available length).

Outputs:

- `results/lewm_video/black_bowl_comparison.mp4` — 3-row video
- `results/lewm_video/black_bowl_comparison.png` — key-frame montage
- `results/lewm_video/black_bowl_comparison.json` — RGB and latent error per predicted step

For an exact episode:

```bash
python lewm_libero_test_scripts/render_lewm_video.py \
  --checkpoint weights/lewm_libero_v2_200/last.ckpt \
  --config weights/lewm_libero_v2_200/config.yaml \
  --decoder results/lewm_decoder/decoder_best.pt \
  --libero-root /workspace/datasets/libero_spatial \
  --code-root /workspace/prism-ben \
  --task black_bowl_from_table_center --episode demo_8 --split any \
  --start 20 --horizon 50 \
  --output results/lewm_video/demo8.mp4
```

## Interpretations and key limitations

1. **Inspect reconstruction row BEFORE imagined row.** If the encoder-only decoder cannot reconstruct a ground-truth RGB frame, the imagined RGB row is not a reliable diagnostic of prediction quality.
2. The decoder was trained from **encoder** latent distribution, but open-loop predictions can lie outside that distribution. Strange imagined images can reflect decoder distribution shift, not only world-model errors. If it is hard to interpret, a *predictor decoder* trained on predicted embeddings is a second, separate experiment (as in the official LeWM implementation).
3. Pixel reconstruction of a full image from a single 192D [CLS] token is fundamentally underdetermined; blurry video is possible even if control-relevant latent features are useful.
4. `--image-mode imagenet` is consistent with the **previous local evaluator**, not verified with the HF checkpoint's original LIBERO training pipeline. Options: `imagenet`, `flipud_imagenet`, `zero_one`, `minus_one_one`. Choose the SAME mode for decoder training and inference; renderer obtains mode from decoder checkpoint. Verify orientation/normalization using author code before publication.
5. The default action input is RAW 7D. `--action-stats /path/to/training_stats.npz` on renderer optionally loads original training action mean/std. The correct action convention and time alignment must be checked against original training code; no evaluation-set action statistics are inferred by this visualization script.
6. The output frames are predictions **conditioned on actions recorded in demonstrations**, not actions generated by planning in the simulation. Thus no success rate is calculated.
7. `model.load_state_dict(strict=True)` detects parameter incompatibility but does not verify input semantics.

## Independent smoke verification

The Python files were syntax checked and end-to-end tested with synthetic HDF5 demonstrations and a small mock frozen model (one training epoch, new decoder checkpoint, MP4, PNG, JSON). The real Hugging Face checkpoint was **not** available in the artifact-building environment. Therefore, actual Vast.ai execution is still necessary to validate the pretrained model behavior and image quality.

## Further reference

Upstream LeWM provides `train_encoder_decoder.py` and `train_predictor_decoder.py` for encoder/predictor visualizations:

https://github.com/aspamers/lewm

Our scripts are a simplified, LIBERO-specific diagnostic implementation and do not claim pixel-level reconstruction fidelity equal to the author's official `CLSDecoder`.
