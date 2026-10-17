# Changelog — PRISM LIBERO patched pipeline

- Preserved two alternative head architectures: Flat Gaussian MLP and Chunked Prior with learned latent anchors.
- Added `train_lewm_managed.py` to retain original JEPA training loss with best validation checkpoint, every-ten-epoch and last checkpoint. Also writes Lightning trainer checkpoints with optimizer for resuming, `history.jsonl`, `history.csv`, `summary.json`.
- Added episode-disjoint val output options in `convert_libero_to_swm.py` and passed explicit validation HDF5 to LeWM trainer. Training normalization fitted from training HDF5.
- Fixed LeWM planning context to three consecutive image latents plus two past 35D normalized action blocks; predicted candidate block added as third. Updated context after actual five-action execution.
- Unified action normalization, clipping and optional gripper binarization across LeWM predictions and executed actions.
- Corrected success reporting to use LIBERO `env.check_success` rather than generic episode `done`.
- Converted Prior Head gate to non-blocking checkpoint quality annotation. Best, last and every-10-epoch heads are saved. Added train-only Prior Head action scaler, full windows as default, resume and patience.
- Added JSONL logs containing every executed raw action and per-five-step latent prediction, final/subgoal cost gaps, EEF/object/gripper states, optional video and CSV summary. Partial 5-step blocks are excluded from prediction-error summary.
- Added `analyze_trace.py`, `summarize_eval.py` and mock-LIBERO synthetic end-to-end checks, including checkpoint epoch_010 and closed-loop success semantics.

Verification: Python compilation; bash -n; test_pipeline.py synthetic converter/cache/prior/planner/fake-LIBERO evaluator; Lightning callback tested separately on CPU toy model. Real `stable-worldmodel` LeWM training and MuJoCo LIBERO simulation cannot be executed in the authoring environment; GPU and real benchmark results remain unverified.
