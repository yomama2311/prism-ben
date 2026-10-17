"""LeWM trainer for LIBERO with episode-disjoint validation and managed checkpoints.

Run at the root of prism-jepa:
  LIBERO_TRAIN_H5=/abs/train.h5 LIBERO_VAL_H5=/abs/val.h5 \\
  python prism_libero/train_lewm_managed.py data=libero_managed \\
      trainer.max_epochs=50 wandb.enabled=False +checkpoint_every=10

Requires the same stable-worldmodel[train], Hydra and Lightning environment as the
repository's original train.py. Keeps the same loss, encoder and predictor.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor


def lejepa_forward(self, batch, stage, cfg):
    """Equivalent to the upstream LeWM `train.py` forward."""
    ctx_len, n_preds = int(cfg.wm.history_size), int(cfg.wm.num_preds)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    out = self.model.encode(batch)
    pred = self.model.predict(out["emb"][:, :ctx_len], out["act_emb"][:, :ctx_len])
    target = out["emb"][:, n_preds:]
    out["pred_loss"] = (pred - target).pow(2).mean()
    out["sigreg_loss"] = self.sigreg(out["emb"].transpose(0, 1))
    out["loss"] = out["pred_loss"] + cfg.loss.sigreg.weight * out["sigreg_loss"]
    losses = {f"{stage}/{k}": v.detach() for k, v in out.items() if "loss" in k}
    self.log_dict(losses, on_step=True, on_epoch=True, sync_dist=True)
    return out


class ManagedLeWMCheckpoint(pl.Callback):
    """Best by validation prediction loss; every-N and last; JSONL metrics.

    Save object checkpoints (`torch.load` directly returns JEPA) AND full Lightning
    trainer states (optimizer, schedulers) for resuming.
    """
    def __init__(self, out_dir: Path, every=10, patience=12, min_delta=1e-5):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.full = self.out_dir / "checkpoints"
        self.full.mkdir(parents=True, exist_ok=True)
        self.every = int(every)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best, self.best_epoch, self.bad = float("inf"), 0, 0
        self.records = []

    def _metric(self, trainer):
        m = trainer.callback_metrics
        for name in ("validate/pred_loss", "validate/pred_loss_epoch", "val/mse", "val/pred_loss", "validate/mse"):
            if name in m:
                v = m[name]
                return name, float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
        raise RuntimeError("Cannot find validation pred_loss/MSE. Available metrics: " + str(list(m)))

    def _object(self, pl_module, path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(pl_module.model, tmp)
        os.replace(tmp, path)

    def _trainer(self, trainer, path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        trainer.save_checkpoint(str(tmp))
        os.replace(tmp, path)

    def _meta(self, obj):
        tmp = self.out_dir / "summary.json.tmp"
        tmp.write_text(json.dumps(obj, indent=2) + "\n")
        os.replace(tmp, self.out_dir / "summary.json")

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        name, score = self._metric(trainer)
        if not torch.isfinite(torch.tensor(score)):
            raise RuntimeError(f"Validation metric is nonfinite at epoch {epoch}: {score}")
        metrics = {}
        for k, v in trainer.callback_metrics.items():
            try:
                q = float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
            except (TypeError, ValueError):
                continue
            if torch.isfinite(torch.tensor(q)):
                metrics[str(k)] = q
        is_best = score < self.best - self.min_delta
        if is_best:
            self.best, self.best_epoch, self.bad = score, epoch, 0
            self._object(pl_module, self.out_dir / "best_object.ckpt")
            self._trainer(trainer, self.full / "best.ckpt")
        else:
            self.bad += 1
        row = {"epoch": epoch, "seconds": time.time(), "monitor": name,
               "monitor_value": score, "best_metric": self.best,
               "best_epoch": self.best_epoch, "is_best": is_best, **metrics}
        self.records.append(row)
        with (self.out_dir / "history.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        fields = list(dict.fromkeys(key for r in self.records for key in r))
        with (self.out_dir / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.records)
        self._meta(dict(best_epoch=self.best_epoch, best_metric=self.best,
                        monitor=name, completed_epochs=epoch,
                        best_object=str(self.out_dir / "best_object.ckpt"),
                        best_trainer=str(self.full / "best.ckpt")))
        print(f"[LEWM-EPOCH] epoch={epoch} val_prediction_mse={score:.8f} "
              f"best_epoch={self.best_epoch} {'*best' if is_best else ''}", flush=True)
        if self.patience > 0 and self.bad >= self.patience:
            trainer.should_stop = True
            print(f"[LEWM-EARLY-STOP] patience={self.patience} best_epoch={self.best_epoch}", flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        self._object(pl_module, self.out_dir / "last_object.ckpt")
        self._trainer(trainer, self.full / "last.ckpt")
        if self.every > 0 and epoch % self.every == 0:
            self._object(pl_module, self.out_dir / f"epoch_{epoch:03d}_object.ckpt")
            self._trainer(trainer, self.full / f"epoch_{epoch:03d}.ckpt")
        print(f"[LEWM-CKPT] epoch={epoch} last=last_object.ckpt "
              f"periodic={'yes' if self.every > 0 and epoch % self.every == 0 else 'no'}", flush=True)


@hydra.main(version_base=None, config_path="../config/train", config_name="lewm")
def run(cfg):
    torch.set_float32_matmul_precision("high")
    if int(cfg.wm.history_size) != 3 or int(cfg.wm.num_preds) != 1:
        print("[WARN] The LIBERO evaluator assumes history_size=3,num_preds=1", flush=True)
    train_path = str(cfg.data.dataset.path)
    val_path = os.environ.get("LIBERO_VAL_H5", "")
    if not Path(train_path).exists():
        raise FileNotFoundError(f"Training HDF5 not found: {train_path}")
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor("pixels", "pixels", cfg.img_size)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform
    if val_path:
        if Path(val_path).resolve() == Path(train_path).resolve():
            raise ValueError("Train and validation HDF5 must be distinct")
        if not Path(val_path).exists():
            raise FileNotFoundError(val_path)
        kwargs = OmegaConf.to_container(cfg.data.dataset, resolve=True)
        kwargs["path"] = val_path
        val_set = swm.data.HDF5Dataset(**kwargs, transform=None)
        val_set.transform = transform  # uses TRAIN statistics; no val leakage
        train_set = dataset
        print(f"[LEWM-DATA] episode-disjoint: train={train_path} val={val_path}", flush=True)
    else:
        g = torch.Generator().manual_seed(cfg.seed)
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=g)
        print("[WARN] LIBERO_VAL_H5 unset: random clip-level split may leak overlapping windows!", flush=True)

    gen = torch.Generator().manual_seed(cfg.seed)
    trloader = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True,
                                           drop_last=True, generator=gen)
    valloader = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False,
                                            drop_last=False)
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale, patch_size=cfg.patch_size, image_size=cfg.img_size,
        pretrained=False, use_mask_token=False)
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    predictor = ARPredictor(num_frames=cfg.wm.history_size, input_dim=embed_dim,
                            hidden_dim=hidden_dim, output_dim=hidden_dim, **cfg.predictor)
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    predictor_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048,
                         norm_fn=torch.nn.BatchNorm1d)
    jepa = JEPA(encoder, predictor, action_encoder, projector, predictor_proj)
    optimizers = {"model_opt": {"modules": "model", "optimizer": dict(cfg.optimizer),
                                "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
                                "interval": "epoch"}}
    module = spt.Module(model=jepa, sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
                        forward=partial(lejepa_forward, cfg=cfg), optim=optimizers)
    run_id = cfg.get("subdir") or "libero_managed"
    # Managed output root is explicit and not tied to the default Lightning logger.
    home = Path(os.environ.get("STABLEWM_HOME", str(Path.home() / ".stable-wm")))
    out = Path(os.environ.get("LIBERO_LEWM_RUN_DIR", str(home / run_id)))
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml")
    callback = ManagedLeWMCheckpoint(
        out, every=int(cfg.get("checkpoint_every", 10)),
        patience=int(cfg.get("early_stop_patience", 12)))
    logger = WandbLogger(**cfg.wandb.config) if cfg.wandb.enabled else None
    # A no-op checkpoint callback disables Lightning's default extra saver. Our
    # managed callback writes both object-model and trainer-state checkpoints.
    from lightning.pytorch.callbacks import ModelCheckpoint
    noop = ModelCheckpoint(dirpath=out / "lightning_unused", save_top_k=0, save_last=False)
    trainer = pl.Trainer(**cfg.trainer, callbacks=[callback, noop],
                         num_sanity_val_steps=1, logger=logger, enable_checkpointing=True)
    data_module = spt.data.DataModule(train=trloader, val=valloader)
    manager = spt.Manager(trainer=trainer, module=module, data=data_module,
                          ckpt_path=cfg.get("resume_ckpt", None))
    manager()
    if trainer.is_global_zero:
        print(f"[LEWM-BEST] {out / 'best_object.ckpt'}", flush=True)
        print(f"[LEWM-LAST] {out / 'last_object.ckpt'}", flush=True)


if __name__ == "__main__":
    run()
