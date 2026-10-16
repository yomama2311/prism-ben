import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from training_callbacks import EpochSummary
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    # Stable metric names independent of stable_pretraining's stage labels.
    metric_stage = "train" if self.training else "val"
    metrics = {
        f"{metric_stage}/loss": output["loss"].detach(),
        f"{metric_stage}/mse": output["pred_loss"].detach(),
        f"{metric_stage}/sigreg_loss": output["sigreg_loss"].detach(),
    }
    self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True,
                  batch_size=emb.shape[0], prog_bar=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    pl.seed_everything(cfg.seed, workers=True)
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    is_libero = cfg.data.get("kind") == "libero_raw"
    if is_libero:
        from libero_dataset import build_libero_datasets
        libero_run_dir = Path(cfg.output_dir).expanduser().resolve()
        # A fresh directory is required: this entry point intentionally trains from scratch.
        if (libero_run_dir / "config.yaml").exists():
            raise FileExistsError(f"Use a new output_dir; run already exists: {libero_run_dir}")
        train_set, val_set = build_libero_datasets(cfg, libero_run_dir)
    else:
        dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
        transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
        with open_dict(cfg):
            for col in cfg.data.dataset.keys_to_load:
                if col.startswith("pixels"):
                    continue

                normalizer = get_column_normalizer(dataset, col, col)
                transforms.append(normalizer)

                setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

        transform = spt.data.transforms.Compose(*transforms)
        dataset.transform = transform

        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )


    loader_kwargs = dict(cfg.loader)
    if loader_kwargs.get("num_workers", 0) == 0:
        loader_kwargs.pop("prefetch_factor", None)
        loader_kwargs["persistent_workers"] = False
    train = torch.utils.data.DataLoader(train_set, **loader_kwargs,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **loader_kwargs, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = libero_run_dir if is_libero else Path(swm.data.utils.get_cache_dir(), run_id)

    logger = [CSVLogger(save_dir=str(run_dir), name="logs")]
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(**cfg.wandb.config)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
        logger.append(wandb_logger)

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    monitor = cfg.get("checkpoint", {}).get("monitor", "val/mse")
    interval = int(cfg.get("checkpoint", {}).get("every_n_epochs", 10))
    if interval < 1:
        raise ValueError("checkpoint.every_n_epochs must be >= 1")
    best_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints" / "best", filename="best",
        monitor=monitor, mode="min", save_top_k=1, save_last=True,
        every_n_epochs=1, save_on_train_epoch_end=False,
        auto_insert_metric_name=False, enable_version_counter=False,
    )
    periodic_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints" / "periodic", filename="epoch-{epoch:03d}",
        save_top_k=-1, every_n_epochs=interval, save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )
    callbacks = [
        EpochSummary(run_dir),
        ModelObjectCallBack(dirpath=run_dir, filename=cfg.output_model_name,
                            epoch_interval=interval),
        best_callback, periodic_callback,
    ]
    early = cfg.get("early_stopping", {})
    if early.get("enabled", True):
        callbacks.append(EarlyStopping(
            monitor=monitor, mode="min", patience=int(early.get("patience", 15)),
            min_delta=float(early.get("min_delta", 1e-4)),
            check_finite=True, strict=True, verbose=True,
            check_on_train_epoch_end=False,
        ))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=None,  # fresh training; no Cube/PushT checkpoint loaded
    )

    manager()
    return


if __name__ == "__main__":
    run()
