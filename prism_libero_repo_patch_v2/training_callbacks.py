"""Epoch summaries; does not change the model or objective."""
import csv
import json
import time
from pathlib import Path
import torch
from lightning.pytorch.callbacks import Callback


class EpochSummary(Callback):
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.started = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.started = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        row = {'epoch':trainer.current_epoch+1, 'seconds':round(time.monotonic()-self.started,2)}
        metrics = trainer.callback_metrics
        for stage in ('train','val'):
            for name in ('loss','mse','sigreg_loss'):
                key=f'{stage}/{name}'
                value=metrics.get(key)
                row[key]=float(value.detach().cpu()) if value is not None else None
        row['lr']=trainer.optimizers[0].param_groups[0]['lr']
        path=self.run_dir/'epoch_metrics.csv'
        exists=path.exists()
        with path.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if not exists: writer.writeheader()
            writer.writerow(row)
        with (self.run_dir/'epoch_metrics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        values=' | '.join(f'{k}={v:.6f}' if isinstance(v,float) else f'{k}={v}' for k,v in row.items())
        print('\n[EPOCH] '+values,flush=True)

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            # Export actual final model even when early stopping ends before max_epochs.
            torch.save(pl_module.model,self.run_dir/'last_object.ckpt')
            for callback in trainer.checkpoint_callbacks:
                if callback.monitor is not None:
                    score=callback.best_model_score
                    info={'monitor':callback.monitor,'checkpoint':callback.best_model_path,
                          'score':float(score.cpu()) if score is not None else None,
                          'completed_epochs':trainer.current_epoch}
                    (self.run_dir/'best_checkpoint.json').write_text(json.dumps(info,indent=2))
                    print('[BEST] '+json.dumps(info),flush=True)
