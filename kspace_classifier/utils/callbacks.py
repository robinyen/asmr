import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb
import pytorch_lightning as pl

from kspace_classifier.classification_modules.classification_module import MRClassifier
from kspace_classifier.utils import transforms


class LogGenerationCallback(Callback):
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """Called when the validation batch ends."""
        if batch_idx % 32 == 0:
            recon_dict = pl_module.reconstruct_image(batch)
            recon_images = recon_dict["output"]
            target_images = recon_dict["target"]

            trainer.logger.experiment.log(
                {
                    "outputs": [wandb.Image(x) for x in recon_images],
                    "targets": [wandb.Image(x) for x in target_images],
                    "global_step": trainer.global_step,
                }
            )


class LogMaskCallback(Callback):
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: MRClassifier,
        outputs,
        batch,
        batch_idx,
    ):
        """Called when the validation batch ends."""
        if batch_idx == 0:
            if pl_module.mask_fn is not None:

                mask = pl_module.mask_fn.get_mask()
                x = torch.ones(pl_module.config.kspace_shape[0], pl_module.config.kspace_shape[1],pl_module.config.in_channels,  device=mask.device)
                if len(mask.shape) == 4 :
                    mask = mask[0,0,:,:].squeeze(1)
                
                masked_x = mask * x

                if "vol" not in pl_module.config.dataset:
                    trainer.logger.experiment.log(
                        {
                            "mask": [wandb.Image(masked_x.cpu().numpy())],
                            "global_step": trainer.global_step,
                        }
                    )
            else:
                pass