import argparse
import click

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from kspace_classifier.classification_modules import (
    ARMSClassifier,
    MRClassifier,
)
from kspace_classifier.data_modules import MRDataModule
from kspace_classifier.utils.utils import get_config
from kspace_classifier.utils.callbacks import LogMaskCallback


def get_data(config: argparse.Namespace, dev_mode: bool) -> pl.LightningDataModule:
    print("Using workers -> ", config.num_workers)
    return MRDataModule(config=config, dev_mode=dev_mode)


def get_model(config: argparse.ArgumentParser) -> MRClassifier:
    if config.classifier_type == "arms":
        model = ARMSClassifier(config=config)
    else:
        raise NotImplementedError(
            f"classifier type {config.classifier_type} not implemented"
        )

    if config.resume_train:
        print(f"Loading ckpt from {config.ckpt_path}")
        model = model.load_from_checkpoint(config.ckpt_path)
        resume_from_checkpoint = config.ckpt_path
    else:
        resume_from_checkpoint = None

    return model, resume_from_checkpoint


def train_model(
    config: argparse.Namespace,
    datamodule: pl.LightningDataModule,
    offline: bool,
) -> pl.LightningModule:

    print(config)
    project_name = (
        config.project_name
        if config.project_name
        else f"kspace_classifier_{config.dataset}"
    )

    wandb_logger = WandbLogger(
        project=project_name,
        save_dir=config.ckpt_dir,
        log_model=True if not offline else False,
        offline=offline,
        config=config
    )

    checkpoint_filename = (
        f"{config.classifier_type}-{config.dataset}-{config.model_type}-"
        + "{epoch}-{best_metric:.2f}"
    )

    model, resume_from_checkpoint = get_model(config)

    # checkpoint callback
    model_checkpoint = ModelCheckpoint(
        monitor="best_metric",
        filename=checkpoint_filename,
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    # learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # mask monitoring
    # mask_callback = LogMaskCallback()


    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=config.n_epochs,
        logger=wandb_logger,
        deterministic=False,
        callbacks=[model_checkpoint, lr_monitor],
        gradient_clip_val=None,#config.grad_clip_val,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=10,
        accumulate_grad_batches=config.accumulate_grad_batches,
    )
    torch.use_deterministic_algorithms(True, warn_only=True)

    # train the models
    trainer.fit(model, datamodule, ckpt_path=resume_from_checkpoint)

    # test the models : disabling testing for now, validation is still performed in train loop
    # trainer.test(model, datamodule, ckpt_path=resume_from_checkpoint)


@click.command()
@click.option("--config_path", required=True, help="Config file path.")
@click.option(
    "--offline",
    type=bool,
    default=False,
    required=False,
    help="run wandb in offline mode.",
)
@click.option(
    "--dev_mode",
    type=bool,
    default=False,
    required=False,
    help="run code in debug mode.",
)
def main(config_path, offline, dev_mode):
    print(config_path)
    config = get_config(config_path)
    datamodule = get_data(config=config, dev_mode=dev_mode)

    if dev_mode:
        config.use_weighted_sampler = False

    train_model(config=config, datamodule=datamodule, offline=offline)


if __name__ == "__main__":
    main()
