import argparse
from pathlib import Path
from typing import List, Union
import joblib

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from kspace_classifier.data_modules.brain_data import BrainDataset

from .knee_data import KneeDataset



class MRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: argparse.ArgumentParser,
        dev_mode: bool,
    ):
        super().__init__()
        self.config = config
        assert self.config.task in ["classification", "reconstruction"]
        self.dev_mode = dev_mode

        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        if self.config.use_weighted_sampler:
            assert Path(self.config.train_sampler_filename).is_file()
            self.train_sampler_weights = joblib.load(self.config.train_sampler_filename)
        else:
            print(
                "***" * 10,
                #"Not using Weighted Random Sampler For Training RM",
                "***" * 10,
            )
            self.train_sampler_weights = None

        if self.config.dataset in ["knee", "knee_singlehead"]:
            self.dataset_class = KneeDataset
        elif self.config.dataset == "brain":
            self.dataset_class = BrainDataset
        elif self.config.dataset == "prostate_t2":
            self.dataset_class = ProstateT2Dataset
        elif self.config.dataset == "prostate_dwi":
            raise NotImplementedError
        else:
            raise NotImplementedError(f"Dataset {self.config.dataset} not implemented")

        self.setup()

    def load_combined_datasets(self):
        if self.config.combine_class_recon_splits:
            transforms = self.train_transforms if self.config.task == 'classification' else None

            train_class = self.dataset_class(
                config=self.config,
                mode="train_class",
                dev_mode=self.dev_mode,
                transforms=transforms,
            )
            train_recon = self.dataset_class(
                config=self.config,
                mode="train_recon",
                dev_mode=self.dev_mode,
                transforms=transforms,
            )
            self.train_dataset = ConcatDataset([train_class, train_recon])

            val_class = self.dataset_class(
                config=self.config, mode="val_class", dev_mode=self.dev_mode
            )
            val_recon = self.dataset_class(
                config=self.config, mode="val_recon", dev_mode=self.dev_mode
            )
            self.val_dataset = ConcatDataset([val_class, val_recon])

    def setup(self, stage=None):
        if self.config.task == "classification":
            train_mode = "train_class"
            val_mode = "val_class"
            test_mode = "test_class"
        elif self.config.task == "reconstruction":
            train_mode = "train_recon"
            val_mode = "val_recon"
            test_mode = None
        else:
            raise NotImplementedError(f"Task {self.config.task} not implemented")

        if self.config.combine_class_recon_splits:
            self.load_combined_datasets()
        else:
            self.train_dataset = self.dataset_class(
                config=self.config, mode=train_mode, dev_mode=self.dev_mode
            )
            self.val_dataset = self.dataset_class(
                config=self.config, mode=val_mode, dev_mode=self.dev_mode
            )
        if self.config.task == "classification":
            self.test_dataset = self.dataset_class(
                config=self.config, mode=test_mode, dev_mode=self.dev_mode
            )
        else:
            self.test_dataset = None

    def train_dataloader(self) -> DataLoader:
        if self.config.task == "classification":
            if self.config.use_weighted_sampler:
                sampling_dict = dict(sampler=self.train_sampler_weights)
            else:
                sampling_dict = dict(shuffle=True)
        elif self.config.task == "reconstruction":
            sampling_dict = dict(shuffle=True)

        return DataLoader(
            self.train_dataset,        
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=True,
            pin_memory=True,
            **sampling_dict,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.num_workers,
            shuffle=self.config.val_shuffle,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.config.task == "classification":
            return DataLoader(
                self.test_dataset,
                batch_size=self.config.val_batch_size,
                num_workers=self.config.num_workers,
            )
        elif self.task == "reconstruction":
            raise NotImplementedError("Reconstruction has no test set")
