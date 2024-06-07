import argparse
import math
import os
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import joblib

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from joblib import dump, load
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, sampler
from torchvision import transforms
from tqdm.auto import tqdm


class MRDataset(Dataset):
    def __init__(
        self,
        config: argparse.ArgumentParser,
        mode: str,
        dev_mode: bool = False,
        transforms: transforms.Compose = None,
    ):
        super().__init__()
        # read csv file with filenames
        self.config = config
        self.config.split_csv_file = Path(self.config.split_csv_file)
        assert self.config.split_csv_file.is_file()

        self.metadata = pd.read_csv(self.config.split_csv_file)
        if dev_mode:
            print("**" * 10, "Using dev mode", "**" * 10)
            self.metadata = self.metadata.iloc[:4000]

        assert "data_split" in self.metadata.columns
        assert "location" in self.metadata.columns

        metadata_grouped = self.metadata.groupby("data_split")
        self.metadata_by_mode = {
            e: metadata_grouped.get_group(e) for e in metadata_grouped.groups
        }

        self.transforms = transforms

        self.mode = mode
        assert mode in self.metadata_by_mode

    def __len__(self):
        assert self.mode in self.metadata_by_mode
        return self.metadata_by_mode[self.mode].shape[0]

    def apply_transforms(self, image, kspace):
        if 'val' in self.mode or 'test' in self.mode:
            return image, kspace

        if self.transforms is not None:
            if image is not np.NaN and kspace is not np.NaN:
                
                image = self.transforms(image)

                stacked_tensor = np.stack([kspace.real, kspace.imag], axis=2)
                stacked_tensor = self.transforms(stacked_tensor)
                kspace = stacked_tensor[0] + 1j * stacked_tensor[1]

            elif image is not None and kspace is np.NaN:
                image = self.transforms(image)

            elif image is None and kspace is not np.NaN:
                stacked_tensor = np.stack([kspace.real, kspace.imag])
                stacked_tensor = self.transforms(stacked_tensor)

                kspace = stacked_tensor[0] + 1j * stacked_tensor[1]

        return image, kspace

    def get_metadata_value(self, index, key):
        assert self.mode in self.metadata_by_mode
        assert key in self.metadata_by_mode[self.mode].iloc[index]
        return self.metadata_by_mode[self.mode].iloc[index][key]

    def __getitem__(self, index):
        raise NotImplementedError
