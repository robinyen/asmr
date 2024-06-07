import argparse
import math
import os
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from joblib import dump, load
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, sampler
from torch.utils.data.dataset import ConcatDataset
from tqdm.auto import tqdm
from .mr_dataset import MRDataset  


import time


class BrainDataset(MRDataset):
    def __init__(
        self,
        config: argparse.ArgumentParser,
        mode: str,
        coil_type="sc",
        dev_mode: bool = False,
        image_only: bool = False,
        pad_target: bool = True,
        exclude_kspace: bool = False,
        resize_target: bool = False,
        ignore_target: bool = False,
    ):
        super().__init__(config=config, mode=mode, dev_mode=dev_mode)
        self.coil_type = coil_type
        assert self.coil_type in {"mc", "sc"}

        self.exclude_kspace = exclude_kspace

        self.image_only = image_only
        self.pad_target = pad_target
        self.resize_target = resize_target
        self.ignore_target = ignore_target
        if self.coil_type == "mc":
            self.pad_shape = (384, 384)
        else:
            self.pad_shape = (768, 400)

        assert not (self.pad_target and self.ignore_target)
        assert not (self.resize_target and self.ignore_target)
        assert not (self.resize_target is True and self.pad_target is True)

    def parse_label(self, label_arr: Sequence[str]):
        label_arr = label_arr.replace("[", "").replace("]", "").replace("'", "")
        label_arr = label_arr.split(",")

        labels_ = ["Edema", "Enlarged ventricles", "Mass", "Abnormal"]

        new_labels = []
        if "None" in label_arr:
            for _ in range(0, len(labels_)):
                new_labels.append(0.0)
            return {
                "Edema": torch.tensor([new_labels[0]]).long(),
                "Enlarged ventricles": torch.tensor([new_labels[1]]).long(),
                "Mass": torch.tensor([new_labels[2]]).long(),
                "Abnormal":  torch.tensor([new_labels[3]]).long()
            }

        """
            1. Edema
            2. Enlarged Ventricles
            3. Extra-Axial Mass
            4. Mass
        """

        ## labels placeholder for now..
        for label in label_arr:
            for i in range(0, len(labels_)):
                gl = labels_[i]
                if gl == "Abnormal":
                    new_labels.append(i + 1)
                    break
                elif gl.lower() in label.lower():
                    new_labels.append(i + 1)
                    break

        ohe_labels = []
        for i in range(0, len(labels_)):
            if i + 1 in new_labels:
                ohe_labels.append(1.0)
            else:
                ohe_labels.append(0.0)

        return {
                "Edema": torch.tensor([ohe_labels[0]]).long(),
                "Enlarged ventricles": torch.tensor([ohe_labels[1]]).long(),
                "Mass": torch.tensor([ohe_labels[2]]).long(),
                "Abnormal":  torch.tensor([ohe_labels[3]]).long()
        }

    def _pad_target_data(self, target):
        img_shape = self.pad_shape
        if target.shape[0] == img_shape[0] and target.shape[1] == img_shape[1]:
            return target

        n_cols, n_rows = target.shape[1], target.shape[0]

        n_rows_to_fill = img_shape[1] - n_cols
        n_cols_to_fill = img_shape[0] - n_rows

        zero_filled_img = np.zeros((img_shape[0], img_shape[1]), dtype=target.dtype)

        if n_cols_to_fill != 0 and n_rows_to_fill != 0:
            zero_filled_img[
                n_cols_to_fill // 2 : -n_cols_to_fill // 2,
                n_rows_to_fill // 2 : -n_rows_to_fill // 2,
            ] = target
        elif n_cols_to_fill == 0:
            zero_filled_img[:, n_rows_to_fill // 2 : -n_rows_to_fill // 2] = target
        else:
            zero_filled_img[n_cols_to_fill // 2 : -n_cols_to_fill // 2, :] = target

        return zero_filled_img



    def __getitem__(self, index):
        assert self.mode in self.metadata_by_mode
        loc = self.get_metadata_value(index, "location")

        info = self.metadata_by_mode[self.mode].iloc[index]
        kspace_key = "sc_kspace" if self.coil_type == "sc" else "mc_kspace"
        target_key = "recon_esc" if self.coil_type == "sc" else "recon_rss"

        if not (self.exclude_kspace and self.ignore_target):
            with h5py.File(loc) as f:
                keys = f.keys()

                kspace_data = np.NaN
                if not self.exclude_kspace:
                    kspace_data = f[kspace_key][:]

                target_data = np.NaN
                if not self.ignore_target:
                    target_data = f[target_key][:]

                if self.pad_target:
                    target_data = self._pad_target_data(target_data)



                parameters = {
                    kspace_key: kspace_data,
                    target_key: target_data,
                    "sc_kspace": f["sc_kspace"][:],
                    "volume_id": info.volume_id,
                    "slice_id": info.slice_id,
                    "label": self.parse_label(info.labels),
                    "data_split": info.data_split,
                    "dataset": info.dataset,
                    "location": info.location,
                    "max_value": info.max_value,
                }
        else:
            parameters = {
                "volume_id": info.volume_id,
                "slice_id": info.slice_id,
                "label": self.parse_label(info.labels),
                "data_split": info.data_split,
                "dataset": info.dataset,
                "location": info.location,
                "max_value": info.max_value,
            }

        sample = parameters
        return sample


def label_to_bin(label_):
    val = 0
    for i in range(1, len(label_) + 1):
        val = val + 2 ** (i - 1) * label_[-i]
    return val




def get_label(dataset):
    return [
        label_to_bin(dataset[idx]["label"]).item()
        for idx in tqdm(range(0, len(dataset)))
    ]
