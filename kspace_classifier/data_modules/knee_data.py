import argparse
from collections import namedtuple
import math
from typing import Sequence, Dict
import h5py
import numpy as np

import torch
from torchvision import transforms

from .mr_dataset import MRDataset

class KneeDataset(MRDataset):
    def __init__(
        self,
        config: argparse.ArgumentParser,
        mode: str,
        dev_mode: bool = False,
        transforms: transforms.Compose = None,
    ):
        super().__init__(
            config=config, mode=mode, dev_mode=dev_mode, transforms=transforms
        )
        assert self.config.coil_type in ["sc", "mc"]

    def parse_label(self, label_arr: Sequence[str]) -> Dict:
        label_arr = label_arr.replace("[", "").replace("]", "").replace("'", "")
        label_arr = label_arr.split(",")

        if "None" in label_arr:
            return dict(
                    acl=torch.zeros(1).long(),
                    mtear=torch.zeros(1).long(),
                    abnormal=torch.zeros(1).long(),
                )

        new_labels = []

        for label in label_arr:
            if "ACL" in label:
                new_labels.append(1)
            elif "Meniscus Tear" in label:
                new_labels.append(2)
            else:
                new_labels.append(3)

        abnormal = 1 if 3 in new_labels else 0
        mtear = 1 if 2 in new_labels else 0
        acl = 1 if 1 in new_labels else 0

        
        label_dict = dict(
                acl=torch.tensor([acl]).long(),
                mtear=torch.tensor([mtear]).long(),
                abnormal=torch.tensor([abnormal]).long(),
            )
        
        ret_dict = label_dict
            
        return ret_dict

    def __getitem__(self, index):
        assert self.mode in self.metadata_by_mode
        loc = self.get_metadata_value(index, "location")

        if self.config.coil_type == "sc":
            kspace_key, target_key = "sc_kspace", "recon_esc"
        elif self.config.coil_type == "mc":
            kspace_key, target_key = "mc_kspace", "recon_rss"
        else:
            raise NotImplementedError(f"coil type {self.config.coil_type} not implemented")

        with h5py.File(loc) as f:
            kspace_arr = f[kspace_key][:] if not self.config.ignore_kspace else np.NaN
            target_arr = f[target_key][:] if not self.config.ignore_image else np.NaN

            sampled_indices = f["sampled_indices"][:]
            target_arr, kspace_arr = self.apply_transforms(image=target_arr, kspace=kspace_arr)

        info = self.metadata_by_mode[self.mode].iloc[index]
        labels = self.parse_label(
            label_arr=info.labels,
        )
        
        kspace_arr = np.expand_dims(kspace_arr, axis=0)
        
        assert kspace_arr.shape == (self.config.in_channels, self.config.kspace_shape[0], self.config.kspace_shape[1]), kspace_arr.shape

        parameters = {
            kspace_key: kspace_arr,
            target_key: target_arr,
            "sampled_indices": sampled_indices,
            "volume_id": info.volume_id,
            "slice_id": info.slice_id,
            "data_split": info.data_split,
            "dataset": info.dataset,
            "location": info.location,
            "max_value": info.max_value,
            "label": labels,
        }

        return parameters
