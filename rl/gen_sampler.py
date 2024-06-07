# main file

import hydra
import numpy as np
import torch

from kspace_classifier.data_modules import MRDataModule

from hydra.utils import get_original_cwd, to_absolute_path
from hydra.core.hydra_config import HydraConfig
import os


from tqdm import tqdm
import logging
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, sampler
import joblib


def get_data(config):

    return MRDataModule(config=config, dev_mode=config.dev_mode)


def label_to_bin(label_):
    val = 0

    keys = list(label_.keys())
    for i in range(1, len(label_) + 1):

        val = val + 2 ** (i - 1) * label_[keys[i-1]]

    return val


def label_to_bin_with_target(label_, target_class):
    val = 0

    keys = list(label_.keys())
    if target_class in keys:

        pass

    for i in range(1, len(label_) + 1):
        if keys[i-1] == target_class and label_[keys[i-1]] == torch.tensor([1]):
            val = torch.tensor([1])

            break
        else:
            val = torch.tensor([0])

    return val


def get_sampler_weights_binning(
    dataset, save_filename=None, target_class=None,
):
    Y_tr = []

    for i in tqdm(range(len(dataset))):
        if target_class is None:
            label = label_to_bin(dataset[i]["label"]).item()
        else:
            label = label_to_bin_with_target(
                dataset[i]["label"], target_class=target_class).item()
        Y_tr.append(label)

    Y_tr = np.array(Y_tr).astype(int)

    unique_vals = np.unique(Y_tr)

    if target_class is None:
        class_sample_count = np.array(
            [len(np.where(Y_tr == t)[0])
             for t in range(0, max(unique_vals) + 1)]
        )
    else:
        class_sample_count = np.array(
            [len(np.where(Y_tr == t)[0]) for t in unique_vals]
        )
        class_sample_count = []
        for t in unique_vals:
            if t == 0:

                class_sample_count.append(0)
            else:
                class_sample_count.append(len(np.where(Y_tr == t)[0]))

    weight = [1.0 / cnt if cnt > 0 else 0 for cnt in class_sample_count]
    samples_weight = np.array([weight[t] for t in Y_tr])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    joblib.dump(sampler, save_filename)
    print(f"sampler saved to {save_filename}")


def get_label(dataset):
    return [
        label_to_bin(dataset[idx]["label"]).item()
        for idx in tqdm(range(0, len(dataset)))
    ]


@hydra.main(version_base=None, config_path='cfgs', config_name='train_asmr')
def main(cfg):

    print(f"Current working directory : {os.getcwd()}")
    print(f"hydra path:{HydraConfig.get().run.dir}")
    run_dir = Path(HydraConfig.get().run.dir)
    print(f"run_dir:{run_dir}")
    print(f"data:{cfg.env.dataset}")
    cfg.env.batch_size = cfg.num_envs

    data_module = get_data(cfg.env)

    sp = get_sampler_weights_binning(
        data_module.train_dataset, f"./samplers/{cfg.env.dataset}_sampler.p")


if __name__ == "__main__":
    main()
