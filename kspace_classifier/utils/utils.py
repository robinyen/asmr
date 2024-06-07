import numpy as np
from joblib import dump
from typing import Sequence, Tuple
import torch
import yaml
import argparse

from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm


def string_to_list(string):
    return list(string.split(","))

def load_yaml(path):
    with open(path, mode="r") as f:
        return yaml.safe_load(f)

def get_config(config_path: str) -> argparse.ArgumentParser:

    config = load_yaml(config_path)
    config = argparse.Namespace(**config)
    return config


def get_sampler_weights(dataset, save_filename):
    Y_tr = []

    for i in tqdm(range(len(dataset))):
        label = sum(list(dataset[i]["label"].values())).item()
        Y_tr.append(label)

    Y_tr = np.array(Y_tr).astype(np.long)

    class_sample_count = np.array(
        [len(np.where(Y_tr == t)[0]) for t in np.unique(Y_tr)]
    )

    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in Y_tr])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    dump(sampler, save_filename)


def transfer_bal_dataloader_to_device(dataloader, device, dataset, n_samples=200):        
    if dataset == "cifar10":
        X, Y = [], []
        count = 0
        for batch in dataloader:
            batch["sc_kspace"] = batch["sc_kspace"].to(device)
            batch["label"] = batch["label"]['cifar10'].to(device)
    
            X.append(batch['sc_kspace'])
            Y.append(batch['label'])
            
            count += batch['sc_kspace'].shape[0]
            
            if count >= n_samples:
                break

        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)

        print(X.shape, Y.shape)

        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        return None, dataloader
    else:
        pass
    
    dataloader_on_device = []
    print(f"Warning: Only using {n_samples} samples for selection")

    normal = []
    abnormal = []

    for _, batch in enumerate(dataloader):
        batch["sc_kspace"] = batch["sc_kspace"].to(device)
        batch_size = batch["sc_kspace"].shape[0]
        batch_keys = list(batch.keys())

        for sample_idx in range(batch_size):
            label_sum = sum(
                [
                    batch["label"][label_name][sample_idx]
                    for label_name in batch["label"]
                ]
            )

            sample = {}
            for key in batch_keys:
                if key == "label":
                    sample[key] = {}
                    for label_key in batch[key]:
                        sample[key][label_key] = batch[key][label_key][sample_idx]
                else:
                    sample[key] = batch[key][sample_idx]

            if label_sum == 0:
                if len(normal) <= n_samples // 2:
                    normal.append(sample)
            else:
                if (
                    len(abnormal) <= n_samples // 2
                    and sample["label"]["abnormal"] == 1
                ):
                    abnormal.append(sample)

        if (
            len(normal) > n_samples // 2
            and len(abnormal) > n_samples // 2
        ):
            break

    dataloader_on_device = normal + abnormal

    # divide dataloader_on_device into batches
    batch_size = min(128, n_samples // 4)
    n_batches = len(dataloader_on_device) // batch_size

    print("Number of samples: ", len(dataloader_on_device))
    print("Number of batches: ", n_batches)

    ret = []
    for idx in range(n_batches):
        new_batch = {key: [] for key in batch_keys}

        batch_to_split = dataloader_on_device[
            idx * batch_size : (idx + 1) * batch_size
        ]

        for key in batch_keys:
            if key in ["slice_id", "max_value"]:
                new_batch[key] = torch.tensor([x[key] for x in batch_to_split])
            elif key == "label":
                new_batch[key] = {}
                for label_key in batch["label"]:
                    new_batch[key][label_key] = torch.stack(
                        ([x[key][label_key] for x in batch_to_split]), dim=0
                    )                
            elif key == "location":
                new_batch[key] = [x[key] for x in batch_to_split]
            elif key not in [
                "label",
                "location",
                "data_split",
                "dataset",
                "volume_id",
                "slice_id",
                "max_value",
            ]:
                new_batch[key] = torch.stack(
                    ([x[key] for x in batch_to_split]), dim=0
                )
        ret.append(new_batch)

    print(f"Number of batches: {len(ret)}")

    X, Y = [], []   

    for batch in ret:
        X.append(batch["sc_kspace"])
        Y.append(batch["label"]["abnormal"])   

    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)     

    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    return ret, dataloader