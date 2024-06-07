# main file

import hydra
import numpy as np
import torch
from kspace_classifier.data_modules import MRDataModule
from torch.utils.data import Dataset, DataLoader

from rl.asmr_env import Asmr_Env
from kspace_classifier.classification_modules import ARMSClassifier
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from collections import deque, defaultdict
from rl.utils import eval_mode, set_seed_everywhere
from hydra.utils import get_original_cwd, to_absolute_path
from hydra.core.hydra_config import HydraConfig
import torch.nn as nn
import os
import joblib


from logger import Logger
import time
from tqdm import tqdm

import logging
from pathlib import Path
from torch.utils.checkpoint import checkpoint


from kspace_classifier.metrics.classification_metrics import (
    compute_accuracy,
    evaluate_classifier,
)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_data(config):
    return MRDataModule(config=config, dev_mode=config.dev_mode)


@hydra.main(version_base=None, config_path='cfgs', config_name='eval_asmr')
def main(cfg):
    print(cfg)
    print(f"Current working directory : {os.getcwd()}")
    print(f"hydra path:{HydraConfig.get().run.dir}")
    run_dir = Path(HydraConfig.get().run.dir)

    data_module = get_data(cfg.env)

    if cfg.eval_data == 'val':
        val_loader = data_module.val_dataloader()
    else:
        val_loader = data_module.test_dataloader()

    #logging.info(f"-----length of eval_dataloader:{len(val_loader)}-----")

    set_seed_everywhere(cfg.seed)

    import wandb

    wandb.init(
        project=cfg.project_name,
        sync_tensorboard=True,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=run_dir,
    )


    print(cfg.env)

    eval_envs = prepare_evaluate_envs(cfg, val_loader)

    
    ac = hydra.utils.instantiate(
        cfg.model, action_space=eval_envs.action_space)

    ac.to(cfg.device)

    work_dir = Path.cwd()

    global global_step
    global_step = 0

    load_snapshot(ac, cfg.load_from_snapshot_base_dir)

    eval_stats = {}
    start = cfg.eval_range[0]
    end = cfg.eval_range[1]
    for num_line in range(start, end+1):
        logging.info(
            f"=================Eval at budget of {num_line} line=================")
        eval_envs.set_budget(num_line)

        eval_stats[num_line] = evaluate(ac, eval_envs, None, num_line=num_line)

    # dump the evaluation stats
    ckpt_path = Path(cfg.load_from_snapshot_base_dir)
    if cfg.eval_data == 'val':
        joblib.dump(eval_stats, ckpt_path /
                    f"ppo_ts_val_stats_{start}_{end}.pkl")
    else:
        joblib.dump(eval_stats, ckpt_path /
                    f"ppo_ts_test_stats_{start}_{end}.pkl")


def load_snapshot(model, load_from_snapshot_base_dir):

    snapshot_base_dir = Path(load_from_snapshot_base_dir)
    snapshot = snapshot_base_dir / f'best_model.pt'
    if not snapshot.exists():
        logging.info(
            f"---WARNING---[Train.py] snapshot:{snapshot} not exists---WARNING---")
        return None
    logging.info(f"[eval_asmr.py] load snapshot:{snapshot}")
    model.load_state_dict(torch.load(snapshot))


def prepare_evaluate_envs(cfg, val_loader):

    observation_space = cfg.env.observation_space

    if cfg.env_version == "Asmr_Env":
        logging.info("===============using Asmr_Env in evaluation")

        reward_model = ARMSClassifier.load_from_checkpoint(
            cfg.env.reward_model_ckpt,  map_location=cfg.device)

        envs = Asmr_Env(reward_model, val_loader,
                        observation_space=observation_space,  device=cfg.device,
                        num_classes=cfg.env.num_classes,  eval=True, fixed_budget=True,
                        scale_reward=cfg.env.scale_reward,
                        sampled_indices=cfg.env.sampled_indices,
                        reward_mode=cfg.env.reward_mode, srange=cfg.env.srange,
                        delay_step=cfg.env.delay_step,
                        evaluation_only=True,
                        )

    return envs


def evaluate(ac, envs, writer=None, logger=None, best_metric={"accuracy": 0.0, "auc": 0.0}, snapshot_dir=None,
             num_line=1):

    global global_step
    avg_returns = deque(maxlen=10000)

    num_steps = len(envs.data_loader) * num_line

    envs.factory_reset()
    logging.debug(
        f"[evaluate] num_steps:{num_steps}, num_line:{num_line}, num_envs:{envs.num_envs}")

    obs = envs.reset()
    episode_reward = 0
    device = 'cuda'
    obs_mt = torch.tensor(envs.get_remain_epi_lines()).to(device)

    preds_logs = defaultdict(list)
    labels_logs = defaultdict(list)

    num_done = 0
    step = 0

    while True:
        step += 1

        with torch.no_grad(), eval_mode(ac):
            cur_mask = envs.get_cur_mask_2d(eval_mode=True)
            input_dict = {"kspace": obs, 'mt': obs_mt}
            action, _, _, _ = ac.get_action_and_value(
                input_dict, cur_mask, deterministic=True)

        with torch.no_grad():
            obs, reward, done, info = envs.step(
                action, get_metrics=True, get_reward=False)
            obs_mt = torch.tensor(envs.get_remain_epi_lines()).to(device)

        # reward is not used in testing phase
        episode_reward += reward
        if sum(done) == reward.shape[0]:

            avg_returns.append(episode_reward)
            episode_reward = 0

            for key in info.keys():

                preds_logs[key].append(info[key]['preds'])
                labels_logs[key].append(info[key]['labels'])

            num_done += 1
            logging.debug(f"final masks:{info[key]['final_masks']}")
            #print(f'doing logging at step:{step}, num_done:{num_done}')
            if num_done == len(envs.data_loader):
                break

    avg_returns = np.concatenate(avg_returns, axis=0)
    print(
        f'[EVAL] total_episode:{len(avg_returns)}')

    # compute the evalution metrics from meta data
    eval_metrics = {}
    for label_name in preds_logs.keys():

        preds_logs[label_name] = torch.cat(preds_logs[label_name], dim=0)
        labels_logs[label_name] = torch.cat(labels_logs[label_name], dim=0)

        eval_metrics[label_name] = evaluate_classifier(
            preds_logs[label_name], labels_logs[label_name], envs.num_classes)

    stats_all = {}
    stats_cur_line = {}
    for key in eval_metrics.keys():
        for key_ind in eval_metrics[key]:

            new_key = f"val_{key_ind}_{key}"
            stats_cur_line[new_key] = eval_metrics[key][key_ind]

    stats_cur_line['val_auc_mean'] = np.mean(
        [eval_metrics[key]["auc"] for key in eval_metrics.keys()])
    stats_all[num_line] = stats_cur_line

    print("stats_all:", stats_all)
    return stats_cur_line


if __name__ == "__main__":

    main()
