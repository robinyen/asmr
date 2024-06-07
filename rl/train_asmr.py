# main file

from kspace_classifier.metrics.classification_metrics import (
    compute_accuracy,
    evaluate_classifier,
)
from pathlib import Path
import logging
import time
from logger import Logger
import os
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from rl.utils import eval_mode, set_seed_everywhere
from collections import deque, defaultdict
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from kspace_classifier.classification_modules import ARMSClassifier
from rl.asmr_env import Asmr_Env
from kspace_classifier.data_modules import MRDataModule
import hydra
import numpy as np
import torch

import warnings
warnings.filterwarnings('ignore')


def get_optimizer(parameters, optimizer='adamw', lr=1e-3, weight_decay=5e-4, lr_momentum=0.9):
    if optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=lr_momentum,
            weight_decay=weight_decay,
        )
    return optimizer


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_data(config):

    return MRDataModule(config=config, dev_mode=config.dev_mode)


@hydra.main(version_base=None, config_path='cfgs', config_name='train_asmr')
def main(cfg):

    print(f"Current working directory : {os.getcwd()}")
    print(f"hydra path:{HydraConfig.get().run.dir}")
    run_dir = Path(HydraConfig.get().run.dir)
    cfg.snapshot_dir = (run_dir / Path("models")).resolve()
    cfg.env.batch_size = cfg.num_envs

    data_module = get_data(cfg.env)

    train_loader = data_module.train_dataloader()

    val_loader = data_module.val_dataloader()

    logging.debug(
        f"-----length train_loader:{len(train_loader)} val_loader:{len(val_loader)}-----")

    set_seed_everywhere(cfg.seed)

    num_envs = cfg.num_envs
    num_steps = cfg.num_steps
    ppo_batch_size = int(num_envs * num_steps)
    cfg.ppo_batch_size = ppo_batch_size

    import wandb

    wandb.init(
        project=cfg.project_name,
        sync_tensorboard=True,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=run_dir,

    )

    envs = prepare_train_envs(cfg, train_loader)
    eval_envs = prepare_evaluate_envs(cfg, val_loader)

    ac = hydra.utils.instantiate(cfg.model, action_space=envs.action_space)

    ac.to(cfg.device)

    writer = SummaryWriter(f"{run_dir}/tb")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()])),
    )

    work_dir = Path.cwd()

    global global_step
    global_step = 0

    load_snapshot(ac, cfg.load_from_snapshot_base_dir)

    train(cfg, ac, envs, eval_envs, writer)


def train(cfg, ac, envs, eval_envs, writer):
    parameters = filter(lambda p: p.requires_grad, ac.parameters())
    optimizer = get_optimizer(parameters, optimizer=cfg.optim.name,
                              lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    total_params = sum(p.numel() for p in ac.parameters() if p.requires_grad)
    

    num_envs = cfg.num_envs
    num_steps = cfg.num_steps
    ppo_batch_size = cfg.ppo_batch_size
    minibatch_size = int(cfg.ppo_batch_size // cfg.num_minibatches)

    global global_step
    best_metric = {"accuracy": 0.0, 'auc': 0.0}

    device = torch.device(cfg.device)

    kmask_shape = (envs.act_dim,)

    print(f"num_steps:{num_steps}, num_envs:{num_envs}")
    obs = torch.zeros((num_steps, num_envs) +
                      envs.observation_space, dtype=torch.complex64).to(device)
    obs_kmask = torch.zeros((num_steps, num_envs) +
                            kmask_shape, dtype=torch.bool).to(device)

    actions = torch.zeros((num_steps, num_envs)).to(device)
    obs_mt = torch.zeros((num_steps, num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    print(f"obs shape:{obs.shape}")

    episode_return = 0
    episode_returns = deque(maxlen=200)

    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = cfg.total_timesteps // ppo_batch_size

    next_obs_mt = torch.tensor(
        envs.get_remain_epi_lines(), dtype=torch.long).to(device)

    for update in range(1, num_updates + 1):

        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done
            obs_mt[step] = next_obs_mt
            logging.debug(f'step{step}, obs_mt:{next_obs_mt}')

            with torch.no_grad():

                cur_mask = envs.get_cur_mask_2d()

                input_dict = {"kspace": next_obs,  "mt": next_obs_mt}
                action, logprob, _, value = ac.get_action_and_value(
                    input_dict, cur_mask)
                values[step] = value

            actions[step] = action
            logprobs[step] = logprob

            obs_kmask[step] = cur_mask.to(device)
            next_obs, reward, done, info = envs.step(action, get_metrics=True)

            next_obs_mt = torch.tensor(envs.get_remain_epi_lines()).to(device)
            episode_return += reward
            logging.debug(
                f'update:{update}, step:{step}, reward:{reward[0]}, done :{done[0]}')

            rewards[step].copy_(reward).to(device).view(-1)

            next_obs, next_done = next_obs.to(
                device), torch.Tensor(done).to(device)

            if sum(done) == num_envs:
                episode_returns.append(episode_return)
                episode_return = 0

        with torch.no_grad():

            next_value = ac.get_value(
                {'kspace': next_obs,  "mt": next_obs_mt}).reshape(1, -1)

            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.gamma * \
                    nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * \
                    cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.observation_space)
        b_obs_kmask = obs_kmask.reshape((-1,) + kmask_shape)

        b_obs_mt = obs_mt.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(ppo_batch_size)
        clipfracs = []
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, ppo_batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                input_dict = {
                    "kspace": b_obs[mb_inds],  "mt": b_obs_mt[mb_inds]}
                _, newlogprob, entropy, newvalue = ac.get_action_and_value(
                    input_dict, b_obs_kmask[mb_inds], a=b_actions.long()[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   cfg.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * \
                    torch.clamp(
                        ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * \
                    entropy_loss + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    ac.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl is not None:
                if approx_kl > cfg.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate",
                          optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl",
                          old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance",
                          explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step /
                          (time.time() - start_time)), global_step)

        if update % cfg.log_interval == 0 and len(episode_returns) > 1:
            print(
                f'[TRAIN] Update: {update}, FPS: {int(global_step / (time.time() - start_time))}, Mean Ret: {np.mean(episode_returns)}')
            writer.add_scalar("charts/train_mean_return",
                              np.mean(episode_returns), global_step)

            episode_return = 0
            episode_returns = deque(maxlen=200)
            avg_infos_accuracy = defaultdict(list)

        if update % cfg.eval_interval == 0:
            evaluate(ac, eval_envs, writer, best_metric=best_metric,
                     snapshot_dir=cfg.snapshot_dir)


def save_snapshot(model, snapshot_dir, save_last=False, cur_is_best=True):

    if snapshot_dir is None:
        return

    logging.info(f"[Train.py] save at snapshot_dir:{snapshot_dir}")
    snapshot_dir.mkdir(exist_ok=True, parents=True)
    if cur_is_best:
        snapshot = snapshot_dir / f'best_model.pt'
        with open(snapshot, 'wb') as f:
            torch.save(model.state_dict(), f)
    if save_last:
        snapshot = snapshot_dir / f'last_model.pt'
        with open(snapshot, 'wb') as f:
            torch.save(model.state_dict(), f)


def load_snapshot(model, load_from_snapshot_base_dir):

    snapshot_base_dir = Path(load_from_snapshot_base_dir)
    snapshot = snapshot_base_dir / f'best_model.pt'
    if not snapshot.exists():
        return None
    logging.info(f"[Train.py] load snapshot:{snapshot}")
    model.load_state_dict(torch.load(snapshot))


def prepare_train_envs(cfg, train_loader):

    observation_space = tuple(cfg.env.observation_space)
    print(cfg.env)

    if cfg.env_version == "Asmr_Env":
        

        reward_model = ARMSClassifier.load_from_checkpoint(
            cfg.env.reward_model_ckpt,  map_location=cfg.device)

        envs = Asmr_Env(reward_model, train_loader,
                        observation_space=observation_space, device=cfg.device,
                        num_classes=cfg.env.num_classes,  fixed_budget=cfg.env.train_fixed_budget,
                        scale_reward=cfg.env.scale_reward,
                        sampled_indices=cfg.env.sampled_indices,
                        reward_mode=cfg.env.reward_mode, srange=cfg.env.srange,
                        delay_step=cfg.env.delay_step,
                        )

    else:
        raise ValueError("env_version not supported")
    return envs


def prepare_evaluate_envs(cfg, val_loader):

    observation_space = cfg.env.observation_space

    if cfg.env_version == "Asmr_Env":
        

        reward_model = ARMSClassifier.load_from_checkpoint(
            cfg.env.reward_model_ckpt,  map_location=cfg.device)

        envs = Asmr_Env(reward_model, val_loader,
                        observation_space=observation_space,  device=cfg.device,
                        num_classes=cfg.env.num_classes,  eval=True, fixed_budget=cfg.env.eval_fixed_budget,
                        scale_reward=cfg.env.scale_reward,
                        sampled_indices=cfg.env.sampled_indices,
                        reward_mode=cfg.env.reward_mode, srange=cfg.env.srange,
                        delay_step=cfg.env.delay_step,
                        )

    else:
        raise ValueError("env_version not supported")

    return envs


def evaluate(ac, envs, writer=None, logger=None, best_metric={"auc": 0.0}, snapshot_dir=None, device='cuda'):

    global global_step
    avg_returns = deque(maxlen=10000)
    avg_infos_accuracy = defaultdict(list)

    envs.factory_reset()

    obs = envs.reset()
    episode_reward = 0

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

        episode_reward += reward

        if sum(done) == reward.shape[0]:

            avg_returns.append(episode_reward)
            episode_reward = 0

            for key in info.keys():

                preds_logs[key].append(info[key]['preds'])
                labels_logs[key].append(info[key]['labels'])

            num_done += 1
            logging.debug(f"final masks:{info[key]['final_masks']}")

            if num_done == len(envs.data_loader):
                break

    avg_returns = np.concatenate(avg_returns, axis=0)
    print(
        f'[EVAL] total_episode:{len(avg_returns)}, avg_returns:{np.mean(avg_returns)}')

    eval_metrics = {}
    for label_name in preds_logs.keys():

        preds_logs[label_name] = torch.cat(preds_logs[label_name], dim=0)
        labels_logs[label_name] = torch.cat(labels_logs[label_name], dim=0)

        eval_metrics[label_name] = evaluate_classifier(
            preds_logs[label_name], labels_logs[label_name], envs.num_classes)

    avg_auc = 0.0
    eval_buffer_for_log = {}
    if writer is not None:
        writer.add_scalar("charts/eval_mean_return",
                          np.mean(avg_returns), global_step)

        for key in preds_logs.keys():

            writer.add_scalar(f"charts/eval_mean_accuracy_{key}",
                              np.mean(avg_infos_accuracy[key]), global_step)
            writer.add_scalar(
                f"charts/eval_auc_{key}", eval_metrics[key]["auc"], global_step)
            writer.add_scalar(
                f"charts/eval_accuracy_{key}", eval_metrics[key]["accuracy"], global_step)
            writer.add_scalar(
                f"charts/balanced_accuracy_{key}", eval_metrics[key]["balanced_accuracy"], global_step)

            key_score_auc = eval_metrics[key]["auc"]

            avg_auc += key_score_auc / len(preds_logs.keys())

            logging.info(
                f'[EVAL] {key} auc:{eval_metrics[key]["auc"]}, accuracy:{eval_metrics[key]["accuracy"]}, balanced_accuracy:{eval_metrics[key]["balanced_accuracy"]}')

        writer.add_scalar(f"charts/eval_mean_auc", avg_auc, global_step)

        if avg_auc >= best_metric['auc']:
            best_metric['auc'] = avg_auc
            eval_buffer_for_log = eval_metrics
            logging.info(
                f"[EVAL] globalstep:{global_step} best_metric:{best_metric}, best_buffer:{eval_buffer_for_log}")
            save_snapshot(ac, snapshot_dir)
        writer.add_scalar(f"charts/best_metric",
                          best_metric['auc'], global_step)
        save_snapshot(ac, snapshot_dir, cur_is_best=False, save_last=True)


if __name__ == "__main__":

    main()
