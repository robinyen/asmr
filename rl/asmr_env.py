import torch
import argparse

import argparse
import math
import os
from collections import namedtuple
from pathlib import Path


import torch
from torch.utils.data import Dataset, DataLoader
from kspace_classifier.metrics.classification_metrics import (
    compute_accuracy,
    evaluate_classifier,
)
import logging
from rl.utils import eval_mode
import numpy as np


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Asmr_Env:

    def __init__(self, reward_model, data_loader, observation_space=(3, 640, 400), device='cuda', num_classes=10,
                 k_fraction=0.1, eval=False, fixed_budget=False, sampled_indices=[33, 367],
                 scale_reward=True,
                 reward_mode=10,
                 srange=[5, 51],
                 delay_step=7,
                 get_label=False,
                 evaluation_only=False,
                 budget=40,
                 ):
        self.state = 0
        self.done = False

        self.data_loader = data_loader
        self.data_loader_iter = iter(self.data_loader)

        self.sampled_indices = sampled_indices

        self.observation_space = observation_space

        self.action_space = Namespace(
            n=sampled_indices[1]-sampled_indices[0]+1)
        self.act_dim = self.action_space.n

        self.num_envs = data_loader.batch_size
        self.device = device
        self.num_classes = num_classes
        self.k_fraction = k_fraction

        self.eval = eval

        self.fixed_budget = fixed_budget

        self.scale_reward = scale_reward

        self.reward_mode = reward_mode
        self.previous_raw_reward = 0.0
        self.delay_step = delay_step

        self.srange = srange

        self.set_reward_model(reward_model)

        self.get_label = get_label
        self.evaluation_only = evaluation_only
        self.budget = budget

    def factory_reset(self):
        self.data_loader_iter = iter(self.data_loader)

    def reset(self):

        try:
            batch = next(self.data_loader_iter)
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            batch = next(self.data_loader_iter)

        if self.eval and not self.fixed_budget:
            random_number = np.random.choice([12, 20, 32, 40, 50])
            self.set_budget(random_number)
            
        elif not self.fixed_budget:
            # randomly sampled number of lines for training in each episode
            random_number = np.random.randint(self.srange[0], self.srange[1])
            self.set_budget(random_number)
            
        else:
            logging.debug(f"budget:{self.budget}")

        batch["sc_kspace"] = batch["sc_kspace"].to(self.device)

        if len(batch["sc_kspace"].shape) == 3:

            batch["sc_kspace"] = batch["sc_kspace"].unsqueeze(1)
        self.state = batch

        kspace = batch["sc_kspace"]
        batch_size = kspace.shape[0]

        num_cols = kspace.shape[-1]
        self.num_cols = num_cols

        mask = torch.zeros(batch_size, 1, 1, num_cols)
        mask = mask.to(self.device)

        self.accumulated_mask = mask
        self.counter = 0

        logging.debug(f'[env] kspace:{kspace.shape}')
        logging.debug(f'[env] accumulated_mask:{self.accumulated_mask.shape}')

        s0 = kspace * self.accumulated_mask
        self.done = torch.zeros(batch_size)

        self.previous_raw_reward = 0.0

        return s0

    def get_allpass_mask(self):
        batch_size = self.state['sc_kspace'].shape[0]
        num_cols = self.state['sc_kspace'].shape[-1]
        mask = torch.zeros(batch_size, 1, 1, num_cols)
        mask = mask.to(self.device)
        mask = ~mask.bool()
        mask = mask.squeeze()
        return mask[:, self.sampled_indices[0]:self.sampled_indices[1]+1]

    def get_state(self):
        return self.state

    def get_reward(self, accumulated_mask):

        with torch.no_grad(), eval_mode(self.reward_model):

            log_prob_dict, logits = self.reward_model.log_prob(
                self.state, accumulated_mask)

        label = self.state['label']

        reward = 0

        if self.counter <= self.delay_step:
            # reveal reward after delay-step
            return torch.zeros(self.state['sc_kspace'].shape[0])

        if self.scale_reward:
            # normalize reward to improve stability
            for label_name in label:

                reward += torch.sign(log_prob_dict[label_name]) * (torch.sqrt(torch.abs(
                    log_prob_dict[label_name]) + 1)-1) + 0.001 * log_prob_dict[label_name]

            reward /= len(label)

        else:

            for label_name in label:

                reward += log_prob_dict[label_name]

            reward /= len(label)

        return reward

    def get_metrics(self, accumulated_mask, eval=False):
        with torch.no_grad(), eval_mode(self.reward_model):
            arr_preds = self.reward_model(self.state, accumulated_mask)
        label = self.state['label']
        eval_metrics = {}
        for label_name in label:

            gt = label[label_name].detach().cpu()
            pred = arr_preds[label_name].detach().cpu()

            eval_metrics[label_name] = {}

            if self.eval:
                eval_metrics[label_name]['preds'] = pred
                eval_metrics[label_name]['labels'] = gt.squeeze(1)
                eval_metrics[label_name]['final_masks'] = [
                    t.nonzero().cpu().numpy().squeeze() for t in accumulated_mask[:].squeeze()]

        return eval_metrics

    def set_reward_model(self, reward_model):

        reward_model.eval()
        self.reward_model = reward_model

    def get_cur_mask_2d(self, eval_mode=False):

        cur_mask = ~self.accumulated_mask.bool()
        cur_mask = cur_mask.squeeze()
        return cur_mask[:, self.sampled_indices[0]:self.sampled_indices[1]+1]

    def check_action_is_valid(self, action):

        tmp = action[1].cpu().numpy()
        cur_accu = torch.nonzero(self.accumulated_mask[1, 0, 0, :], as_tuple=True)[
            0].cpu().numpy()
        logging.info(f"tmp:{tmp}")
        logging.info(f"cur_accu:{cur_accu}")

    def get_remain_epi_lines(self):
        return self.budget - self.counter

    def get_cur_label(self):
        return self.state['label']

    def set_budget(self, num_lines):
        self.budget = num_lines

    def reach_budget(self,):
        return self.counter >= self.budget

    def get_accumulated_mask(self):
        # return the selected masks
        return self.accumulated_mask

    def step(self, action, get_metrics=False, get_reward=True):
        info = {}

        action = action+self.sampled_indices[0]
        action = torch.Tensor(action)

        action = torch.nn.functional.one_hot(action, self.num_cols)
        action = action.unsqueeze(1).unsqueeze(1)  # B, 1, 1, num_cols

        self.accumulated_mask = torch.max(self.accumulated_mask, action)

        # update the counter that track the number of action taken
        self.counter += 1

        state = self.get_state()
        observation = state['sc_kspace'] * self.accumulated_mask

        if get_reward:
            reward = self.get_reward(self.accumulated_mask)
        else:
            reward = torch.zeros(observation.shape[0])

        logging.debug(
            f'counter:{self.counter}, max len:{self.budget}')

        if self.reach_budget():
            done = torch.ones(reward.shape)
            if get_metrics:
                metrics = self.get_metrics(self.accumulated_mask)
                for key in metrics:
                    info[key] = metrics[key]
            observation = self.reset()
        else:
            done = torch.zeros(reward.shape)
        return observation, reward, done, info


def find_closest_greater_index(nums, value):
    greater = [num for num in nums if num <= value]
    if not greater:
        return min(nums)
    distances = [(num, abs(num - value)) for num in greater]
    min_dist = min(distances, key=lambda x: x[1])
    return min_dist[0]


def train_dataloader(config, ori_dataset):

    return DataLoader(
        ori_dataset,
        batch_size=config.batch_size,
        num_workers=3,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )


def main():
    pass


if __name__ == "__main__":
    main()
