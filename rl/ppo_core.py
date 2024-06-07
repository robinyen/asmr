import numpy as np
import scipy.signal


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import logging


from kspace_classifier.classification_modules.arms_models.fft_conv import FFTConv2d
from torchvision import models
from kspace_classifier.utils import transforms
from rl.nn_utils import MaskedCategorical
from rl.ppo_core_net_mt import Kspace_Net_MT, Kspace_Net_Critic_MT


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class KspaceMaskedCategoricalActor_MT(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, dataset, image_shape, dropout,
                 pretrained, model_type, dropout_extra, aux_shape, using_init, feature_dim,
                 mt_shape):
        super().__init__()
        if aux_shape is not None and mt_shape is not None:
            pass
        elif aux_shape is None and mt_shape is not None:
            self.logits_net = Kspace_Net_MT(obs_dim, act_dim, hidden_sizes, activation,
                                            dataset,
                                            image_shape,
                                            dropout,
                                            pretrained,
                                            model_type,
                                            dropout_extra,
                                            aux_shape,
                                            using_init,
                                            feature_dim,
                                            mt_shape,
                                            )
        else:
            raise NotImplementedError("Not supported")

    def _distribution(self, obs, mask):
        logits = self.logits_net(obs)
        return MaskedCategorical(logits=logits, mask=mask)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class KspaceMaskedActorCritic_MT(nn.Module):

    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(64, 64),
                 activation=nn.Tanh,
                 dataset='cifar10',
                 image_shape=[32, 32],
                 dropout=0.0,
                 pretrained=False,
                 model_type='resnet50',
                 model_type_critic='resnet50',
                 dropout_extra=False,
                 aux_shape=None,
                 using_init=False,
                 feature_dim=50,
                 mt_shape=256,
                 ):

        super().__init__()
        self._cur_mask = None

        obs_dim = observation_space

        self.pi = KspaceMaskedCategoricalActor_MT(obs_dim, action_space.n, hidden_sizes, activation,
                                                  dataset,
                                                  image_shape,
                                                  dropout,
                                                  pretrained,
                                                  model_type,
                                                  dropout_extra,
                                                  aux_shape,
                                                  using_init,
                                                  feature_dim,
                                                  mt_shape)
        

        if aux_shape is not None and mt_shape is not None:
            pass
        elif aux_shape is None and mt_shape is not None:
            
            self.v = Kspace_Net_Critic_MT(obs_dim, hidden_sizes, activation,
                                          dataset,
                                          image_shape,
                                          dropout,
                                          pretrained,
                                          model_type_critic,
                                          dropout_extra,
                                          aux_shape,
                                          using_init,
                                          feature_dim,
                                          mt_shape)
        else:
            raise NotImplementedError("Not supported")
        

    def get_action_and_value(self, obs, mask, a=None, deterministic=False):

        pi = self.pi._distribution(obs, mask)
        if a is None:
            if deterministic:

                a = pi.mode
            else:
                a = pi.sample()
        logp_a = self.pi._log_prob_from_distribution(pi, a)
        v = self.v(obs)

        dist_entropy = pi.entropy()

        return a, logp_a, dist_entropy, v

    def get_action_and_value_aux(self, obs, mask, a=None, deterministic=False, obs_aux=None):

        pi = self.pi._distribution(obs, mask)
        if a is None:
            if deterministic:
                a = pi.mode
            else:
                a = pi.sample()
        logp_a = self.pi._log_prob_from_distribution(pi, a)
        v = self.v(obs)
        dist_entropy = pi.entropy()

        return a, logp_a, dist_entropy, v

    def act(self, obs):
        return self.step(obs)[0]

    def get_value(self, obs):
        return self.v(obs)
