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


class Kspace_Net_MT(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation,
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
                 ):
        super().__init__()
        self.dataset = dataset
        self.image_shape = image_shape
        self.dropout = dropout
        self.pretrained = pretrained
        self.model_type = model_type
        self.act_dim = act_dim
        self.dropout_extra = dropout_extra
        self.aux_shape = 0 
        self.mt_shape = mt_shape  

        if (
            self.dataset == "cifar10"
        ):
            in_channels = 3
            out_channels = 3
        else:
            in_channels = 1
            out_channels = 1

        self.conv_kspace = FFTConv2d(
            in_channels, out_channels, kernel_size=5, stride=1, bias=False
        )
        elementwise_affine = True

        self.layernorm = nn.LayerNorm(
            elementwise_affine=elementwise_affine, normalized_shape=self.image_shape
        )


        if self.pretrained == "imagenet":
            pretrained = True
        else:
            pretrained = False
        weights = "IMAGENET1K_V1" if pretrained else None

        if self.model_type == "resnet50":
            resnet50 = models.resnet50(weights=weights)
            try:
                if self.dropout_extra:
                    resnet50.layer1.add_module(
                        "dropout", nn.Dropout2d(p=self.dropout)
                    )
                    resnet50.layer2.add_module(
                        "dropout", nn.Dropout2d(p=self.dropout)
                    )
                    resnet50.layer3.add_module(
                        "dropout", nn.Dropout2d(p=self.dropout)
                    )
            except:
                pass
            self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
            num_ftrs = 2048
        elif self.model_type == "resnet18":
            resnet18 = models.resnet18(weights=weights)
            self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
            num_ftrs = 512
        elif self.model_type == "alexnet":
            self.backbone = models.alexnet(weights=weights).features
            num_ftrs = 256
        elif self.model_type == "vit_b_16":
            vit = models.vit_b_16(weights=weights)
            self.backbone = nn.Sequential(*list(vit.children())[:-1])
            num_ftrs = 768
        elif self.model_type in [ 'resnet18_lv2']:
            resnet18 = models.resnet18(weights=weights)
            if self.dataset == "cifar10" or self.dataset == "cifar100":
                resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                resnet18.maxpool = nn.Identity()
            self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
            num_ftrs = 512

        else:
            raise NotImplementedError(
                f"Model type {self.model_type} not implemented"
            )

        self.dropout = nn.Dropout2d(p=self.dropout)

        
        if self.model_type in [ 'resnet18_lv2']:
            hidden_dim = 256
            self.trunk = nn.Sequential(nn.Linear(num_ftrs, feature_dim), nn.LayerNorm(
                feature_dim), nn.Tanh())
            trunk_dim = feature_dim + sum(self.mt_shape)
            policy_layer = []
            policy_layer += [nn.Linear(trunk_dim, hidden_dim),
                             nn.ReLU(inplace=True)]
            policy_layer += [nn.Linear(hidden_dim, self.act_dim)]
            self.out_layer = nn.Sequential(*policy_layer)
        else:
            self.out_layer = nn.Linear(num_ftrs, self.act_dim)
        if using_init:
            logging.debug(f"[ppo_core] using weight init")
            self.trunk.apply(weight_init)
            self.out_layer.apply(weight_init)

    def forward(self, input_dict):
        kspace = input_dict['kspace']
        mt = input_dict['mt']

        kspace = torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(
                kspace, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )
        kspace = torch.fft.fftn(kspace, dim=(-2, -1))

        if len(kspace.shape) == 3:
            kspace = kspace.unsqueeze(1)

        out_complex = self.conv_kspace(kspace)

        out_mag = out_complex.abs()

        out_mag = transforms.center_crop(out_mag, self.image_shape)
        out_mag = self.layernorm(out_mag)

        if out_mag.shape[1] == 1:
            out = out_mag.repeat(1, 3, 1, 1)
        else:
            out = out_mag
        out = self.backbone(out)

        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        
        
        h = self.trunk(out)

        mt_vec = torch.nn.functional.one_hot(mt, num_classes=self.mt_shape[0])

        if len(mt_vec.shape) == 1:
            out = torch.cat((h, mt_vec.repeat(out.shape[0], 1)), dim=-1)
        else:
            out = torch.cat((h, mt_vec), dim=-1)
        

        out = self.out_layer(out)

        return out


class Kspace_Net_Critic_MT(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation,
                 dataset,
                 image_shape,
                 dropout,
                 pretrained,
                 model_type,
                 dropout_extra,
                 aux_shape,
                 using_init,
                 feature_dim,
                 mt_shape,):
        super().__init__()


        self.dataset = dataset
        self.image_shape = image_shape
        self.dropout = dropout
        self.pretrained = pretrained
        self.model_type = model_type
        self.dropout_extra = dropout_extra
        self.aux_shape = 0 
        self.mt_shape = mt_shape

        if (
            self.dataset == "cifar10"
        ):
            in_channels = 3
            out_channels = 3
        else:
            in_channels = 1
            out_channels = 1

        self.conv_kspace = FFTConv2d(
            in_channels, out_channels, kernel_size=5, stride=1, bias=False
        )
        elementwise_affine = True
        self.layernorm = nn.LayerNorm(
            elementwise_affine=elementwise_affine, normalized_shape=self.image_shape
        )

        if self.pretrained == "imagenet":
            pretrained = True
        else:
            pretrained = False
        weights = "IMAGENET1K_V1" if pretrained else None

        if self.model_type == "resnet50":
            resnet50 = models.resnet50(weights=weights)
            try:
                if self.dropout_extra:
                    resnet50.layer1.add_module(
                        "dropout", nn.Dropout2d(p=self.dropout)
                    )
                    resnet50.layer2.add_module(
                        "dropout", nn.Dropout2d(p=self.dropout)
                    )
                    resnet50.layer3.add_module(
                        "dropout", nn.Dropout2d(p=self.dropout)
                    )
            except:
                pass
            self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
            num_ftrs = 2048
        elif self.model_type == "resnet18":
            resnet18 = models.resnet18(weights=weights)
            self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
            num_ftrs = 512
        elif self.model_type == "alexnet":
            self.backbone = models.alexnet(weights=weights).features
            num_ftrs = 256
        elif self.model_type == "vit_b_16":
            vit = models.vit_b_16(weights=weights)
            self.backbone = nn.Sequential(*list(vit.children())[:-1])
            num_ftrs = 768
        elif self.model_type in [ 'resnet18_lv2']:
            resnet18 = models.resnet18(weights=weights)
            if self.dataset == "cifar10" or self.dataset == "cifar100":
                resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                resnet18.maxpool = nn.Identity()
            self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
            num_ftrs = 512
        else:
            raise NotImplementedError(
                f"Model type {self.model_type} not implemented"
            )

        self.dropout = nn.Dropout2d(p=self.dropout)

        if self.model_type in [ 'resnet18_lv2']:
            hidden_dim = 256
            self.trunk = nn.Sequential(nn.Linear(num_ftrs, feature_dim), nn.LayerNorm(
                feature_dim), nn.Tanh())
            trunk_dim = feature_dim + sum(self.mt_shape)
            critic_layer = []
            critic_layer += [nn.Linear(trunk_dim, hidden_dim),
                             nn.ReLU(inplace=True)]
            critic_layer += [nn.Linear(hidden_dim,1)]
            self.out_layer = nn.Sequential(*critic_layer)
        else:
            self.out_layer = nn.Linear(num_ftrs, 1)

        if using_init:
            logging.debug(f"[ppo_core] using weight init")
            self.trunk.apply(weight_init)
            self.out_layer.apply(weight_init)

    def forward(self, input_dict):
        kspace = input_dict['kspace']
        mt = input_dict['mt']
        kspace = torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(
                kspace, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )
        kspace = torch.fft.fftn(kspace, dim=(-2, -1))

        if len(kspace.shape) == 3:
            kspace = kspace.unsqueeze(1)

        out_complex = self.conv_kspace(kspace)

        out_mag = out_complex.abs()

        out_mag = transforms.center_crop(out_mag, self.image_shape)
        out_mag = self.layernorm(out_mag)

        if out_mag.shape[1] == 1:
            out = out_mag.repeat(1, 3, 1, 1)
        else:
            out = out_mag
        out = self.backbone(out)

        
        out = out.view(out.size(0), -1)

        
        h = self.trunk(out)
        mt_vec = torch.nn.functional.one_hot(mt, num_classes=self.mt_shape[0])

        if len(mt_vec.shape) == 1:
            out = torch.cat((h, mt_vec.repeat(out.shape[0], 1)), dim=-1)
        else:
            out = torch.cat((h, mt_vec), dim=-1)


        out = self.out_layer(out)

        return out.squeeze()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

