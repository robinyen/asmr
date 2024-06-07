import torch
from torch._C import _ImperativeEngine
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchvision import models

from .fft_conv import FFTConv2d
from kspace_classifier.utils import transforms
from kspace_classifier.utils.utils import string_to_list
from kspace_classifier.classification_modules.classification_models.preact_resnet import PreActResNet18FFT, PreActResNet34FFT, PreActResNet50FFT, PreActResNet101FFT


class kSpaceNet(nn.Module):
    

    def __init__(
        self,
        config,
    ):
        """
               
        """
        super(kSpaceNet, self).__init__()
        self.num_classes = config.num_classes
        self.num_labels = config.num_labels
        self.label_names = string_to_list(config.label_names)
        self.config = config

        self.conv_kspace = FFTConv2d(
            self.config.in_channels, self.config.in_channels, kernel_size=5, stride=1, bias=False
        )
        self.layernorm = nn.LayerNorm(
            elementwise_affine=False, normalized_shape=self.config.image_shape
        )
        self.dropout = nn.Dropout(p=self.config.dropout)

        if self.config.pretrained == "imagenet":
            pretrained = True
        else:
            pretrained = False
        #print(f"Pretrained: {pretrained}, using imagenet weights")
        weights = "IMAGENET1K_V1" if pretrained else None

        if self.config.model_type == "resnet50":
            resnet50 = models.resnet50(weights=weights)
            if self.config.dataset ==  "cifar10":
                print('changing resnet18 for cifar dataset')
                resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                resnet50.maxpool = nn.Identity()

            self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
            num_ftrs = 2048
        elif self.config.model_type == "resnet18":
            resnet18 = models.resnet18(weights=weights)
            if self.config.dataset ==  "cifar10":
                print('changing resnet18 for cifar dataset')
                resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                resnet18.maxpool = nn.Identity()

            self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
            num_ftrs = 512
        elif self.config.model_type == "alexnet":
            self.backbone = models.alexnet(weights=weights).features
            num_ftrs = 256
        elif self.config.model_type == "vit_b_16":
            vit = models.vit_b_16(weights=weights)
            self.backbone = nn.Sequential(*list(vit.children())[:-1])
            num_ftrs = 768
        elif self.config.model_type == "preact_resnet18":
            preact_resnet = PreActResNet18FFT()
            self.backbone = preact_resnet
            num_ftrs = 512
        elif self.config.model_type == "preact_resnet34":
            preact_resnet = PreActResNet34FFT()
            self.backbone = preact_resnet
            num_ftrs = 512
        elif self.config.model_type == "preact_resnet50":
            preact_resnet = PreActResNet50FFT()
            self.backbone = preact_resnet
            num_ftrs = 2048
        elif self.config.model_type == "preact_resnet101":
            preact_resnet = PreActResNet101FFT()
            self.backbone = preact_resnet
            num_ftrs = 2048

        else:
            raise NotImplementedError(
                f"Model type {self.config.model_type} not implemented"
            )


        self.label_layers = (
            nn.ModuleList()
        )  ## List of linear layers for multiple output heads
        for _ in range(0, self.config.num_labels):
            self.label_layers.append(nn.Linear(num_ftrs, self.config.num_classes))

    def forward(self, kspace):
        """

        :param kspace: (complex tensor) of size(batch, channel, height, width), channel=1 for MRI data
        :return: (dict) {disease name: prob for each class}
        """
        

        kspace = torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(kspace, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )
        kspace = torch.fft.fftn(kspace, dim=(-2, -1))

        out_complex = self.conv_kspace(kspace)

        
        
        out_mag = out_complex.abs()
        
        out_mag = transforms.center_crop(out_mag, self.config.image_shape)

        if self.config.dataset != 'cifar10' :
            out_mag = self.layernorm(out_mag)

        
        if out_mag.shape[1] == 1:
            out = out_mag.repeat(1, 3, 1, 1)
        else:
            out = out_mag

        out = self.backbone(out)

        out = self.dropout(out)
        out = out.view(out.size(0), -1)

        out_dict = {}

        for i in range(0, len(self.label_layers)):
            out_dict[self.label_names[i]] = self.label_layers[i](out)

        return out_dict
