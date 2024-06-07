import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from kspace_classifier.utils.utils import string_to_list
from torchvision import transforms

class PreTrainedResNet(nn.Module):
    def __init__(self, config):
        super(PreTrainedResNet, self).__init__()
        self.num_classes = config.num_classes        
        self.num_labels = config.num_labels
        self.label_names = string_to_list(config.label_names)        
        self.config = config

        if self.config.pretrained == "imagenet":
            pretrained = True
        else:
            pretrained = False
        #print(f"Pretrained: {pretrained}, using imagenet weights")            
        weights = "IMAGENET1K_V1" if pretrained else None


        if self.config.model_type == 'resnet50':            
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
        else:
            raise NotImplementedError(f"Model type {self.config.slice_model_type} not implemented")
        
        self.dropout = nn.Dropout2d(p=self.config.dropout)
    
        self.label_layers = nn.ModuleList() ## List of linear layers for multiple output heads
        for _ in range(0, self.config.num_labels):
            self.label_layers.append(nn.Linear(num_ftrs, self.config.num_classes))
        
    def forward(self, x):        

        if x.shape[1] == 1:
            batch_size = x.shape[0]
            out = x.repeat(1, 3, 1, 1).float()        
        elif self.config.dataset == "cifar10" or x.shape[1] == 3:
            batch_size = x.shape[0]
            out = x.float()   
        
        out = self.backbone(out)
        out = self.dropout(out)

        out = out.view(batch_size, -1)
        
        out_dict = {}
        
        for i in range(0, len(self.label_layers)):
            out_dict[self.label_names[i]] = self.label_layers[i](out)                
        
        return out_dict

            