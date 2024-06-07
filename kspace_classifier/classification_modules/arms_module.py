import time

import argparse
import torch.nn as nn
import torch




from .arms_models import kSpaceNet
from .classification_module import MRClassifier
from torch.distributions import Categorical
from copy import deepcopy

from kspace_classifier.utils.utils import transfer_bal_dataloader_to_device

def get_model(config) -> nn.Module:
    if config.model_type in ["resnet50", 'resnet18', "alexnet", "vit_b_16", "preact_resnet18", "preact_resnet34", "preact_resnet50", "preact_resnet101"]:
        model = kSpaceNet(config=config)
    else:
        raise NotImplementedError(f"Model type {config.model_type} not implemented")
    return model


class ARMSClassifier(MRClassifier):
    """
    This class is direct classifier, where it takes input kspace and perform classification directly,
    without reconstruction step
    """

    def __init__(
        self, config: argparse.ArgumentParser, fixed_mask: torch.Tensor = None
    ):
        super().__init__(config)
        self.save_hyperparameters()

        # get model depending on data and model type
        self.model = get_model(config=deepcopy(config))
        self.fixed_mask = fixed_mask
        self.coil_type = config.coil_type

        self.val_operating_point = None
    
    def mask_kspace(self, kspace, mask_fn, fixed_mask):
        batch_size = kspace.shape[0]
        if len(kspace.shape) < 4 :
            # add channels dimension
            kspace = kspace.unsqueeze(1)

        if self.fixed_mask is None and fixed_mask is None:
            if self.config.greedy_search is True:
                arr_masks = []
                for _ in range(batch_size):
                    
                    mask_fn.k_fraction = torch.rand(1).item() * (self.config.max_sampling_rate - self.config.min_sampling_rate) + self.config.min_sampling_rate
                    mask_fn.center_fraction = mask_fn.k_fraction / 2                
                    
                    mask = mask_fn.get_mask()                                        
                    arr_masks.append(mask)
                
                arr_masks = torch.stack(arr_masks).squeeze(-1)

                assert arr_masks.shape == (batch_size, self.config.kspace_shape[1]), arr_masks.shape
                assert kspace.shape == (batch_size, self.config.in_channels, *self.config.kspace_shape), kspace.shape
                
                arr_masks = arr_masks.view(batch_size, 1, 1, self.config.kspace_shape[1]).repeat(1, self.config.in_channels, 1, 1)
                masked_kspace = arr_masks * kspace                

                assert masked_kspace.shape == kspace.shape
            else:
                masked_kspace = mask_fn(kspace)
        elif fixed_mask is not None and self.fixed_mask is None:
            masked_kspace = kspace * fixed_mask
        elif self.fixed_mask is not None and fixed_mask is None:
            masked_kspace = kspace * self.fixed_mask
        else:
            raise ValueError("Cannot set both fixed_mask and self.fixed_mask")

        return masked_kspace

    def forward(self, batch, fixed_mask=None):
        if self.coil_type == "sc":
            kspace = batch["sc_kspace"]
        else:
            raise NotImplementedError(f"Coil type {self.coil_type} not implemented")
        
        masked_kspace = self.mask_kspace(kspace, self.mask_fn, fixed_mask)

        return self.model(masked_kspace)

    def log_prob(self, batch, fixed_mask):
        assert self.eval, 'model not in eval mode'
        logits = self.forward(batch=batch, fixed_mask=fixed_mask)        
        batch_size = batch['sc_kspace'].shape[0]

        log_prob_dict = {label_name: torch.zeros(batch_size) for label_name in self.arr_label_names}

        for label_name in logits:
            with torch.no_grad():
                log_probs = nn.functional.log_softmax(logits[label_name], dim=-1)            
            y_idx = batch["label"][label_name]
            
            for i in range(batch_size):            
                log_prob_dict[label_name][i] = log_probs[i][y_idx[i].item()]       
            
            assert log_prob_dict[label_name].shape == (batch["sc_kspace"].shape[0],)
        return log_prob_dict, logits
    
    def entropy(self, batch, fixed_mask):
        assert self.eval, 'model not in eval mode'
        logits = self.forward(batch=batch, fixed_mask=fixed_mask)
        batch_size = batch['sc_kspace'].shape[0]

        entropy_dict = {label_name: torch.zeros(batch_size) for label_name in self.arr_label_names}

        for label_name in logits:
            with torch.no_grad():
                probs = nn.functional.softmax(logits[label_name], dim=-1)                         
            
            for i in range(batch_size): 
                # Need to minimize entropy  (vs maximizing MI)          
                entropy_dict[label_name][i] = -1 * Categorical(probs=probs[i]).entropy()

            assert entropy_dict[label_name].shape == (batch["sc_kspace"].shape[0],)

        return entropy_dict, logits

    # estimate value function for a given mask
    def value_function(self, dataloader, fixed_mask):
        assert self.eval, 'model not in eval mode'
        if self.config.val_fn_type == "MI":
            val_fn = self.log_prob
        elif self.config.val_fn_type == "entropy":
            val_fn = self.entropy
        else:
            raise NotImplementedError(
                f"Value function {self.config.val_fn} not implemented"
            )

        value = {label_name: 0 for label_name in self.arr_label_names}
        n_samples = 0

        for _, batch in enumerate(dataloader):
            with torch.no_grad():
                val_batch, logits_batch = val_fn(batch=batch, fixed_mask=fixed_mask)
            n_samples += batch["sc_kspace"].shape[0]

            for label_name in self.arr_label_names:
                value[label_name] += val_batch[label_name].sum()

        
        for label_name in self.arr_label_names:
            value[label_name] = value[label_name] / n_samples

        return value, logits_batch
    
    def value_function_per_batch(self, batch, fixed_mask):
        assert self.eval, "model not in eval mode"
        if self.config.val_fn_type == "MI":
            val_fn = self.log_prob
        elif self.config.val_fn_type == "entropy":
            val_fn = self.entropy
        else:
            raise NotImplementedError(
                f"Value function {self.config.val_fn} not implemented"
            )

        with torch.no_grad():
            val_batch, logits_batch = val_fn(batch=batch, fixed_mask=fixed_mask)

        
        n_samples = batch["sc_kspace"].shape[0]
        for label_name in self.arr_label_names:
            val_batch[label_name] = val_batch[label_name] / n_samples

        return val_batch, logits_batch

    def forward_active(self, batch, mask_sequence, threshold=0.):
        
        kspace = batch["sc_kspace"]

        n_masks = len(mask_sequence)
        batch_size = kspace.shape[0]        

        prev_val = {label: torch.zeros(batch_size, device=self.device) for label in self.arr_label_names}
        final_preds = {label: torch.zeros(batch_size, device=self.device) for label in self.arr_label_names}
        stopping_number = torch.zeros(batch_size, device=self.device)

        stop = [False for _ in range(batch_size)]
        
        curr_val  = {label: torch.zeros(batch_size, device=self.device) for label in self.arr_label_names}        

        masked_kspace = kspace * mask_sequence[9]
        with torch.no_grad():        
            logits = self.model(masked_kspace)            

        log_prob = {}
        preds = {}
        
        for label_name in self.arr_label_names:
            log_prob[label_name] = nn.functional.log_softmax(logits[label_name], dim=-1)                        
            preds[label_name] = torch.argmax(logits[label_name], dim=-1)
            
        for sample_idx in range(batch_size):
            for label in self.arr_label_names:
                label_pred = preds[label_name][sample_idx]
                prev_val[label_name][sample_idx] = log_prob[label_name][sample_idx][label_pred]

        for mask_idx in range(10, n_masks):
            masked_kspace = kspace * mask_sequence[mask_idx]

            with torch.no_grad():        
                logits = self.model(masked_kspace)            

            log_prob = {}
            preds = {}
            
            for label_name in self.arr_label_names:
                log_prob[label_name] = nn.functional.log_softmax(logits[label_name], dim=-1)                        
                preds[label_name] = torch.argmax(logits[label_name], dim=-1)
                
            for sample_idx in range(batch_size):
                diff = 0

                for label in self.arr_label_names:
                    label_pred = preds[label_name][sample_idx]
                    curr_val[label_name][sample_idx] = log_prob[label_name][sample_idx][label_pred]
                
                    diff += torch.abs(curr_val[label][sample_idx] - prev_val[label][sample_idx])                                                     
                
                if diff <= threshold and mask_idx > 0 and not stop[sample_idx]:
                    for label_name in self.arr_label_names:                                              
                        final_preds[label_name][sample_idx] = preds[label_name][sample_idx]                        
                        
                    stopping_number[sample_idx] = mask_idx       

                    stop[sample_idx] = True               

            prev_val = deepcopy(curr_val)

        return final_preds, stopping_number
