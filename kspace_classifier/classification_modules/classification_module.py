import argparse
from typing import Dict, Tuple, List, Union
import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn as nn
import wandb
from joblib import load
from kspace_classifier.classification_modules.focal_loss import FocalLoss

from kspace_classifier.metrics.classification_metrics import (
    compute_accuracy,
    evaluate_classifier,
)


from kspace_classifier.masks import (
    VariableDensityMask,
    #UniformRandomMask,
    MaskFunction,

)


class MRClassifier(pl.LightningModule):
    """
    Base class for classification module
    """

    def __init__(self, config: argparse.ArgumentParser):
        super().__init__()
        self.save_hyperparameters()
        self.config = config        

        # data and task type
        assert self.config.dataset in [
            "prostate_t2",
            "knee",
            "brain",   
        ]
        assert self.config.classifier_type in ["arms", "recon", "rss"]
        assert self.config.coil_type in ["sc"]

        
        self.mask_fn = self.get_mask_fn()

        self.arr_label_names = self.config.label_names.split(",")
        
        self.validation_step_outputs = []

        if self.config.loss_type in ["weighted_cross_entropy","focal_loss"] :
            #print("--" * 30, f"\n NOTE: Using weighted loss : {self.config.loss_type}", "\n", "--" * 30)
            try:
                self.loss_fn_weights = load(self.config.loss_fn_weights_filename)
            except (AttributeError, FileNotFoundError) as e:
                try:
                    self.loss_fn_weights = torch.tensor(self.config.class_weights)
                except AttributeError:
                    print('--'*30, 'Using default weights -> no weighting', '--'*30)
                    self.loss_fn_weights = torch.ones(self.config.num_classes, 2) * self.config.num_classes
                    

        self.criterion = {}
        for idx, label_name in enumerate(self.arr_label_names):
            if self.config.loss_type == "weighted_cross_entropy":
                self.criterion[label_name] = nn.CrossEntropyLoss(weight=self.loss_fn_weights[idx].float().cuda())
            elif self.config.loss_type == "cross_entropy":
                self.criterion[label_name] = nn.CrossEntropyLoss()
            elif self.config.loss_type == "focal_loss" :
                self.criterion[label_name] = FocalLoss(weight=self.loss_fn_weights[idx].float().to('cuda'))
            else:
                raise NotImplementedError(
                    f"Loss {self.config.loss_type} not implemented"
                )
        self.val_operating_point = None

        # To keep track of best metric across epochs
        self.best_metric = 0
        


    def on_fit_start(self) -> None:
        if self.mask_fn is not None:
            self.mask_fn.device = self.device

        
        self.criterion = {}
        for idx, label_name in enumerate(self.arr_label_names):
            if self.config.loss_type == "weighted_cross_entropy":
                self.criterion[label_name] = nn.CrossEntropyLoss(reduction='mean', weight=self.loss_fn_weights[idx].float().cuda())
            elif self.config.loss_type == "cross_entropy":
                self.criterion[label_name] = nn.CrossEntropyLoss(reduction='mean')
            elif self.config.loss_type == "focal_loss" :
                self.criterion[label_name] = FocalLoss(weight=self.loss_fn_weights[idx].float().to('cuda'))
            else:
                raise NotImplementedError(
                    f"Loss {self.config.loss_type} not implemented"
                )
        return super().on_fit_start()

    
    def get_mask_fn(self) -> MaskFunction:
        # mask arguments
        mask_configs = dict(
            target_shape=self.config.kspace_shape,
            k_fraction=self.config.k_fraction,
            sampled_indices=self.config.sampled_indices,
            center_fraction=self.config.center_fraction,
            batch_size=self.config.batch_size
            if self.training
            else self.config.val_batch_size,
            device=self.device,
        )

        if self.config.mask_type == "vds":
            mask_fn = VariableDensityMask(**mask_configs)
        elif self.config.mask_type == "none":
            mask_fn = None
        else:
            raise NotImplementedError(
                f"Mask type {self.config.mask_type} not implemented..."
            )
        return mask_fn

    def forward(self, batch):
        raise NotImplementedError

    def loss_fn(
        self, preds: torch.Tensor, labels: torch.Tensor, label_name: str
    ) -> torch.Tensor:
        batch_size = preds.shape[0]
        labels = labels.squeeze(1)

        assert labels.shape == (batch_size, )
        return self.criterion[label_name](preds, labels)

    
    def training_step(self, batch, batch_idx):
        arr_preds = self.forward(batch=batch)

        loss = 0
        for label_name in self.arr_label_names:
            label = batch["label"][label_name]
            pred = arr_preds[label_name]
            acc_condition = compute_accuracy(
                preds=pred, labels=label.squeeze(1), num_classes=self.config.num_classes
            )

            loss_condition = self.loss_fn(
                preds=pred, labels=label, label_name=label_name
            )

            loss = loss + loss_condition

            self.log(
                f"train_{label_name}_acc", acc_condition, prog_bar=True, sync_dist=True
            )

        loss_regularizer = torch.zeros_like(loss)

        self.log("train_loss", loss.detach(), prog_bar=True, sync_dist=False)
        
        for label_name in self.arr_label_names:
            arr_preds[label_name] = arr_preds[label_name].detach()
        return {"loss": loss, "preds": arr_preds, "labels": batch["label"]}

    def validation_step(self, batch, batch_idx):
        arr_preds = self.forward(batch=batch)
        loss = 0
        loss_dict = {}

        for label_name in self.arr_label_names:
            label = batch["label"][label_name]
            pred = arr_preds[label_name]
            
            loss_condition = self.loss_fn(
                preds=pred, labels=label, label_name=label_name
            ).item()
            
            loss = loss + loss_condition
            loss_dict[f"val_loss_{label_name}"] = loss_condition

            batch_size = len(pred)

        ret = {
                "loss": loss,
                **loss_dict,
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "labels": batch["label"],
                "preds": arr_preds,
            }

        self.validation_step_outputs.append(
            ret
        )

        return ret

    def collate_results(self, logs: Tuple) -> Dict:
        loss_list = []
        n_samples = 0

        labels_dict = {label_name: [] for label_name in self.arr_label_names}
        loss_dict = {label_name: [] for label_name in self.arr_label_names}
        preds_dict = {label_name: [] for label_name in self.arr_label_names}

        for log in logs:
            loss_list.append(log["loss"] * log["batch_size"])

            n_samples += log["batch_size"]
            labels = log["labels"]
            preds = log["preds"]

            for label_name in labels:
                loss_dict[label_name].append(log[f"val_loss_{label_name}"])
                preds_dict[label_name].append(preds[label_name])
                labels_dict[label_name].append(labels[label_name])

        for label_name in self.arr_label_names:
            labels_dict[label_name] = torch.cat(labels_dict[label_name], dim=0)
            preds_dict[label_name] = torch.cat(preds_dict[label_name], dim=0)
            loss_dict[label_name] = sum(loss_dict[label_name]) / n_samples

        loss = np.sum(loss_list) / n_samples

        return {
            "preds": preds_dict,
            "labels": labels_dict,
            "loss": loss,
            "loss_dict": loss_dict,
        }
    
    def on_validation_epoch_end(self):
        val_logs = self.collate_results(self.validation_step_outputs)

        preds = self.all_gather(val_logs["preds"])
        labels = self.all_gather(val_logs["labels"])
        loss = self.all_gather(val_logs["loss"])
        if self.trainer.is_global_zero :
            for label_name in self.arr_label_names :
                if len(preds[label_name].shape) == 3 :
                    preds[label_name] = preds[label_name].reshape(preds[label_name].shape[0]*preds[label_name].shape[1], preds[label_name].shape[2])
                    labels[label_name] = labels[label_name].reshape(labels[label_name].shape[0]*labels[label_name].shape[1])
                labels[label_name] = labels[label_name].squeeze(-1)
            loss = loss.mean()
            avg_bal_acc = 0.0
            avg_auc = 0.0
            self.val_operating_point = {}
            logging_settings = dict(prog_bar=True, sync_dist=False, rank_zero_only=True)
            eval_metrics = {}

            for label_name in self.arr_label_names:
                eval_metrics[label_name] = evaluate_classifier(
                    preds=preds[label_name],
                    labels=labels[label_name],
                    num_classes=self.config.num_classes,
                )

            # log metrics
            if self.config.dataset in ["cifar10"]:
                raise NotImplementedError(
                    f"Dataset {self.config.dataset} not implemented"
                )

            elif self.config.dataset in ["knee", "brain", "prostate_t2"]:
                for key in self.arr_label_names:
                    key_auc = eval_metrics[key]["auc"]
                    key_bal_acc = eval_metrics[key]["balanced_accuracy"]

                    self.val_operating_point[key] = eval_metrics[key]["operating_point"]
            

                    self.log(f"val_auc_{key}", key_auc, **logging_settings)
                    self.log(f"val_bal_acc_{key}", key_bal_acc, **logging_settings)

                    avg_auc += key_auc / len(self.arr_label_names)
                    avg_bal_acc += key_bal_acc / len(self.arr_label_names)

                if avg_auc > self.best_metric:
                    self.best_metric = avg_auc
            else:
                raise NotImplementedError(
                    f"Dataset {self.config.dataset} not implemented"
                )



            self.log(f"val_loss", loss, **logging_settings)
            self.log(f"val_auc_mean", avg_auc, **logging_settings)

        self.log(f"best_metric", self.best_metric, prog_bar=True, sync_dist=True, reduce_fx=torch.sum)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch=batch, batch_idx=batch_idx)

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def configure_optimizers(self):
        
        if self.config.mask_type in ["concrete"]:
            parameters_mask = list(self.mask_fn.parameters())
            parameters_task = list(self.model.parameters())
            parameters = [
                {"params": parameters_task, "lr": self.config.lr},
                {"params": parameters_mask, "lr": self.config.lr_mask_params},
            ]
        else:
            parameters = list(self.model.parameters())

        # configure optimizer

        if self.config.optimizer == "adamw":
            optimizer = optim.AdamW(
                parameters,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adam":
            optimizer = optim.Adam(
                parameters,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "sgd":
            optimizer = optim.SGD(
                parameters,
                lr=self.config.lr,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(f"Optim {self.config.optimizer} not implemented")

        # configure learning rate scheduler
        if self.config.lr_scheduler:
            try :
                if self.config.lr_scheduler_type == 'multistep' :
                    scheduler = optim.lr_scheduler.MultiStepLR(
                        optimizer=optimizer,
                        milestones=[5,12,18,22,25],
                        gamma=self.config.lr_gamma,
                    )
                elif self.config.lr_scheduler_type == 'plateau' :
                    print("Using plateau scheduler")
                    scheduler = {
                        "scheduler" : optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer = optimizer,
                            mode = 'max',
                            factor=self.config.lr_gamma,
                            patience=5
                        ),
                        "monitor" : "best_metric"
                    }
                elif self.config.lr_scheduler_type == 'onecycle' :
                    print("Using one cycle lr")
                    steps_per_epoch = 45000 // self.config.batch_size
                    scheduler = {
                        "scheduler": optim.lr_scheduler.OneCycleLR(
                            optimizer,
                            0.1,
                            epochs=self.trainer.max_epochs,
                            steps_per_epoch=steps_per_epoch,
                        ),
                        "interval": "step",
                    }

                else :
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer=optimizer,
                        step_size=self.config.lr_step_size,
                        gamma=self.config.lr_gamma,
                    )
            except :
                print("Could not find scheduler_type, using default step scheduler")
                scheduler = optim.lr_scheduler.StepLR(
                        optimizer=optimizer,
                        step_size=self.config.lr_step_size,
                        gamma=self.config.lr_gamma,
                    )
                        
            return [optimizer], [scheduler]
        else:
            return [optimizer]
