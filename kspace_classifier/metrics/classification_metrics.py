from typing import Dict

import torchmetrics.classification
from sklearn import metrics
import torch
from torchmetrics import functional
from torchmetrics.classification import BinaryAUROC
from torchmetrics import AUROC
import numpy as np


def compute_auc(preds, labels, num_classes):
    metric = AUROC(task='binary')
    return metric(
        preds=preds,
        target=labels,
    )


def compute_auc_multiclass(preds, labels, num_classes):
    auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
    
    return auroc(preds=preds.detach().cpu(), target=labels.detach().cpu())


def compute_accuracy(preds, labels, num_classes):
    accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
    return accuracy(preds=preds.detach().cpu(), target=labels.detach().cpu())


def get_operating_point(preds, labels, operating_point=None, threshold=0.1):
    preds = preds.cpu()
    assert preds.shape[1] == 2
    preds_positive = preds[:, 1].numpy()

    labels = labels.cpu().numpy().astype(int)

    if operating_point is None:
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds_positive)

        operating_point = thresholds[fpr > 0.25][0]

        try:
            fnr = 1 - tpr
            operating_point = thresholds[fnr < threshold][0]
        except IndexError:
            operating_point = thresholds[fpr > 0.25][0]

    test_predictions = (preds_positive > operating_point).astype(int)

    sensitivity = metrics.recall_score(y_true=labels, y_pred=test_predictions)
    specificity = metrics.recall_score(
        y_true=np.abs(1 - labels), y_pred=np.abs(1 - test_predictions)
    )
    balanced_acc = metrics.balanced_accuracy_score(
        y_true=labels, y_pred=test_predictions
    )

    precision = metrics.precision_score(y_true=labels, y_pred=test_predictions)
    npv = metrics.precision_score(y_true=1.0 - labels, y_pred=1.0 - test_predictions)

    ret = dict(
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        balanced_accuracy=balanced_acc,
        operating_point=operating_point,
        npv=npv,
    )

    return ret


def evaluate_classifier(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    operating_point: float = None,
) -> Dict:
    batch_size = preds.shape[0]

    assert labels.shape == (batch_size,)
    assert preds.shape == (batch_size, num_classes)

    if num_classes == 2:                
        auc = compute_auc(preds=preds[:, 1], labels=labels, num_classes=num_classes).item()
        metrics_dict = get_operating_point(preds, labels, operating_point=operating_point)        
        accuracy = compute_accuracy(
                preds=preds, labels=labels, num_classes=num_classes
            )

        ret = dict(
        auc=auc,
        accuracy=accuracy,
        **metrics_dict,
        )

        return ret

    elif num_classes > 2:
        accuracy = compute_accuracy(
                preds=preds, labels=labels, num_classes=num_classes
            )

        balanced_acc = np.nan
        specificity = np.nan
        sensitivity = np.nan
        precision = np.nan
        operating_point = np.nan
        npv = np.nan
        auc = np.nan

        ret = dict(
        auc=auc,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        balanced_accuracy=balanced_acc,
        accuracy=accuracy,
        operating_point=operating_point,
        npv=npv,
        )

        return ret
                  


def classifier_metrics(
    val_preds: torch.Tensor,
    val_labels: torch.Tensor,
    test_preds: torch.Tensor,
    test_labels: torch.Tensor,
    threshold: float,
):
    _, _, _, _, operating_point, _ = get_operating_point(
        preds=val_preds, labels=val_labels, operating_point=None, threshold=threshold
    )

    return evaluate_classifier(
        preds=test_preds, labels=test_labels, operating_point=operating_point
    )
