"""
DocForensics AI — Segmentation Evaluation Metrics (Phase 5)
Computes pixel-level localization metrics:
- Dice Score (F1-Score)
- Intersection over Union (IoU / Jaccard Index)
- Precision (Positive Predictive Value)
- Recall (Sensitivity / True Positive Rate)
"""

import torch
import numpy as np
from typing import Dict, Any, Union


def compute_batch_metrics(
    logits_or_probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Computes average Dice, IoU, Precision, and Recall across a batch of masks.

    Args:
        logits_or_probs: Tensor of shape [B, 1, H, W] (raw logits or sigmoid probabilities)
        targets: Tensor of shape [B, 1, H, W] with ground truth in {0.0, 1.0}
        threshold: Binarization probability threshold
        smooth: Epsilon for numerical stability
    """
    with torch.no_grad():
        if logits_or_probs.min() < 0.0 or logits_or_probs.max() > 1.0:
            probs = torch.sigmoid(logits_or_probs)
        else:
            probs = logits_or_probs

        preds = (probs >= threshold).float()
        targets = targets.float()

        batch_size = targets.size(0)

        preds_flat = preds.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        dice_list = []
        iou_list = []
        prec_list = []
        rec_list = []

        for b in range(batch_size):
            p = preds_flat[b]
            t = targets_flat[b]

            tp = (p * t).sum().item()
            fp = (p * (1.0 - t)).sum().item()
            fn = ((1.0 - p) * t).sum().item()
            total_target_pos = t.sum().item()

            if total_target_pos == 0:
                # Authentic sample (all zeros in ground truth)
                if fp == 0:
                    # Correctly predicted clean
                    dice_list.append(1.0)
                    iou_list.append(1.0)
                    prec_list.append(1.0)
                    rec_list.append(1.0)
                else:
                    # False alarm on clean document
                    dice_list.append(0.0)
                    iou_list.append(0.0)
                    prec_list.append(0.0)
                    rec_list.append(1.0)
            else:
                # Tampered sample
                dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
                iou = (tp + smooth) / (tp + fp + fn + smooth)
                prec = (tp + smooth) / (tp + fp + smooth)
                rec = (tp + smooth) / (tp + fn + smooth)

                dice_list.append(dice)
                iou_list.append(iou)
                prec_list.append(prec)
                rec_list.append(rec)

        return {
            "dice": float(np.mean(dice_list)),
            "iou": float(np.mean(iou_list)),
            "precision": float(np.mean(prec_list)),
            "recall": float(np.mean(rec_list)),
        }


class MetricTracker:
    """Accumulates batch metrics over an entire epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.dice_sum = 0.0
        self.iou_sum = 0.0
        self.precision_sum = 0.0
        self.recall_sum = 0.0
        self.count = 0

    def update(self, loss: float, metrics: Dict[str, float], batch_size: int = 1):
        self.loss_sum += loss * batch_size
        self.dice_sum += metrics["dice"] * batch_size
        self.iou_sum += metrics["iou"] * batch_size
        self.precision_sum += metrics["precision"] * batch_size
        self.recall_sum += metrics["recall"] * batch_size
        self.count += batch_size

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
        return {
            "loss": self.loss_sum / self.count,
            "dice": self.dice_sum / self.count,
            "iou": self.iou_sum / self.count,
            "precision": self.precision_sum / self.count,
            "recall": self.recall_sum / self.count,
        }
