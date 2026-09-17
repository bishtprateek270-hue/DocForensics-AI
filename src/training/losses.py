"""Loss Functions for Document Tampering Segmentation with Extreme Target Sparsity.

Implements and mathematically validates:
1. DiceLoss
2. BCEDiceLoss
3. FocalLoss
4. FocalDiceLoss
5. TverskyLoss (configurable alpha=FP penalty, beta=FN penalty)
6. FocalTverskyLoss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice Loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice



class TverskyLoss(nn.Module):
    r"""Tversky Loss for extreme class imbalance and tiny-target segmentation.

    Mathematical Formula:
        T(\alpha, \beta) = \frac{TP + \epsilon}{TP + \alpha \cdot FP + \beta \cdot FN + \epsilon}
        Loss = 1 - T(\alpha, \beta)

    Coefficients:
        - \alpha (fp_weight): Weight assigned to False Positives.
        - \beta (fn_weight): Weight assigned to False Negatives.

    Tuning for High Recall (Tiny Text Tampering):
        Setting \beta > \alpha (e.g. \beta = 0.7, \alpha = 0.3) penalizes FALSE NEGATIVES
        more severely than False Positives, forcing the network to detect small,
        low-contrast character manipulations.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
            raise ValueError(f"Alpha ({alpha}) and Beta ({beta}) must be in [0, 1].")
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Tversky Loss.

        Args:
            logits: [B, 1, H, W] unnormalized model outputs.
            targets: [B, 1, H, W] ground-truth binary targets in {0.0, 1.0}.
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # True Positives, False Positives, False Negatives
        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs_flat) * targets_flat).sum()

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky_index


class FocalTverskyLoss(nn.Module):
    r"""Focal Tversky Loss.

    Formula:
        Loss = (1 - T(\alpha, \beta))^{\gamma}

    Where \gamma (\gamma \ge 1.0) suppresses loss from well-classified easy background examples
    and concentrates gradients on difficult, small tampered text regions.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tversky_loss = self.tversky(logits, targets)
        return torch.pow(tversky_loss, self.gamma)


class FocalLoss(nn.Module):
    """Binary Focal Loss with sigmoid activation."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)
        return (focal_weight * bce_loss).mean()


class FocalDiceLoss(nn.Module):
    """Combined Focal Loss and Soft Dice Loss."""

    def __init__(
        self, focal_weight: float = 0.5, dice_weight: float = 0.5, gamma: float = 2.0
    ) -> None:
        super().__init__()
        self.focal = FocalLoss(gamma=gamma)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.smooth = 1e-6

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_l = self.focal(logits, targets)

        probs = torch.sigmoid(logits).view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs * targets_flat).sum()
        cardinality = probs.sum() + targets_flat.sum()
        dice_l = 1.0 - (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )

        return self.focal_weight * focal_l + self.dice_weight * dice_l
