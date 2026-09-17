"""
DocForensics AI — Advanced Segmentation Loss Functions (Phase 10)
Handles severe foreground/background pixel imbalance and tiny region localization:
- Soft Dice Loss (Direct region overlap optimization)
- BCE + Dice Loss (Balanced boundary gradient and overlap score)
- Focal Loss (Down-weights easy background pixels)
- Tversky Loss (Asymmetric penalty prioritizing recall on tiny manipulated regions)
- Focal Tversky Loss (Non-linear focusing on hard, tiny tampering boundaries)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Loss = 1 - (2 * (P * Y) + smooth) / (P^2 + Y^2 + smooth)
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = (probs_flat * probs_flat).sum(dim=1) + (targets_flat * targets_flat).sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCEWithLogitsLoss + DiceLoss for robust document tampering localization.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, pos_weight: Optional[float] = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        pos_w_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for focusing on hard foreground tampering boundaries.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss
        return loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss for optimizing recall on tiny manipulated regions by penalizing false negatives.
    TI = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    Loss = 1 - TI
    """

    def __init__(self, alpha: float = 0.30, beta: float = 0.70, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - targets_flat)).sum(dim=1)
        fn = ((1.0 - probs_flat) * targets_flat).sum(dim=1)

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky_index.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss: Non-linear focusing on small, highly imbalanced tampering targets.
    Loss = (1 - TI) ** (1 / gamma)
    """

    def __init__(self, alpha: float = 0.30, beta: float = 0.70, gamma: float = 1.33, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - targets_flat)).sum(dim=1)
        fn = ((1.0 - probs_flat) * targets_flat).sum(dim=1)

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow(1.0 - tversky_index, 1.0 / self.gamma)
        return focal_tversky.mean()


def get_loss_function(name: str = "bce_dice", **kwargs) -> nn.Module:
    """Factory helper to load loss function by name."""
    name_clean = name.lower().strip()
    if name_clean == "bce_dice":
        return BCEDiceLoss(**kwargs)
    elif name_clean == "focal":
        return FocalLoss(**kwargs)
    elif name_clean == "tversky":
        return TverskyLoss(**kwargs)
    elif name_clean == "focal_tversky":
        return FocalTverskyLoss(**kwargs)
    elif name_clean == "dice":
        return DiceLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function '{name}'. Supported: bce_dice, focal, tversky, focal_tversky, dice")
