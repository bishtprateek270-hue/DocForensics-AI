"""
DocForensics AI — Segmentation Loss Functions (Phase 5)
Handles severe foreground/background pixel imbalance in document tampering tasks:
- Soft Dice Loss (Direct region overlap optimization)
- BCE + Dice Loss (Balanced boundary gradient and overlap score)
- Focal Loss (Down-weights easy background pixels)
"""

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
        
        # Flatten spatial dimensions: [B, -1]
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

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, pos_weight: float = None):
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
