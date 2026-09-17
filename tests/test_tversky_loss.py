"""Mathematical validation and behavior tests for Tversky and Focal Tversky losses."""

import pytest
import torch

from src.training.losses import (
    FocalDiceLoss,
    FocalLoss,
    FocalTverskyLoss,
    TverskyLoss,
)


def test_tversky_fn_penalty_greater_than_fp_penalty():
    """Prove that with alpha=0.3 and beta=0.7, False Negatives yield higher loss than False Positives."""
    alpha = 0.3
    beta = 0.7
    loss_fn = TverskyLoss(alpha=alpha, beta=beta)

    # Base setup: 10 positive pixels, 90 background pixels
    # Case 1: Pure False Negative (Predict background when target is foreground)
    # Target has 10 positive pixels, prediction is all zeros (logits = -10.0) -> FN = 10, FP = 0, TP = 0
    logits_fn = torch.full((1, 1, 10, 10), -10.0)
    targets = torch.zeros((1, 1, 10, 10))
    targets[0, 0, :2, :5] = 1.0  # 10 positive pixels
    loss_with_fn = loss_fn(logits_fn, targets).item()

    # Case 2: Pure False Positive of identical pixel magnitude (10 FP pixels, TP = 0, FN = 0)
    # Target is all background, prediction is 10 foreground pixels
    logits_fp = torch.full((1, 1, 10, 10), -10.0)
    logits_fp[0, 0, :2, :5] = 10.0  # 10 positive predictions
    targets_all_bg = torch.zeros((1, 1, 10, 10))
    loss_with_fp = loss_fn(logits_fp, targets_all_bg).item()

    # When TP = 0:
    # Tversky index for FN = eps / (beta * 10 + eps) -> index ~ eps / 7.0 -> Loss ~ 1.0 - eps/7.0
    # Tversky index for FP = eps / (alpha * 10 + eps) -> index ~ eps / 3.0 -> Loss ~ 1.0 - eps/3.0
    # With partial TP (e.g. TP=5):
    logits_tp_fn = torch.full((1, 1, 10, 10), -10.0)
    logits_tp_fn[0, 0, 0, :5] = 10.0  # 5 TP, 5 FN
    loss_partial_fn = loss_fn(logits_tp_fn, targets).item()

    logits_tp_fp = torch.full((1, 1, 10, 10), -10.0)
    logits_tp_fp[0, 0, 0, :5] = 10.0  # 5 TP
    logits_tp_fp[0, 0, 5, :5] = 10.0  # 5 FP
    targets_partial = torch.zeros((1, 1, 10, 10))
    targets_partial[0, 0, 0, :5] = 1.0  # 5 targets -> TP=5, FP=5, FN=0
    loss_partial_fp = loss_fn(logits_tp_fp, targets_partial).item()

    print(f"Loss with 5 TP + 5 FN (beta=0.7): {loss_partial_fn:.6f}")
    print(f"Loss with 5 TP + 5 FP (alpha=0.3): {loss_partial_fp:.6f}")

    # Mathematical Proof: Loss(FN) > Loss(FP) because beta (0.7) > alpha (0.3)
    # T_idx(FN) = 5 / (5 + 0.7*5) = 5 / 8.5 = 0.5882 -> Loss = 0.4118
    # T_idx(FP) = 5 / (5 + 0.3*5) = 5 / 6.5 = 0.7692 -> Loss = 0.2308
    assert loss_partial_fn > loss_partial_fp, (
        f"Expected FN loss ({loss_partial_fn}) > FP loss ({loss_partial_fp}) "
        f"for beta ({beta}) > alpha ({alpha})"
    )


def test_focal_tversky_loss():
    """Verify Focal Tversky loss forward pass and gradient backpropagation."""
    loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
    logits = torch.randn((2, 1, 64, 64), requires_grad=True)
    targets = torch.zeros((2, 1, 64, 64))
    targets[:, :, 10:20, 10:20] = 1.0

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_dice_loss():
    """Verify Focal Dice loss forward pass and gradient backpropagation."""
    loss_fn = FocalDiceLoss(focal_weight=0.5, dice_weight=0.5)
    logits = torch.randn((2, 1, 64, 64), requires_grad=True)
    targets = torch.zeros((2, 1, 64, 64))
    targets[:, :, 10:20, 10:20] = 1.0

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
