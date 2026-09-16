"""
DocForensics AI — Unit Tests for U-Net Model, Loss Functions, and Metrics (Phase 5)
"""

import pytest
import torch
import numpy as np

from src.models.unet import UNet, get_unet_model
from src.training.losses import DiceLoss, BCEDiceLoss, FocalLoss
from src.evaluation.metrics import compute_batch_metrics, MetricTracker


def test_unet_architecture_output_shape():
    """Verify UNet forward pass produces expected output shape [B, 1, H, W]."""
    model = UNet(in_channels=3, num_classes=1, base_channels=16)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)

    assert out.shape == (2, 1, 256, 256), f"Output shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN values"


def test_dice_loss_zero_and_perfect():
    """Verify DiceLoss returns ~0.0 for perfect overlap and >0.0 for mismatch."""
    criterion = DiceLoss()
    logits_perfect = torch.ones(2, 1, 64, 64) * 10.0  # Sigmoid -> ~1.0
    targets_perfect = torch.ones(2, 1, 64, 64)

    loss_perfect = criterion(logits_perfect, targets_perfect)
    assert loss_perfect.item() < 0.05, f"Perfect loss should be near 0, got {loss_perfect.item()}"

    logits_opposite = -torch.ones(2, 1, 64, 64) * 10.0  # Sigmoid -> ~0.0
    loss_mismatch = criterion(logits_opposite, targets_perfect)
    assert loss_mismatch.item() > 0.90, f"Mismatch loss should be near 1, got {loss_mismatch.item()}"


def test_bce_dice_loss():
    """Verify combined BCEDiceLoss calculates finite scalar gradients."""
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    loss = criterion(logits, targets)
    assert loss.dim() == 0, "Loss must be scalar"
    loss.backward()
    assert logits.grad is not None, "Gradient was not computed"


def test_compute_batch_metrics():
    """Verify compute_batch_metrics correctly computes Dice, IoU, Precision, Recall."""
    # Create synthetic perfect match
    logits = torch.ones(2, 1, 32, 32) * 5.0
    targets = torch.ones(2, 1, 32, 32)

    metrics = compute_batch_metrics(logits, targets)
    assert abs(metrics["dice"] - 1.0) < 1e-4
    assert abs(metrics["iou"] - 1.0) < 1e-4
    assert abs(metrics["precision"] - 1.0) < 1e-4
    assert abs(metrics["recall"] - 1.0) < 1e-4


def test_metric_tracker_accumulation():
    """Verify MetricTracker accumulates over multiple batches accurately."""
    tracker = MetricTracker()
    m1 = {"dice": 0.8, "iou": 0.7, "precision": 0.9, "recall": 0.85}
    m2 = {"dice": 0.6, "iou": 0.5, "precision": 0.7, "recall": 0.75}

    tracker.update(0.5, m1, batch_size=2)
    tracker.update(0.3, m2, batch_size=2)

    res = tracker.compute()
    assert abs(res["dice"] - 0.7) < 1e-4
    assert abs(res["loss"] - 0.4) < 1e-4
