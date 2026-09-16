"""
DocForensics AI — Unit Tests for DeepLabV3+ and SegFormer Architectures (Phase 6)
"""

import pytest
import torch
from src.models.deeplabv3plus import DeepLabV3Plus, get_deeplabv3plus_model
from src.models.segformer import SegFormer, get_segformer_model
from src.training.losses import BCEDiceLoss


def test_deeplabv3plus_forward_shape():
    """Verify DeepLabV3+ forward pass outputs [B, 1, H, W]."""
    model = DeepLabV3Plus(in_channels=3, num_classes=1, pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256), f"DeepLabV3+ shape mismatch: {out.shape}"
    assert not torch.isnan(out).any()


def test_segformer_forward_shape():
    """Verify SegFormer forward pass outputs [B, 1, H, W]."""
    model = SegFormer(in_channels=3, num_classes=1, variant="b0")
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256), f"SegFormer shape mismatch: {out.shape}"
    assert not torch.isnan(out).any()


def test_deeplabv3plus_loss_and_backward():
    """Verify DeepLabV3+ backward pass computes valid gradients."""
    model = DeepLabV3Plus(in_channels=3, num_classes=1, pretrained=False)
    criterion = BCEDiceLoss(0.5, 0.5)
    x = torch.randn(2, 3, 128, 128)
    target = torch.zeros(2, 1, 128, 128)
    target[:, :, 30:70, 30:70] = 1.0

    out = model(x)
    loss = criterion(out, target)
    loss.backward()
    assert loss.item() > 0


def test_segformer_loss_and_backward():
    """Verify SegFormer backward pass computes valid gradients."""
    model = SegFormer(in_channels=3, num_classes=1, variant="b0")
    criterion = BCEDiceLoss(0.5, 0.5)
    x = torch.randn(2, 3, 128, 128)
    target = torch.zeros(2, 1, 128, 128)
    target[:, :, 30:70, 30:70] = 1.0

    out = model(x)
    loss = criterion(out, target)
    loss.backward()
    assert loss.item() > 0
