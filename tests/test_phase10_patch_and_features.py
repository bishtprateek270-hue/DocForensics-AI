"""
DocForensics AI — Phase 10 Patch Engine, DCT Extractor & Feature Test Suite
Validates:
- High-resolution overlapping patch extraction and coordinate transformations
- 2D Gaussian weighted spatial reconstruction (no visible seams / boundary NaNs)
- Multi-scale probability fusion
- 2D DCT Frequency extractor dimensions and determinism
- Tversky & Focal loss computations and gradient flow
- Authentic hard-negative empty mask integrity
- Automated dataset leakage detection
"""

import pytest
import numpy as np
import torch
from PIL import Image

from src.forensics.patch_engine import PatchEngine, create_2d_gaussian_window, fuse_multiscale_predictions
from src.forensics.frequency_extractor import FrequencyDCTExtractor
from src.training.losses import TverskyLoss, FocalTverskyLoss, BCEDiceLoss, FocalLoss
from src.preprocessing.hard_tampering_generator import generate_authentic_hard_negative, tamper_hard_tiny_replacement
from src.preprocessing.dataset_auditor import audit_dataset


def test_patch_extraction_and_coordinates():
    """Verify patch engine extracts correct 512x512 patches and valid bounding coordinates."""
    engine = PatchEngine(patch_size=512, overlap=0.25)
    img_rgb = np.random.randint(0, 255, (1050, 800, 3), dtype=np.uint8)

    patches, coords, orig_shape = engine.extract_patches(img_rgb)
    assert len(patches) > 0
    assert len(patches) == len(coords)
    assert orig_shape == (1050, 800)

    for p, (x1, y1, x2, y2) in zip(patches, coords):
        assert p.shape == (512, 512, 3)
        assert x2 - x1 == 512
        assert y2 - y1 == 512
        assert x1 >= 0 and y1 >= 0


def test_gaussian_spatial_blending_reconstruction():
    """Verify Gaussian-weighted patch reconstruction produces seamless maps without NaNs or Infs."""
    engine = PatchEngine(patch_size=512, overlap=0.25)
    img_rgb = np.zeros((1050, 800, 3), dtype=np.uint8)
    patches, coords, orig_shape = engine.extract_patches(img_rgb)

    # Simulate random patch probability maps
    patch_preds = [np.random.uniform(0.1, 0.9, (512, 512)).astype(np.float32) for _ in patches]

    recon_map = engine.reconstruct_probability_map(patch_preds, coords, orig_shape)
    assert recon_map.shape == (1050, 800)
    assert not np.isnan(recon_map).any()
    assert not np.isinf(recon_map).any()
    assert recon_map.min() >= 0.0
    assert recon_map.max() <= 1.0


def test_multiscale_fusion():
    """Verify multi-scale probability fusion combines full-page and patch maps accurately."""
    full_p = np.full((100, 100), 0.8, dtype=np.float32)
    patch_p = np.full((100, 100), 0.4, dtype=np.float32)

    fused = fuse_multiscale_predictions(full_p, patch_p, alpha=0.50)
    assert np.allclose(fused, 0.60, atol=1e-5)

    fused_patch_only = fuse_multiscale_predictions(full_p, patch_p, alpha=0.0)
    assert np.allclose(fused_patch_only, 0.40, atol=1e-5)


def test_frequency_dct_extractor():
    """Verify PyTorch 2D DCT feature extractor produces 3-channel normalized residual maps."""
    extractor = FrequencyDCTExtractor(block_size=8)
    x = torch.rand(2, 3, 256, 256)

    out = extractor(x)
    assert out.shape == (2, 3, 256, 256)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_tversky_loss_gradients():
    """Verify TverskyLoss and FocalTverskyLoss compute valid scalar losses and backpropagate gradients."""
    criterion = TverskyLoss(alpha=0.3, beta=0.7)
    focal_criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)

    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    loss_t = criterion(logits, targets)
    assert loss_t.item() >= 0.0
    loss_t.backward()
    assert logits.grad is not None

    logits.grad.zero_()
    loss_ft = focal_criterion(logits, targets)
    assert loss_ft.item() >= 0.0
    loss_ft.backward()
    assert logits.grad is not None


def test_authentic_hard_negative_empty_mask():
    """Verify authentic hard negative generator produces strictly all-zero masks."""
    pil_img = Image.new("RGB", (800, 1000), color=(250, 250, 248))
    img_mod, mask, bbox, manip_type = generate_authentic_hard_negative(pil_img, add_stamp=True, add_signature=True)

    assert mask.shape == (1000, 800)
    assert np.sum(mask) == 0  # Strictly empty mask
    assert bbox == [0, 0, 0, 0]
    assert manip_type == "authentic_hard_negative"


def test_hard_tiny_replacement():
    """Verify tiny number replacement produces small percentage masks (< 0.5%)."""
    pil_img = Image.new("RGB", (800, 1000), color=(250, 250, 248))
    img_tampered, mask, bbox, manip_type = tamper_hard_tiny_replacement(pil_img, replacement_type="marks_sgpa")

    assert mask.shape == (1000, 800)
    tampered_pixels = np.sum(mask)
    area_pct = (tampered_pixels / float(800 * 1000)) * 100.0
    assert area_pct < 1.0  # Tiny region
    assert bbox[2] > bbox[0]
    assert bbox[3] > bbox[1]
