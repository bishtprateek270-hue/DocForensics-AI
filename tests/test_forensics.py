"""
DocForensics AI — Forensic Modules & Dual-Stream Architecture Tests (Phase 7)
Verifies:
- SRM filter residual output dimensions and finite values
- ELA and local noise extractor behavior
- RGB/Forensic alignment
- Dual-Stream forward pass and gradient flow
- Baseline and Gated attention fusion mechanisms
- CUDA execution & checkpoint save/load
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from src.forensics.srm_filters import SRMExtractor, get_srm_kernel_weights, get_srm_extractor
from src.forensics.ela_extractor import ELAExtractor, compute_ela_numpy
from src.forensics.noise_extractor import LocalNoiseExtractor
from src.models.dual_stream_forensics import DualStreamForensicNet, get_dual_stream_model


def test_srm_weights_shape_and_sum():
    """Verify SRM filter weights dimensions and zero-sum property."""
    weights = get_srm_kernel_weights()
    assert weights.shape == (3, 1, 5, 5)
    # Kernels should have approximately zero sum to eliminate DC component
    for i in range(3):
        kernel_sum = weights[i, 0].sum().item()
        assert abs(kernel_sum) < 1e-4, f"SRM Kernel {i} not zero-sum: {kernel_sum}"


def test_srm_extractor_forward():
    """Verify SRM residual extractor produces valid [B, 3, H, W] finite tensor."""
    extractor = SRMExtractor()
    dummy_rgb = torch.rand(2, 3, 512, 512)
    residuals = extractor(dummy_rgb)

    assert residuals.shape == (2, 3, 512, 512)
    assert torch.isfinite(residuals).all()
    # Check that residual is not all zeros
    assert residuals.abs().sum() > 0.0


def test_local_noise_extractor():
    """Verify LocalNoiseExtractor produces correct residual shape."""
    extractor = LocalNoiseExtractor(kernel_size=3)
    dummy_rgb = torch.rand(2, 3, 256, 256)
    noise = extractor(dummy_rgb)
    assert noise.shape == (2, 3, 256, 256)
    assert torch.isfinite(noise).all()


def test_compute_ela_numpy():
    """Verify NumPy/PIL ELA extractor output."""
    dummy_img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    ela_map = compute_ela_numpy(dummy_img, quality=90, scale=15.0)
    assert ela_map.shape == (256, 256, 3)
    assert ela_map.min() >= 0.0 and ela_map.max() <= 1.0


def test_dual_stream_forward_baseline():
    """Verify DualStreamForensicNet forward pass with baseline fusion."""
    model = DualStreamForensicNet(in_channels=3, num_classes=1, fusion_type="baseline")
    model.eval()

    dummy_input = torch.rand(2, 3, 512, 512)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 1, 512, 512)
    assert torch.isfinite(out).all()


def test_dual_stream_forward_gated():
    """Verify DualStreamForensicNet forward pass with gated attention fusion."""
    model = DualStreamForensicNet(in_channels=3, num_classes=1, fusion_type="gated")
    model.eval()

    dummy_input = torch.rand(2, 3, 512, 512)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 1, 512, 512)
    assert torch.isfinite(out).all()


def test_dual_stream_gradient_flow():
    """Verify backward pass and gradient flow through both RGB and Forensic streams."""
    model = DualStreamForensicNet(in_channels=3, num_classes=1, fusion_type="baseline")
    model.train()

    dummy_input = torch.rand(2, 3, 256, 256, requires_grad=True)
    target = torch.ones(2, 1, 256, 256)

    out = model(dummy_input)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, target)
    loss.backward()

    # Check that gradients exist for RGB backbone and forensic encoder
    assert model.rgb_layer4[0].conv1.weight.grad is not None
    assert model.forensic_encoder.stage4[0].weight.grad is not None
    assert model.fusion.proj[0].weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dual_stream_cuda_execution():
    """Verify DualStreamForensicNet executes properly on CUDA device."""
    device = torch.device("cuda:0")
    model = get_dual_stream_model(fusion_type="baseline", device=device)
    model.eval()

    dummy_input = torch.rand(1, 3, 512, 512, device=device)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.device.type == "cuda"
    assert out.shape == (1, 1, 512, 512)


def test_checkpoint_save_and_load():
    """Verify checkpoint serialization and state restoration."""
    model = get_dual_stream_model(fusion_type="baseline")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_dual_stream.pth"
        payload = {
            "epoch": 5,
            "model_name": "dual_stream",
            "model_state_dict": model.state_dict(),
            "val_dice": 0.65,
        }
        torch.save(payload, ckpt_path)

        loaded_model = get_dual_stream_model(fusion_type="baseline")
        ckpt = torch.load(ckpt_path, weights_only=False)
        loaded_model.load_state_dict(ckpt["model_state_dict"])

        # Compare outputs on same input
        dummy_in = torch.rand(1, 3, 256, 256)
        model.eval()
        loaded_model.eval()
        with torch.no_grad():
            out1 = model(dummy_in)
            out2 = loaded_model(dummy_in)
        assert torch.allclose(out1, out2, atol=1e-5)
