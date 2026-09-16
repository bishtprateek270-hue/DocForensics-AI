"""
DocForensics AI — Spatial Rich Model (SRM) Filters (Phase 7)
PyTorch GPU-accelerated high-pass filter residual extraction for forensic analysis.
Extracts noise residual features to expose manipulation artifacts (copy-move, inpainting, splicing)
without modifying or destroying the original RGB image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


def get_srm_kernel_weights() -> torch.Tensor:
    """
    Returns 3 standardized 5x5 SRM / High-Pass filter kernels:
    1. KB (Ker-Böhme 1st-order edge / point predictor)
    2. KV (Ker-Voloshynovskiy 2nd-order edge / edge predictor)
    3. Laplacian 5x5 (Square high-pass curvature predictor)
    
    All kernels are zero-sum normalized to suppress image semantics and isolate high-frequency residuals.
    Output tensor shape: [3, 1, 5, 5]
    """
    # 1. 1st-order edge / point predictor filter (KB)
    kb = np.array([
        [ 0,  0,  0,  0,  0],
        [ 0, -1,  2, -1,  0],
        [ 0,  2, -4,  2,  0],
        [ 0, -1,  2, -1,  0],
        [ 0,  0,  0,  0,  0]
    ], dtype=np.float32) / 4.0

    # 2. 2nd-order edge predictor filter (KV)
    kv = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8,-12,  8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1]
    ], dtype=np.float32) / 12.0

    # 3. 5x5 Laplacian high-pass square filter
    lap = np.array([
        [ 0,  0, -1,  0,  0],
        [ 0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [ 0, -1, -2, -1,  0],
        [ 0,  0, -1,  0,  0]
    ], dtype=np.float32) / 16.0

    weights = np.stack([kb, kv, lap], axis=0)[:, np.newaxis, :, :]  # [3, 1, 5, 5]
    return torch.from_numpy(weights)


class SRMExtractor(nn.Module):
    """
    GPU-accelerated Spatial Rich Model (SRM) High-Pass Residual Extractor.
    Applies fixed non-trainable SRM filters to grayscale or RGB image tensors.
    
    Inputs:
        x: [B, 3, H, W] RGB tensor in range [0, 1] or normalized.
    Outputs:
        residuals: [B, 3, H, W] Forensic residual map normalized to [-1, 1] or [0, 1].
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, requires_grad: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Fixed SRM filter weights [3, 1, 5, 5]
        srm_weights = get_srm_kernel_weights()  # 3 kernels

        # Grayscale weights (standard Rec.601 luminance coefficients)
        self.register_buffer("luma_weights", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

        # Register SRM Conv layer
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )
        self.conv.weight = nn.Parameter(srm_weights, requires_grad=requires_grad)

        # Batch normalization for forensic stability
        self.norm = nn.BatchNorm2d(3, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 3-channel SRM residual feature map.
        x: [B, 3, H, W]
        """
        # Convert RGB to luminance [B, 1, H, W]
        if x.size(1) == 3:
            luma = (x * self.luma_weights).sum(dim=1, keepdim=True)
        else:
            luma = x[:, :1, :, :]

        # Extract 3 high-pass residuals [B, 3, H, W]
        residuals = self.conv(luma)

        # Truncate and normalize residuals (standard steganalysis residual clipping)
        residuals = torch.clamp(residuals, -3.0, 3.0)
        residuals = self.norm(residuals)
        return residuals


def get_srm_extractor(device: Optional[torch.device] = None) -> SRMExtractor:
    """Factory builder for SRMExtractor."""
    extractor = SRMExtractor()
    if device is not None:
        extractor = extractor.to(device)
    return extractor
