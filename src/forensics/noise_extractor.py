"""
DocForensics AI — Local Noise & Median Residual Extractor (Phase 7)
Computes local noise patterns using median filter subtraction and high-frequency residuals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LocalNoiseExtractor(nn.Module):
    """
    Computes local noise residual by subtracting a 3x3 local average/blur from the original image.
    This reveals high-frequency local sensor noise and inpainting smoothing inconsistencies.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # 3x3 box blur kernel for local mean estimation
        weight = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        self.register_buffer("filter_weight", weight)
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        Returns: [B, C, H, W] noise residual map
        """
        b, c, h, w = x.shape
        # Apply filter per channel
        smoothed = []
        for ch in range(c):
            ch_in = x[:, ch : ch + 1, :, :]
            ch_smooth = F.conv2d(ch_in, self.filter_weight, padding=self.padding)
            smoothed.append(ch_smooth)
        smoothed = torch.cat(smoothed, dim=1)

        # Residual = Image - Smoothed
        residual = x - smoothed
        return residual
