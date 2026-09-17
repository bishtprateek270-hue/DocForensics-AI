"""
DocForensics AI — Frequency / DCT Feature Extractor (Phase 10)
Extracts block-based Discrete Cosine Transform (DCT) high-frequency residual features
to detect frequency domain discontinuities, compression grid misalignments, and resampling.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_dct_filter_bank(block_size: int = 8) -> torch.Tensor:
    """
    Constructs an orthogonal 2D DCT-II basis filter bank of size (block_size*block_size, 1, block_size, block_size).
    """
    basis = np.zeros((block_size, block_size, block_size, block_size), dtype=np.float32)
    for u in range(block_size):
        for v in range(block_size):
            alpha_u = 1.0 / math.sqrt(block_size) if u == 0 else math.sqrt(2.0 / block_size)
            alpha_v = 1.0 / math.sqrt(block_size) if v == 0 else math.sqrt(2.0 / block_size)
            for x in range(block_size):
                for y in range(block_size):
                    cos_u = math.cos((2 * x + 1) * u * math.pi / (2.0 * block_size))
                    cos_v = math.cos((2 * y + 1) * v * math.pi / (2.0 * block_size))
                    basis[u, v, x, y] = alpha_u * alpha_v * cos_u * cos_v

    # Reshape to (64, 1, 8, 8)
    filters = basis.reshape(block_size * block_size, 1, block_size, block_size)
    return torch.from_numpy(filters)


class FrequencyDCTExtractor(nn.Module):
    """
    PyTorch Differentiable / Deterministic Block-DCT High-Frequency Energy Extractor.
    Extracts 3-channel frequency residual maps:
    Channel 0: High-frequency AC energy
    Channel 1: Mid-frequency band energy
    Channel 2: Block boundary discontinuity magnitude
    """

    def __init__(self, block_size: int = 8, device: Optional[torch.device] = None):
        super().__init__()
        self.block_size = block_size
        filters = get_dct_filter_bank(block_size)
        self.register_buffer("filters", filters)  # (64, 1, 8, 8)

        # High-frequency and Mid-frequency index masks (u+v >= 4)
        hf_indices = []
        mf_indices = []
        for u in range(block_size):
            for v in range(block_size):
                idx = u * block_size + v
                if u + v >= 5:
                    hf_indices.append(idx)
                elif 2 <= u + v < 5:
                    mf_indices.append(idx)

        self.register_buffer("hf_indices", torch.tensor(hf_indices, dtype=torch.long))
        self.register_buffer("mf_indices", torch.tensor(mf_indices, dtype=torch.long))

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """
        Extracts multi-channel frequency representation from RGB tensor.
        
        Args:
            x_rgb: Input tensor (B, 3, H, W) normalized in range [0, 1] or standard ImageNet.
            
        Returns:
            freq_feats: (B, 3, H, W) frequency residual tensor.
        """
        # Convert RGB to Luminance Y
        # Y = 0.299*R + 0.587*G + 0.114*B
        if x_rgb.size(1) == 3:
            y_lum = (0.299 * x_rgb[:, 0:1] + 0.587 * x_rgb[:, 1:2] + 0.114 * x_rgb[:, 2:3])
        else:
            y_lum = x_rgb[:, 0:1]

        b, c, h, w = y_lum.shape
        pad_h = (self.block_size - (h % self.block_size)) % self.block_size
        pad_w = (self.block_size - (w % self.block_size)) % self.block_size

        if pad_h > 0 or pad_w > 0:
            y_padded = F.pad(y_lum, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            y_padded = y_lum

        # Apply 2D DCT convolution with stride = 1 for dense pixel-level frequency representation
        # Pad by (block_size // 2) to maintain exact spatial resolution
        pad_same = self.block_size // 2
        y_conv_pad = F.pad(y_padded, (pad_same, pad_same, pad_same, pad_same), mode="reflect")

        # Convolve with DCT basis
        dct_coeffs = F.conv2d(y_conv_pad, self.filters, stride=1)  # (B, 64, H, W)

        # 1. High-frequency energy
        hf_coeffs = torch.index_select(dct_coeffs, 1, self.hf_indices)
        hf_energy = torch.sqrt(torch.mean(hf_coeffs ** 2, dim=1, keepdim=True) + 1e-8)

        # 2. Mid-frequency energy
        mf_coeffs = torch.index_select(dct_coeffs, 1, self.mf_indices)
        mf_energy = torch.sqrt(torch.mean(mf_coeffs ** 2, dim=1, keepdim=True) + 1e-8)

        # 3. Local High-Pass Laplacian Residual
        laplacian_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32, device=x_rgb.device).unsqueeze(0).unsqueeze(0)
        lap_residual = torch.abs(F.conv2d(y_padded, laplacian_kernel, padding=1))

        # Crop back to original (H, W) if padded
        hf_energy = hf_energy[:, :, :h, :w]
        mf_energy = mf_energy[:, :, :h, :w]
        lap_residual = lap_residual[:, :, :h, :w]

        # Stack into 3-channel frequency representation
        freq_map = torch.cat([hf_energy, mf_energy, lap_residual], dim=1)

        # Robust min-max normalization per sample
        b_dim = freq_map.size(0)
        norm_maps = []
        for i in range(b_dim):
            sample = freq_map[i]
            s_min = sample.min()
            s_max = sample.max()
            s_norm = (sample - s_min) / (s_max - s_min + 1e-6)
            norm_maps.append(s_norm)

        return torch.stack(norm_maps, dim=0)
