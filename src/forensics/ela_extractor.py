"""
DocForensics AI — Error Level Analysis (ELA) Extractor (Phase 7)
Computes Error Level Analysis (JPEG compression difference) to detect compression inconsistencies.
Can be computed on PIL/NumPy images or PyTorch batches.
"""

import io
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from typing import Optional, Union


def compute_ela_numpy(
    image_np: np.ndarray,
    quality: int = 90,
    scale: float = 15.0,
) -> np.ndarray:
    """
    Computes ELA for a single HxWxC uint8 or float32 image.
    
    Args:
        image_np: [H, W, 3] image in range [0, 255] uint8 or [0.0, 1.0] float32.
        quality: JPEG compression quality factor (typically 75-90).
        scale: Multiplier to enhance small difference artifacts.
        
    Returns:
        ela_map: [H, W, 3] float32 in range [0.0, 1.0].
    """
    if image_np.dtype != np.uint8:
        if image_np.max() <= 1.0:
            image_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image_np = image_np.clip(0, 255).astype(np.uint8)

    im = Image.fromarray(image_np)
    buffer = io.BytesIO()
    im.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    diff = ImageChops.difference(im, recompressed)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    if max_diff == 0:
        max_diff = 1

    scale_factor = scale
    diff = ImageEnhance.Brightness(diff).enhance(scale_factor)
    ela_np = np.array(diff, dtype=np.float32) / 255.0
    return ela_np


class ELAExtractor(nn.Module):
    """
    Error Level Analysis Extractor Module.
    Accepts normalized RGB tensors and computes ELA maps.
    """

    def __init__(self, quality: int = 90, scale: float = 15.0):
        super().__init__()
        self.quality = quality
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] tensor on GPU/CPU.
        Returns: [B, 3, H, W] ELA difference map.
        """
        device = x.device
        b, c, h, w = x.shape
        ela_list = []

        # Detach to CPU for PIL JPEG encoding
        x_cpu = x.detach().cpu()
        for i in range(b):
            img = x_cpu[i].permute(1, 2, 0).numpy()  # [H, W, 3]
            # Unnormalize if standardized
            if img.min() < 0.0:
                # Approximate unnormalization (assuming ImageNet or [-1, 1])
                img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            ela_np = compute_ela_numpy(img, quality=self.quality, scale=self.scale)
            ela_t = torch.from_numpy(ela_np).permute(2, 0, 1)  # [3, H, W]
            ela_list.append(ela_t)

        ela_tensor = torch.stack(ela_list, dim=0).to(device)
        return ela_tensor
