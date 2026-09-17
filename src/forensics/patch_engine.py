"""
DocForensics AI — High-Resolution Overlapping Patch & Multi-Scale Fusion Engine (Phase 10)
Provides:
1. Overlapping patch extraction from original-resolution document images
2. 2D Gaussian distance-weighted spatial blending to eliminate seam artifacts
3. Multi-scale probability fusion combining full-page (512x512) and high-res patch maps
"""

import math
from typing import List, Tuple, Optional
import numpy as np
import cv2
import torch


def create_2d_gaussian_window(patch_size: int = 512, sigma_scale: float = 0.25) -> np.ndarray:
    """Creates a 2D Gaussian weight window for smooth patch edge tapering."""
    sigma = patch_size * sigma_scale
    x = np.linspace(-patch_size / 2, patch_size / 2, patch_size)
    gauss_1d = np.exp(-0.5 * (x / sigma) ** 2)
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    # Normalize so peak is 1.0 and minimum non-zero
    gauss_2d = gauss_2d / np.max(gauss_2d)
    return np.clip(gauss_2d, 1e-4, 1.0).astype(np.float32)


class PatchEngine:
    """
    Manages high-resolution patch extraction, coordinate mapping, and seamless reconstruction.
    """

    def __init__(self, patch_size: int = 512, overlap: float = 0.25):
        self.patch_size = patch_size
        self.overlap = max(0.0, min(0.75, overlap))
        self.stride = max(32, int(patch_size * (1.0 - self.overlap)))
        self.window = create_2d_gaussian_window(patch_size)

    def extract_patches(
        self,
        image_rgb: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], Tuple[int, int]]:
        """
        Extracts overlapping patches from full-resolution document image.
        
        Args:
            image_rgb: Original image array (H, W, 3).
            
        Returns:
            patches: List of (patch_size, patch_size, 3) RGB images.
            coords: List of (x1, y1, x2, y2) in original document coordinates.
            orig_shape: (H, W) tuple.
        """
        h, w = image_rgb.shape[:2]
        patches: List[np.ndarray] = []
        coords: List[Tuple[int, int, int, int]] = []

        # If document is smaller than patch size in either dimension, pad it
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)

        if pad_h > 0 or pad_w > 0:
            padded_img = cv2.copyMakeBorder(
                image_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
            )
        else:
            padded_img = image_rgb

        h_pad, w_pad = padded_img.shape[:2]

        y_steps = list(range(0, h_pad - self.patch_size + 1, self.stride))
        if y_steps[-1] + self.patch_size < h_pad:
            y_steps.append(h_pad - self.patch_size)

        x_steps = list(range(0, w_pad - self.patch_size + 1, self.stride))
        if x_steps[-1] + self.patch_size < w_pad:
            x_steps.append(w_pad - self.patch_size)

        for y in y_steps:
            for x in x_steps:
                patch = padded_img[y:y + self.patch_size, x:x + self.patch_size].copy()
                patches.append(patch)
                coords.append((x, y, x + self.patch_size, y + self.patch_size))

        return patches, coords, (h, w)

    def reconstruct_probability_map(
        self,
        patch_predictions: List[np.ndarray],
        coords: List[Tuple[int, int, int, int]],
        orig_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Reconstructs full-resolution probability heatmap using 2D Gaussian weighted blending.
        
        Args:
            patch_predictions: List of (patch_size, patch_size) probability arrays [0.0 - 1.0].
            coords: List of (x1, y1, x2, y2) coordinates.
            orig_shape: Target document (H, W).
            
        Returns:
            Reconstructed probability map (H, W) in range [0.0, 1.0].
        """
        h_orig, w_orig = orig_shape
        max_x = max([c[2] for c in coords], default=w_orig)
        max_y = max([c[3] for c in coords], default=h_orig)

        acc_map = np.zeros((max_y, max_x), dtype=np.float32)
        weight_map = np.zeros((max_y, max_x), dtype=np.float32)

        for pred, (x1, y1, x2, y2) in zip(patch_predictions, coords):
            # Ensure prediction is 2D float32
            p_float = pred.astype(np.float32) if pred.dtype != np.float32 else pred
            if p_float.shape != (self.patch_size, self.patch_size):
                p_float = cv2.resize(p_float, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)

            acc_map[y1:y2, x1:x2] += (p_float * self.window)
            weight_map[y1:y2, x1:x2] += self.window

        # Avoid divide by zero
        weight_map = np.maximum(weight_map, 1e-6)
        reconstructed = acc_map / weight_map

        # Crop back to original dimensions
        reconstructed_orig = reconstructed[:h_orig, :w_orig]
        return np.clip(reconstructed_orig, 0.0, 1.0)


def fuse_multiscale_predictions(
    full_page_prob: np.ndarray,
    patch_prob: np.ndarray,
    alpha: float = 0.50
) -> np.ndarray:
    """
    Combines full-page contextual probability and high-resolution patch probability.
    
    Args:
        full_page_prob: Probability map from 512x512 full-page model (interpolated to orig shape).
        patch_prob: Probability map from high-resolution patch engine.
        alpha: Weight for full-page model [0.0 = patch only, 1.0 = full-page only].
        
    Returns:
        Fused probability map (H, W).
    """
    alpha = max(0.0, min(1.0, alpha))
    fused = (alpha * full_page_prob) + ((1.0 - alpha) * patch_prob)
    return np.clip(fused, 0.0, 1.0)
