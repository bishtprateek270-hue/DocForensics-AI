"""Positive-Aware and Hard-Negative Patch Sampler for DocTamper.

During TRAINING only:
- Ground-truth masks are used to extract high-resolution crops around tampered characters (50%).
- Hard negative patches containing legitimate text (clean numbers/dates/tables) with zero tampering are extracted (30%).
- Full-page global context is preserved (20%).

During VALIDATION / INFERENCE:
- Standard full document canonical view is evaluated without ground truth guidance.
"""

from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.doctamper_loader import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DocTamperTransform,
)

logger = logging.getLogger(__name__)


class PositiveAwarePatchTransform:
    """Synchronous crop, augmentation, and normalization transform for patch-based training."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        crop_size: int = 256,
        is_training: bool = True,
        positive_prob: float = 0.50,
        hard_negative_prob: float = 0.30,
        mean: np.ndarray = IMAGENET_MEAN,
        std: np.ndarray = IMAGENET_STD,
    ) -> None:
        self.target_size = target_size
        self.crop_size = crop_size
        self.is_training = is_training
        self.positive_prob = positive_prob
        self.hard_negative_prob = hard_negative_prob
        self.mean = mean
        self.std = std

    def _sample_crop(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample positive, hard-negative, or global context crop."""
        h, w = image.shape[:2]
        if min(h, w) < self.crop_size or not self.is_training:
            return image, mask

        r = np.random.rand()
        pos_coords = np.argwhere(mask > 127)  # [N, 2] (y, x)

        # 1. Positive crop centered near tampered characters (50%)
        if r < self.positive_prob and len(pos_coords) > 0:
            idx = np.random.randint(0, len(pos_coords))
            cy, cx = pos_coords[idx]
            # Add jitter to center
            jitter = int(self.crop_size * 0.25)
            cy += np.random.randint(-jitter, jitter + 1)
            cx += np.random.randint(-jitter, jitter + 1)

            half = self.crop_size // 2
            y1 = max(0, min(h - self.crop_size, cy - half))
            x1 = max(0, min(w - self.crop_size, cx - half))
            y2 = y1 + self.crop_size
            x2 = x1 + self.crop_size

            return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]

        # 2. Hard Negative crop (30%): Legitimate clean text with zero tampering
        elif r < (self.positive_prob + self.hard_negative_prob):
            # Try to find a patch with text (high gradient variance) but zero tampering
            for _ in range(5):
                y1 = np.random.randint(0, h - self.crop_size + 1)
                x1 = np.random.randint(0, w - self.crop_size + 1)
                y2 = y1 + self.crop_size
                x2 = x1 + self.crop_size
                crop_mask = mask[y1:y2, x1:x2]
                if np.sum(crop_mask > 127) == 0:
                    return image[y1:y2, x1:x2], crop_mask
            # Fallback random crop
            return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]

        # 3. Global Context (20%): Full page
        return image, mask

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Sample patch during training
        if self.is_training:
            img_crop, mask_crop = self._sample_crop(image, mask)
        else:
            img_crop, mask_crop = image, mask

        # 2. Resize to model input resolution
        img_resized = cv2.resize(
            img_crop, self.target_size, interpolation=cv2.INTER_LINEAR
        )
        mask_resized = cv2.resize(
            mask_crop, self.target_size, interpolation=cv2.INTER_NEAREST
        )

        # 3. Augmentations (training only)
        if self.is_training:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img_resized = cv2.flip(img_resized, 1)
                mask_resized = cv2.flip(mask_resized, 1)

            # Subtle brightness/contrast adjustment
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.90, 1.10)
                beta = np.random.uniform(-10, 10)
                img_resized = np.clip(img_resized * alpha + beta, 0, 255).astype(
                    np.uint8
                )

            # Moderate JPEG compression artifact simulation
            if np.random.rand() > 0.5:
                quality = np.random.randint(75, 95)
                _, enc = cv2.imencode(".jpg", img_resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
                img_resized = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        # 4. Normalize and convert to tensors
        img_norm = img_resized.astype(np.float32) / 255.0
        img_norm = (img_norm - self.mean) / self.std
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        mask_binary = (mask_resized > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()

        return img_tensor, mask_tensor


class DocTamperPatchDataset(Dataset):
    """DocTamper Dataset with Positive-Aware and Hard-Negative Patch Sampling."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        target_size: Tuple[int, int] = (512, 512),
        crop_size: int = 256,
        is_training: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = PositiveAwarePatchTransform(
            target_size=target_size,
            crop_size=crop_size,
            is_training=is_training,
        )
        self.pairs: List[Tuple[Path, Optional[Path]]] = []
        self._find_pairs(max_samples)

    def _find_pairs(self, max_samples: Optional[int]) -> None:
        img_dir = self.root_dir / "images"
        mask_dir = self.root_dir / "masks"
        valid_exts = {".png", ".jpg", ".jpeg"}

        entries = []
        with os.scandir(img_dir) as it:
            for entry in it:
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_exts:
                    entries.append(entry.name)
        entries.sort()

        if max_samples and len(entries) > max_samples:
            entries = entries[:max_samples]

        for name in entries:
            img_path = img_dir / name
            stem = os.path.splitext(name)[0]
            mask_path = mask_dir / f"{stem}.png"
            if not mask_path.exists():
                mask_path = mask_dir / f"{stem}.jpg"
            if not mask_path.exists():
                mask_path = None
            self.pairs.append((img_path, mask_path))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path = self.pairs[idx]
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if mask_path and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
        else:
            mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)

        img_tensor, mask_tensor = self.transform(img_rgb, mask)
        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "sample_id": img_path.stem,
        }


def create_patch_dataloaders(
    data_path: Union[str, Path],
    batch_size: int = 8,
    val_split: float = 0.15,
    target_size: Tuple[int, int] = (512, 512),
    crop_size: int = 256,
    num_workers: int = 0,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train (patch-sampled) and val (full-doc) DataLoaders."""
    train_dataset = DocTamperPatchDataset(
        root_dir=data_path,
        target_size=target_size,
        crop_size=crop_size,
        is_training=True,
        max_samples=max_samples,
    )
    val_dataset = DocTamperPatchDataset(
        root_dir=data_path,
        target_size=target_size,
        crop_size=crop_size,
        is_training=False,  # Full-page evaluation for validation
        max_samples=max_samples,
    )

    total_len = len(train_dataset)
    val_size = max(1, int(total_len * val_split))
    train_size = total_len - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_len, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_sub = torch.utils.data.Subset(train_dataset, train_indices)
    val_sub = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_sub,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_sub) >= batch_size,
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader
