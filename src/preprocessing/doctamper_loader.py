"""DocTamper Dataset Loader.

Supports loading document tampering datasets formatted either as:
1. LMDB binary databases (standard DocTamper benchmark format with 'image-%09d' / 'label-%09d' keys)
2. Directory structure containing paired images and binary ground truth masks.
"""

from __future__ import annotations

import glob
import io
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Canonical ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DocTamperTransform:
    """Standardized preprocessing and augmentation transform for document tampering detection."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        is_training: bool = True,
        mean: np.ndarray = IMAGENET_MEAN,
        std: np.ndarray = IMAGENET_STD,
    ) -> None:
        self.target_size = target_size
        self.is_training = is_training
        self.mean = mean
        self.std = std

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process RGB image [H, W, 3] uint8 and mask [H, W] uint8.

        Returns:
            image_tensor: [3, target_H, target_W] float32
            mask_tensor: [1, target_H, target_W] float32 (values in {0.0, 1.0})
        """
        # Resize image & mask to target resolution
        img_resized = cv2.resize(
            image, self.target_size, interpolation=cv2.INTER_LINEAR
        )
        mask_resized = cv2.resize(
            mask, self.target_size, interpolation=cv2.INTER_NEAREST
        )

        # Apply training augmentations if enabled
        if self.is_training:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img_resized = cv2.flip(img_resized, 1)
                mask_resized = cv2.flip(mask_resized, 1)

            # Random vertical flip (lower probability for text documents)
            if np.random.rand() > 0.8:
                img_resized = cv2.flip(img_resized, 0)
                mask_resized = cv2.flip(mask_resized, 0)

            # Random subtle brightness/contrast adjustment
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.85, 1.15)
                beta = np.random.uniform(-15, 15)
                img_resized = np.clip(img_resized * alpha + beta, 0, 255).astype(
                    np.uint8
                )

        # Convert image to float32 [0, 1] and normalize
        img_norm = img_resized.astype(np.float32) / 255.0
        img_norm = (img_norm - self.mean) / self.std
        # Transpose [H, W, C] -> [C, H, W]
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        # Convert mask to binary [0.0, 1.0] tensor [1, H, W]
        mask_binary = (mask_resized > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()

        return img_tensor, mask_tensor


class DocTamperLMDBDataset(Dataset):
    """LMDB-based PyTorch Dataset for high-throughput DocTamper data loading.

    Compatible with standard DocTamper benchmark releases.
    """

    def __init__(
        self,
        lmdb_path: Union[str, Path],
        target_size: Tuple[int, int] = (512, 512),
        is_training: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        self.lmdb_path = str(lmdb_path)
        self.transform = DocTamperTransform(
            target_size=target_size, is_training=is_training
        )
        self.max_samples = max_samples
        self.env = None
        self.txn = None
        self.keys: List[str] = []
        self._init_keys()

    def _init_keys(self) -> None:
        """Scan or read index keys from the LMDB database."""
        try:
            import lmdb
        except ImportError as err:
            raise ImportError(
                "The 'lmdb' package is required for DocTamperLMDBDataset. "
                "Install it with 'pip install lmdb'."
            ) from err

        env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            num_samples_bytes = txn.get(b"num-samples")
            if num_samples_bytes is not None:
                total_samples = int(num_samples_bytes.decode("utf-8", "ignore"))
                self.keys = [f"image-{i:09d}" for i in range(1, total_samples + 1)]
            else:
                # Scan all image keys
                cursor = txn.cursor()
                keys = []
                for key, _ in cursor:
                    key_str = key.decode("utf-8", "ignore")
                    if "image" in key_str.lower() or key_str.startswith("img_"):
                        keys.append(key_str)
                self.keys = keys

        env.close()

        if self.max_samples and len(self.keys) > self.max_samples:
            self.keys = self.keys[: self.max_samples]

        logger.info(
            "Initialized DocTamperLMDBDataset at %s with %d samples.",
            self.lmdb_path,
            len(self.keys),
        )

    def _ensure_env(self) -> None:
        """Lazily initialize LMDB environment per-process/thread."""
        if self.env is None:
            import lmdb

            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.txn = self.env.begin(write=False)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._ensure_env()
        img_key = self.keys[idx]

        # Derive corresponding label/mask key
        if img_key.startswith("image-"):
            lbl_key = img_key.replace("image-", "label-")
        elif "image" in img_key:
            lbl_key = img_key.replace("image", "label")
        elif "img" in img_key:
            lbl_key = img_key.replace("img", "mask")
        else:
            lbl_key = f"{img_key}_mask"

        img_bytes = self.txn.get(img_key.encode("utf-8"))
        lbl_bytes = self.txn.get(lbl_key.encode("utf-8"))

        if img_bytes is None:
            raise KeyError(f"Image key '{img_key}' not found in LMDB.")

        # Decode image
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to decode image buffer for key {img_key}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Decode mask
        if lbl_bytes is not None:
            lbl_arr = np.frombuffer(lbl_bytes, dtype=np.uint8)
            mask = cv2.imdecode(lbl_arr, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(
                    (img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8
                )
        else:
            mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)

        # Apply transformation
        img_tensor, mask_tensor = self.transform(img_rgb, mask)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "sample_id": img_key,
        }

    def __del__(self) -> None:
        if self.env is not None:
            self.env.close()


class DocTamperFolderDataset(Dataset):
    """File-system based PyTorch Dataset for extracted DocTamper images and masks."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        target_size: Tuple[int, int] = (512, 512),
        is_training: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = DocTamperTransform(
            target_size=target_size, is_training=is_training
        )
        self.pairs: List[Tuple[Path, Optional[Path]]] = []
        self._find_pairs()

        if max_samples and len(self.pairs) > max_samples:
            self.pairs = self.pairs[:max_samples]

        logger.info(
            "Initialized DocTamperFolderDataset at %s with %d samples.",
            self.root_dir,
            len(self.pairs),
        )

    def _find_pairs(self) -> None:
        """Pair image files with corresponding mask files."""
        # Check standard subfolder layout
        img_dir = self.root_dir / "images"
        mask_dir = self.root_dir / "masks"

        if not img_dir.exists():
            img_dir = self.root_dir / "tampered"
            mask_dir = self.root_dir / "ground_truth"

        if not img_dir.exists():
            img_dir = self.root_dir

        # Gather image paths
        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        all_files = [p for p in img_dir.rglob("*") if p.suffix.lower() in valid_exts]

        # Separate masks from images if in same folder
        image_files = [f for f in all_files if not re.search(r"(_mask|_gt|label)", f.stem, re.I)]

        for img_path in sorted(image_files):
            # Try to find corresponding mask
            mask_path = None
            if mask_dir.exists() and mask_dir != img_dir:
                candidate = mask_dir / f"{img_path.stem}.png"
                if candidate.exists():
                    mask_path = candidate
                else:
                    candidates = list(mask_dir.glob(f"{img_path.stem}.*"))
                    if candidates:
                        mask_path = candidates[0]
            else:
                for suffix in ["_mask.png", "_gt.png", "_label.png", "_mask.jpg"]:
                    candidate = img_path.parent / f"{img_path.stem}{suffix}"
                    if candidate.exists():
                        mask_path = candidate
                        break

            self.pairs.append((img_path, mask_path))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path = self.pairs[idx]

        # Load RGB image
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to read image at {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Load mask
        if mask_path and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(
                    (img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8
                )
        else:
            mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)

        img_tensor, mask_tensor = self.transform(img_rgb, mask)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "sample_id": img_path.stem,
        }


class DocTamperDataset(Dataset):
    """Unified DocTamper Dataset dispatcher that automatically routes to LMDB or Folder dataset."""

    def __init__(
        self,
        data_path: Union[str, Path],
        target_size: Tuple[int, int] = (512, 512),
        is_training: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.is_lmdb = False

        # Detect LMDB
        if (
            self.data_path.is_file()
            and self.data_path.suffix in {".mdb", ".lmdb"}
        ) or (
            self.data_path.is_dir()
            and (
                (self.data_path / "data.mdb").exists()
                or any(self.data_path.glob("*.mdb"))
            )
        ):
            self.is_lmdb = True
            lmdb_dir = (
                self.data_path.parent
                if self.data_path.is_file()
                else self.data_path
            )
            self._impl: Dataset = DocTamperLMDBDataset(
                lmdb_path=lmdb_dir,
                target_size=target_size,
                is_training=is_training,
                max_samples=max_samples,
            )
        else:
            self._impl = DocTamperFolderDataset(
                root_dir=self.data_path,
                target_size=target_size,
                is_training=is_training,
                max_samples=max_samples,
            )

    def __len__(self) -> int:
        return len(self._impl)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._impl[idx]


def create_doctamper_dataloaders(
    data_path: Union[str, Path],
    batch_size: int = 4,
    val_split: float = 0.15,
    target_size: Tuple[int, int] = (512, 512),
    num_workers: int = 0,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create PyTorch DataLoaders for DocTamper training and validation.

    Args:
        data_path: Path to LMDB directory or extracted DocTamper folder.
        batch_size: Batch size per iteration.
        val_split: Fraction of samples for validation.
        target_size: (H, W) resolution (default: (512, 512)).
        num_workers: Number of background data loader processes.
        seed: Random seed for deterministic train/val split.
        max_samples: Optional cap on total dataset size.

    Returns:
        (train_loader, val_loader)
    """
    full_dataset = DocTamperDataset(
        data_path=data_path,
        target_size=target_size,
        is_training=True,
        max_samples=max_samples,
    )

    total_len = len(full_dataset)
    if total_len == 0:
        raise ValueError(f"No samples found in dataset path: {data_path}")

    if val_split > 0 and total_len >= 2:
        val_size = max(1, int(total_len * val_split))
        train_size = total_len - val_size
        if train_size == 0:
            train_size = 1
            val_size = total_len - 1
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=len(train_ds) >= batch_size,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        return train_loader, val_loader
    else:
        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        return train_loader, None
