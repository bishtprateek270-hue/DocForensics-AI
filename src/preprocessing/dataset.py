"""
DocForensics AI — PyTorch Dataset Loader (Phase 4)
Loads document images and ground-truth tampering masks with Albumentations augmentation pipeline.
Ensures zero data contamination:
- 'train' split uses stochastic real-world augmentations
- 'val' and 'test' splits use strictly deterministic resizing & normalization
"""

import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from torch.utils.data import Dataset

from config import BASE_DIR, METADATA_CSV, cfg
from src.preprocessing.augmentations import (
    get_train_augmentation_pipeline,
    get_val_augmentation_pipeline,
    apply_augmentation,
)


class DocForensicsDataset(Dataset):
    """
    Production-ready PyTorch Dataset for Document Tampering Detection & Localization.
    """

    def __init__(
        self,
        metadata_path: Path = METADATA_CSV,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        is_training: Optional[bool] = None,
        normalize: bool = True,
        df: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.split = split.lower()
        self.is_training = is_training if is_training is not None else (self.split == "train")
        self.normalize = normalize

        if df is not None:
            self.df = df.copy()
        else:
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file {metadata_path} not found.")
            master_df = pd.read_csv(metadata_path)
            split_col = "dataset_split" if "dataset_split" in master_df.columns else "split"
            self.df = master_df[master_df[split_col] == self.split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split: '{self.split}'")

        # Select pipeline
        if self.is_training:
            self.pipeline = get_train_augmentation_pipeline(image_size=self.image_size)
        else:
            self.pipeline = get_val_augmentation_pipeline(image_size=self.image_size)

        # ImageNet normalization parameters
        self.mean = np.array(cfg.preprocessing.mean, dtype=np.float32)
        self.std = np.array(cfg.preprocessing.std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]

        img_rel = row.get("tampered_path", row.get("image_path", ""))
        mask_rel = row["mask_path"]

        img_p = BASE_DIR / img_rel
        mask_p = BASE_DIR / mask_rel

        # Load RGB image
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {img_p}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Load single-channel grayscale mask
        mask_gray = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise FileNotFoundError(f"Could not load mask: {mask_p}")

        # Apply augmentation with guaranteed alignment & binary mask
        aug_img, aug_mask = apply_augmentation(
            img_rgb,
            mask_gray,
            pipeline=self.pipeline,
            is_training=self.is_training,
            image_size=self.image_size,
        )

        # Prepare normalized tensor: [3, H, W] in range [0, 1] then normalized
        img_norm = aug_img.astype(np.float32) / 255.0
        if self.normalize:
            img_norm = (img_norm - self.mean) / self.std

        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        # Prepare mask tensor: [1, H, W] in range {0.0, 1.0}
        mask_tensor = torch.from_numpy((aug_mask > 127).astype(np.float32)).unsqueeze(0)

        item = {
            "image": img_tensor,
            "mask": mask_tensor,
            "sample_id": sample_id,
            "is_tampered": torch.tensor(int(row["is_tampered"]), dtype=torch.long),
            "source_type": str(row.get("source_type", "synthetic")),
            "manipulation_type": str(row.get("manipulation_type", "none")),
            "split": self.split,
        }
        return item

    def get_visual_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Returns raw numpy representations for visual auditing:
        (orig_image_rgb, aug_image_rgb, orig_mask, aug_mask, row_metadata)
        """
        row = self.df.iloc[idx]
        img_rel = row.get("tampered_path", row.get("image_path", ""))
        mask_rel = row["mask_path"]
        orig_rel = row.get("original_path")

        img_p = BASE_DIR / img_rel
        mask_p = BASE_DIR / mask_rel

        img_bgr = cv2.imread(str(img_p))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask_gray = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

        if orig_rel and (BASE_DIR / orig_rel).exists():
            orig_bgr = cv2.imread(str(BASE_DIR / orig_rel))
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        else:
            orig_rgb = img_rgb.copy()

        aug_img, aug_mask = apply_augmentation(
            img_rgb,
            mask_gray,
            pipeline=self.pipeline,
            is_training=self.is_training,
            image_size=self.image_size,
        )

        return orig_rgb, aug_img, mask_gray, aug_mask, row.to_dict()
