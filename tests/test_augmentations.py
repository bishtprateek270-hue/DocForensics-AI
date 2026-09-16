"""
DocForensics AI — Unit & Integration Tests for Real-World Data Augmentation (Phase 4)
"""

import hashlib
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
import cv2

from config import BASE_DIR, METADATA_CSV, cfg
from src.preprocessing.augmentations import (
    get_train_augmentation_pipeline,
    get_val_augmentation_pipeline,
    apply_augmentation,
    simulate_scan_degradation,
)
from src.preprocessing.dataset import DocForensicsDataset
from src.preprocessing.augmentation_validator import validate_augmentation_suite, compute_file_md5


def test_train_and_val_pipelines_instantiation():
    """Verify Albumentations pipelines initialize with target dimensions."""
    train_pipe = get_train_augmentation_pipeline(image_size=(512, 512))
    val_pipe = get_val_augmentation_pipeline(image_size=(512, 512))

    assert train_pipe is not None
    assert val_pipe is not None


def test_geometric_alignment_and_binary_mask():
    """Verify geometric transformations keep image and mask perfectly aligned and binary {0, 255}."""
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 245
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw a distinct square in center
    mask[200:300, 300:400] = 255
    img[200:300, 300:400] = [20, 20, 20]  # dark rectangle

    pipeline = get_train_augmentation_pipeline(image_size=(512, 512))

    for _ in range(5):
        aug_img, aug_mask = apply_augmentation(img, mask, pipeline=pipeline, is_training=True)

        assert aug_img.shape == (512, 512, 3), f"Augmented image shape mismatch: {aug_img.shape}"
        assert aug_mask.shape == (512, 512), f"Augmented mask shape mismatch: {aug_mask.shape}"

        # Must be strictly binary {0, 255}
        unique_vals = set(np.unique(aug_mask))
        assert unique_vals.issubset({0, 255}), f"Mask contains non-binary values: {unique_vals}"

        # Non-empty tampered mask check
        assert np.sum(aug_mask > 0) > 0, "Tampered mask became completely empty"


def test_authentic_sample_mask_remains_strictly_zero():
    """Verify authentic document masks have exactly 0 non-zero pixels after all augmentations."""
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    zero_mask = np.zeros((h, w), dtype=np.uint8)

    pipeline = get_train_augmentation_pipeline(image_size=(512, 512))

    for _ in range(5):
        aug_img, aug_mask = apply_augmentation(img, zero_mask, pipeline=pipeline, is_training=True)
        assert np.max(aug_mask) == 0, f"Authentic mask generated non-zero pixel: {np.max(aug_mask)}"
        assert np.sum(aug_mask) == 0, f"Authentic mask sum is non-zero: {np.sum(aug_mask)}"


def test_scan_degradation_function():
    """Verify scanner simulation degrades image without throwing exceptions or changing shapes."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 230
    scan_img = simulate_scan_degradation(img)

    assert scan_img.shape == (400, 400, 3)
    assert scan_img.dtype == np.uint8
    assert scan_img.min() >= 0 and scan_img.max() <= 255


def test_dataset_immutability_on_disk():
    """Verify dataset files on disk are not altered during dataset loading and augmentation."""
    df = pd.read_csv(METADATA_CSV)
    sample_row = df.iloc[0]
    img_p = BASE_DIR / str(sample_row.get("tampered_path", sample_row.get("image_path", "")))
    mask_p = BASE_DIR / str(sample_row["mask_path"])

    orig_img_md5 = compute_file_md5(img_p)
    orig_mask_md5 = compute_file_md5(mask_p)

    # Load via dataset multiple times
    ds = DocForensicsDataset(metadata_path=METADATA_CSV, split="train", is_training=True)
    for _ in range(5):
        _ = ds[0]

    after_img_md5 = compute_file_md5(img_p)
    after_mask_md5 = compute_file_md5(mask_p)

    assert orig_img_md5 == after_img_md5, "Image on disk was modified!"
    assert orig_mask_md5 == after_mask_md5, "Mask on disk was modified!"


def test_pytorch_dataloader_batches():
    """Verify PyTorch DataLoader yields properly formatted and normalized batch tensors."""
    ds_train = DocForensicsDataset(metadata_path=METADATA_CSV, split="train", is_training=True)
    ds_val = DocForensicsDataset(metadata_path=METADATA_CSV, split="val", is_training=False)

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=4, shuffle=False)

    batch_tr = next(iter(train_loader))
    assert batch_tr["image"].shape == (4, 3, 512, 512)
    assert batch_tr["mask"].shape == (4, 1, 512, 512)
    assert batch_tr["image"].dtype == torch.float32
    assert batch_tr["mask"].dtype == torch.float32

    batch_vl = next(iter(val_loader))
    assert batch_vl["image"].shape == (4, 3, 512, 512)
    assert batch_vl["mask"].shape == (4, 1, 512, 512)


def test_augmentation_validator_suite_passes():
    """Run full Phase 4 augmentation validator suite and verify PASSED status."""
    res = validate_augmentation_suite(num_test_samples=50)
    assert res["status"] == "PASSED"
    assert res["error_count"] == 0
    assert res["disk_files_unmodified"] is True
    assert res["pytorch_loader_verified"] is True
