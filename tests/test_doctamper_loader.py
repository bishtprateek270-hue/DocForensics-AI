"""Unit and Integration tests for DocTamper Dataset Loader and Training Pipeline."""

import io
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import lmdb
import numpy as np
import pytest
import torch

from src.models.dual_stream_forensics import DualStreamForensicNet
from src.preprocessing.doctamper_loader import (
    DocTamperDataset,
    DocTamperFolderDataset,
    DocTamperLMDBDataset,
    DocTamperTransform,
    create_doctamper_dataloaders,
)
from src.training.train_doctamper import (
    BCEDiceLoss,
    DiceLoss,
    compute_metrics,
    run_training,
)


@pytest.fixture
def mock_image_mask():
    """Create synthetic test image and binary mask."""
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:150, 50:150] = 255  # tampered region
    return img, mask


@pytest.fixture
def temp_lmdb_dir(mock_image_mask):
    """Create a temporary LMDB database populated with mock DocTamper records."""
    temp_dir = tempfile.mkdtemp()
    img, mask = mock_image_mask

    # Encode to JPEG / PNG bytes
    _, img_bytes = cv2.imencode(".jpg", img)
    _, mask_bytes = cv2.imencode(".png", mask)

    env = lmdb.open(temp_dir, map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        # Write 5 samples in official DocTamper format
        txn.put(b"num-samples", b"5")
        for i in range(1, 6):
            img_key = f"image-{i:09d}".encode("utf-8")
            lbl_key = f"label-{i:09d}".encode("utf-8")
            txn.put(img_key, img_bytes.tobytes())
            txn.put(lbl_key, mask_bytes.tobytes())
    env.close()

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_folder_dir(mock_image_mask):
    """Create a temporary folder structure populated with mock image and mask files."""
    temp_dir = tempfile.mkdtemp()
    img, mask = mock_image_mask

    img_dir = Path(temp_dir) / "images"
    mask_dir = Path(temp_dir) / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    for i in range(4):
        cv2.imwrite(str(img_dir / f"doc_{i:04d}.png"), img)
        cv2.imwrite(str(mask_dir / f"doc_{i:04d}.png"), mask)

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_doctamper_transform(mock_image_mask):
    img, mask = mock_image_mask
    transform = DocTamperTransform(target_size=(512, 512), is_training=False)
    img_tensor, mask_tensor = transform(img, mask)

    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(mask_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 512, 512)
    assert mask_tensor.shape == (1, 512, 512)
    assert mask_tensor.max() <= 1.0 and mask_tensor.min() >= 0.0


def test_lmdb_dataset_loading(temp_lmdb_dir):
    ds = DocTamperLMDBDataset(temp_lmdb_dir, target_size=(512, 512), is_training=False)
    assert len(ds) == 5

    sample = ds[0]
    assert "image" in sample
    assert "mask" in sample
    assert sample["image"].shape == (3, 512, 512)
    assert sample["mask"].shape == (1, 512, 512)
    assert sample["sample_id"] == "image-000000001"


def test_folder_dataset_loading(temp_folder_dir):
    ds = DocTamperFolderDataset(temp_folder_dir, target_size=(512, 512), is_training=False)
    assert len(ds) == 4

    sample = ds[0]
    assert sample["image"].shape == (3, 512, 512)
    assert sample["mask"].shape == (1, 512, 512)


def test_unified_dataset_autodetect(temp_lmdb_dir, temp_folder_dir):
    ds_lmdb = DocTamperDataset(temp_lmdb_dir)
    assert ds_lmdb.is_lmdb is True
    assert len(ds_lmdb) == 5

    ds_folder = DocTamperDataset(temp_folder_dir)
    assert ds_folder.is_lmdb is False
    assert len(ds_folder) == 4


def test_create_dataloaders(temp_lmdb_dir):
    train_loader, val_loader = create_doctamper_dataloaders(
        data_path=temp_lmdb_dir,
        batch_size=2,
        val_split=0.2,
        num_workers=0,
    )
    assert train_loader is not None
    assert val_loader is not None

    batch = next(iter(train_loader))
    assert batch["image"].shape == (2, 3, 512, 512)
    assert batch["mask"].shape == (2, 1, 512, 512)


def test_loss_and_metrics():
    logits = torch.zeros((2, 1, 64, 64))
    targets = torch.zeros((2, 1, 64, 64))
    targets[:, :, 10:30, 10:30] = 1.0

    criterion = BCEDiceLoss()
    loss = criterion(logits, targets)
    assert loss.item() > 0.0

    metrics = compute_metrics(logits, targets)
    assert "dice" in metrics
    assert "iou" in metrics
    assert "precision" in metrics
    assert "recall" in metrics


def test_training_pipeline_execution(temp_lmdb_dir):
    output_ckpt = Path(temp_lmdb_dir) / "test_out_best.pth"
    summary = run_training(
        data_path=temp_lmdb_dir,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        pretrained_path=None,  # Scratch init for fast test
        output_checkpoint=str(output_ckpt),
        val_split=0.2,
        device_name="cpu",
    )
    assert summary is not None
    assert "best_val_dice" in summary
    assert output_ckpt.exists()
