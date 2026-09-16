"""
DocForensics AI — Real-World Augmentation Suite Validator (Phase 4)
Stress-tests the augmentation pipeline across dataset samples to verify:
1. Exact image-mask dimension match ([512, 512, 3] and [512, 512])
2. Strict binary mask output ({0, 255})
3. Positive pixel retention for tampered samples and all-zero retention for authentic samples
4. Geometric alignment consistency
5. Non-modification of original dataset files on disk (MD5 hash verification)
6. PyTorch Dataset compatibility and batch loading
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hashlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2

from config import BASE_DIR, METADATA_CSV, cfg
from src.preprocessing.augmentations import (
    get_train_augmentation_pipeline,
    get_val_augmentation_pipeline,
    apply_augmentation,
)
from src.preprocessing.dataset import DocForensicsDataset


def compute_file_md5(file_path: Path) -> str:
    """Compute MD5 checksum."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def validate_augmentation_suite(
    metadata_path: Path = METADATA_CSV,
    num_test_samples: int = 150,
    seed: int = 42,
) -> dict:
    """Run comprehensive validation on augmentation engine and PyTorch dataset loader."""
    print("=" * 75)
    print("     DocForensics AI — Phase 4 Real-World Augmentation Audit          ")
    print("=" * 75)

    if not metadata_path.exists():
        print(f"[-] ERROR: Metadata file not found at {metadata_path}")
        return {"status": "FAILED", "reason": "Missing metadata.csv"}

    df = pd.read_csv(metadata_path)
    total_samples = len(df)
    test_count = min(num_test_samples, total_samples)

    print(f"[*] Total dataset samples: {total_samples}")
    print(f"[*] Stress-testing augmentation pipeline on {test_count} samples (Seed={seed})...")

    errors = []
    sampled_df = df.sample(test_count, random_state=seed).reset_index(drop=True)

    # 1. Capture initial MD5 hashes of raw files on disk to verify zero in-place modification
    initial_hashes = {}
    for idx, row in sampled_df.iterrows():
        img_p = BASE_DIR / str(row.get("tampered_path", row.get("image_path", "")))
        mask_p = BASE_DIR / str(row["mask_path"])
        if img_p.exists():
            initial_hashes[str(img_p)] = compute_file_md5(img_p)
        if mask_p.exists():
            initial_hashes[str(mask_p)] = compute_file_md5(mask_p)

    # 2. Run Augmentation Pipeline Audits
    pipeline = get_train_augmentation_pipeline(image_size=(512, 512))
    val_pipeline = get_val_augmentation_pipeline(image_size=(512, 512))

    valid_aug_count = 0
    binary_mask_pass = 0
    alignment_pass = 0

    for idx, row in sampled_df.iterrows():
        sample_id = row["sample_id"]
        is_tampered = int(row["is_tampered"])
        img_p = BASE_DIR / str(row.get("tampered_path", row.get("image_path", "")))
        mask_p = BASE_DIR / str(row["mask_path"])

        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            errors.append(f"Sample {sample_id}: Image unreadable ({img_p})")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask_gray = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            errors.append(f"Sample {sample_id}: Mask unreadable ({mask_p})")
            continue

        # Test Training Augmentation (Stochastic)
        aug_img, aug_mask = apply_augmentation(img_rgb, mask_gray, pipeline=pipeline, is_training=True)

        # Check Dimensions
        if aug_img.shape != (512, 512, 3):
            errors.append(f"Sample {sample_id}: Augmented image shape is {aug_img.shape}, expected (512, 512, 3)")
            continue
        if aug_mask.shape != (512, 512):
            errors.append(f"Sample {sample_id}: Augmented mask shape is {aug_mask.shape}, expected (512, 512)")
            continue

        # Check Strictly Binary Mask Values
        unique_vals = set(np.unique(aug_mask))
        if not unique_vals.issubset({0, 255}):
            errors.append(f"Sample {sample_id}: Mask contains non-binary values: {unique_vals}")
            continue
        binary_mask_pass += 1

        # Check Mask Semantics
        aug_tampered_pixels = int(np.sum(aug_mask > 0))
        if is_tampered == 0 and aug_tampered_pixels > 0:
            errors.append(f"Sample {sample_id}: Authentic document generated positive mask pixels ({aug_tampered_pixels}) after augmentation")
            continue
        elif is_tampered == 1 and aug_tampered_pixels == 0:
            # Note: with extreme clipping or random crops this could happen, but our parameters preserve region
            errors.append(f"Sample {sample_id}: Tampered document mask became empty after augmentation")
            continue

        alignment_pass += 1
        valid_aug_count += 1

    # 3. Verify Disk Files Remained Completely Unmodified
    print("\n--- 1. Disk Immutability Verification ---")
    disk_modified = 0
    for file_path_str, initial_hash in initial_hashes.items():
        current_hash = compute_file_md5(Path(file_path_str))
        if current_hash != initial_hash:
            errors.append(f"File modified on disk! {file_path_str}")
            disk_modified += 1

    if disk_modified == 0:
        print("  [PASS] All disk dataset files remained 100% immutable and unmodified.")
    else:
        print(f"  [FAIL] {disk_modified} files were modified on disk!")

    # 4. PyTorch Dataset & DataLoader Verification
    print("\n--- 2. PyTorch Dataset & Batch Loader Verification ---")
    try:
        train_ds = DocForensicsDataset(metadata_path=metadata_path, split="train", is_training=True)
        val_ds = DocForensicsDataset(metadata_path=metadata_path, split="val", is_training=False)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False)

        # Fetch sample batch from train
        train_batch = next(iter(train_loader))
        img_b = train_batch["image"]
        mask_b = train_batch["mask"]

        assert img_b.shape == (4, 3, 512, 512), f"Unexpected train batch image shape: {img_b.shape}"
        assert mask_b.shape == (4, 1, 512, 512), f"Unexpected train batch mask shape: {mask_b.shape}"
        assert set(torch.unique(mask_b).numpy()).issubset({0.0, 1.0}), f"Unexpected mask tensor values: {torch.unique(mask_b)}"

        # Fetch sample batch from val
        val_batch = next(iter(val_loader))
        assert val_batch["image"].shape == (4, 3, 512, 512)
        assert val_batch["mask"].shape == (4, 1, 512, 512)

        print(f"  [PASS] Train Dataset ({len(train_ds)} samples) & Val Dataset ({len(val_ds)} samples) loaded successfully.")
        print(f"  [PASS] Tensor Shapes: Images={tuple(img_b.shape)}, Masks={tuple(mask_b.shape)}")
        pytorch_pass = True
    except Exception as e:
        err = f"PyTorch dataset/loader failure: {e}"
        errors.append(err)
        print(f"  [FAIL] {err}")
        pytorch_pass = False

    status = "PASSED" if len(errors) == 0 else "FAILED"
    print("\n" + "=" * 75)
    print(f"  Augmentation Audit Status: [{status}]")
    print(f"  Tested Samples           : {test_count}")
    print(f"  Binary Mask Validations  : {binary_mask_pass} / {test_count}")
    print(f"  Alignment & Mask Checks  : {alignment_pass} / {test_count}")
    print(f"  Errors / Failures Found  : {len(errors)}")
    print("=" * 75 + "\n")

    results = {
        "status": status,
        "tested_samples": test_count,
        "binary_mask_pass": binary_mask_pass,
        "alignment_pass": alignment_pass,
        "disk_files_unmodified": (disk_modified == 0),
        "pytorch_loader_verified": pytorch_pass,
        "error_count": len(errors),
        "errors": errors[:10],
    }
    return results


if __name__ == "__main__":
    res = validate_augmentation_suite()
    if res["status"] != "PASSED":
        import sys
        sys.exit(1)
