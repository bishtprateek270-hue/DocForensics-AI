"""
DocForensics AI — Dataset Integrity & Data Leakage Validator
Performs comprehensive quality checks:
1. File existence and uncorrupted readability
2. Image-Mask dimension match (Height x Width)
3. Binary mask value range validation (0 and 255)
4. Group-based partition validation (Strict zero family leakage across splits)
5. Non-empty tampered mask check & all-zero authentic mask check
6. Duplicate image hash detection
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2

from config import (
    BASE_DIR,
    DATA_DIR,
    METADATA_CSV,
    TRAIN_DATA_DIR,
    VAL_DATA_DIR,
    TEST_DATA_DIR,
    cfg,
)


def compute_file_md5(file_path: Path) -> str:
    """Compute MD5 checksum for duplicate detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def validate_dataset(metadata_path: Path = METADATA_CSV) -> dict:
    """Run end-to-end dataset validation suite."""
    print("=" * 70)
    print("        DocForensics AI — Dataset Validation & Leakage Audit         ")
    print("=" * 70)

    if not metadata_path.exists():
        print(f"[-] ERROR: Metadata file not found at {metadata_path}")
        return {"status": "FAILED", "reason": "Missing metadata.csv"}

    df = pd.read_csv(metadata_path)
    total_samples = len(df)
    print(f"[*] Total indexed samples in metadata: {total_samples}")

    errors = []
    warnings = []
    md5_seen = {}
    duplicates = []

    # 1. Check Group-Level Data Leakage
    train_families = set(df[df["split"] == "train"]["document_family_id"].unique())
    val_families = set(df[df["split"] == "val"]["document_family_id"].unique())
    test_families = set(df[df["split"] == "test"]["document_family_id"].unique())

    leak_train_val = train_families.intersection(val_families)
    leak_train_test = train_families.intersection(test_families)
    leak_val_test = val_families.intersection(test_families)

    print("\n--- 1. Data Leakage & Group Partition Audit ---")
    print(f"  Train Families : {len(train_families)}")
    print(f"  Val Families   : {len(val_families)}")
    print(f"  Test Families  : {len(test_families)}")

    if not leak_train_val and not leak_train_test and not leak_val_test:
        print("  [PASS] ZERO data leakage detected across splits. All document families are strictly disjoint.")
    else:
        err_msg = f"Data leakage detected! Train-Val: {leak_train_val}, Train-Test: {leak_train_test}, Val-Test: {leak_val_test}"
        errors.append(err_msg)
        print(f"  [FAIL] {err_msg}")

    # 2. Iterate through every single image-mask pair
    print("\n--- 2. File Integrity & Binary Mask Alignment Audit ---")
    
    valid_count = 0
    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        img_rel_path = row["image_path"]
        mask_rel_path = row["mask_path"]
        is_tampered = int(row["is_tampered"])

        img_abs = BASE_DIR / img_rel_path
        mask_abs = BASE_DIR / mask_rel_path

        # Check existence
        if not img_abs.exists():
            errors.append(f"Sample {sample_id}: Image missing at {img_abs}")
            continue
        if not mask_abs.exists():
            errors.append(f"Sample {sample_id}: Mask missing at {mask_abs}")
            continue

        # Check readability
        try:
            with Image.open(img_abs) as pil_img:
                img_w, img_h = pil_img.size
                img_mode = pil_img.mode
        except Exception as e:
            errors.append(f"Sample {sample_id}: Image corrupt ({e})")
            continue

        try:
            mask_arr = cv2.imread(str(mask_abs), cv2.IMREAD_GRAYSCALE)
            if mask_arr is None:
                errors.append(f"Sample {sample_id}: Mask unreadable with OpenCV")
                continue
            mask_h, mask_w = mask_arr.shape
        except Exception as e:
            errors.append(f"Sample {sample_id}: Mask corrupt ({e})")
            continue

        # Check dimension match
        if (img_w != mask_w) or (img_h != mask_h):
            errors.append(f"Sample {sample_id}: Dimension mismatch! Image: {img_w}x{img_h}, Mask: {mask_w}x{mask_h}")
            continue

        # Check mask values
        unique_vals = np.unique(mask_arr)
        for val in unique_vals:
            if val not in (0, 255):
                errors.append(f"Sample {sample_id}: Non-binary value {val} found in mask.")
                break

        # Check authentic vs tampered mask logic
        non_zero_count = int(np.sum(mask_arr > 0))
        if is_tampered == 0 and non_zero_count > 0:
            errors.append(f"Sample {sample_id}: Authentic document has {non_zero_count} positive mask pixels!")
        elif is_tampered == 1 and non_zero_count == 0:
            errors.append(f"Sample {sample_id}: Tampered document has 0 positive mask pixels!")

        # Check file duplicate
        file_hash = compute_file_md5(img_abs)
        if file_hash in md5_seen:
            # Different samples having identical image content
            prev_sample = md5_seen[file_hash]
            if prev_sample != sample_id:
                duplicates.append((sample_id, prev_sample))
        else:
            md5_seen[file_hash] = sample_id

        valid_count += 1

    print(f"  Audited Pairs  : {total_samples}")
    print(f"  Valid Pairs    : {valid_count} / {total_samples}")
    print(f"  Corrupt/Errors : {len(errors)}")
    print(f"  Duplicates     : {len(duplicates)}")

    status = "PASSED" if len(errors) == 0 else "FAILED"
    print("\n" + "=" * 70)
    print(f"  Final Audit Verdict: [{status}]")
    print("=" * 70)

    results = {
        "status": status,
        "total_samples": total_samples,
        "valid_samples": valid_count,
        "error_count": len(errors),
        "errors": errors[:10],  # top 10
        "duplicates_count": len(duplicates),
        "train_families": len(train_families),
        "val_families": len(val_families),
        "test_families": len(test_families),
    }
    return results


if __name__ == "__main__":
    res = validate_dataset()
    if res["status"] != "PASSED":
        sys.exit(1)
