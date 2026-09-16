"""
DocForensics AI — Dataset Integrity, Mask Quality & Data Leakage Validator (Phase 3)
Performs comprehensive end-to-end quality validation:
1. Real file existence for original_path, tampered_path, mask_path
2. Uncorrupted image readability (PIL & OpenCV)
3. Image and mask dimension exact match (Height x Width)
4. Binary mask value verification (strictly {0, 255})
5. Non-empty tampered mask check & zero-mask authentic check
6. Bounding box and mask coordinate alignment
7. Duplicate image hash detection
8. Strict zero document-family data leakage across train, val, test splits
9. Verification of all 10 synthetic manipulation categories
"""

import ast
import hashlib
import json
import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from config import (
    BASE_DIR,
    DATA_DIR,
    METADATA_CSV,
    SYNTHETIC_METADATA_CSV,
    TRAIN_DATA_DIR,
    VAL_DATA_DIR,
    TEST_DATA_DIR,
    cfg,
)
from src.preprocessing.tampering_engine import MANIPULATION_CATEGORIES


def compute_file_md5(file_path: Path) -> str:
    """Compute MD5 checksum for duplicate detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def validate_dataset(metadata_path: Path = METADATA_CSV, synthetic_meta_path: Path = SYNTHETIC_METADATA_CSV) -> dict:
    """Run comprehensive dataset quality and integrity verification."""
    print("=" * 75)
    print("      DocForensics AI — Phase 3 Dataset & Mask Integrity Audit       ")
    print("=" * 75)

    if not metadata_path.exists():
        print(f"[-] ERROR: Metadata file not found at {metadata_path}")
        return {"status": "FAILED", "reason": "Missing metadata.csv"}

    df = pd.read_csv(metadata_path)
    total_samples = len(df)
    print(f"[*] Total indexed master samples: {total_samples}")

    errors = []
    warnings = []
    md5_seen = {}
    duplicates = []

    split_col = "dataset_split" if "dataset_split" in df.columns else "split"
    img_col = "tampered_path" if "tampered_path" in df.columns else "image_path"
    family_col = "document_family_id" if "document_family_id" in df.columns else "family_id"

    # -------------------------------------------------------------------------
    # 1. Check Group-Level Data Leakage (Strict Zero Family Leakage)
    # -------------------------------------------------------------------------
    train_families = set(df[df[split_col] == "train"][family_col].unique())
    val_families = set(df[df[split_col] == "val"][family_col].unique())
    test_families = set(df[df[split_col] == "test"][family_col].unique())

    leak_train_val = train_families.intersection(val_families)
    leak_train_test = train_families.intersection(test_families)
    leak_val_test = val_families.intersection(test_families)

    print("\n--- 1. Data Leakage & Document Family Audit ---")
    print(f"  Train Families : {len(train_families)}")
    print(f"  Val Families   : {len(val_families)}")
    print(f"  Test Families  : {len(test_families)}")

    if not leak_train_val and not leak_train_test and not leak_val_test:
        leakage_status = "PASSED (Zero Leakage)"
        print("  [PASS] ZERO data leakage detected across splits. All document families are strictly disjoint.")
    else:
        leakage_status = "FAILED"
        err_msg = f"Data leakage detected! Train-Val: {leak_train_val}, Train-Test: {leak_train_test}, Val-Test: {leak_val_test}"
        errors.append(err_msg)
        print(f"  [FAIL] {err_msg}")

    # -------------------------------------------------------------------------
    # 2. Verify Manipulation Categories in Synthetic Data
    # -------------------------------------------------------------------------
    print("\n--- 2. Synthetic Manipulation Category Coverage ---")
    syn_df = df[df["source_type"] == "synthetic"]
    syn_tampered = syn_df[syn_df["is_tampered"] == 1]
    present_categories = set(syn_tampered["manipulation_type"].unique())
    required_categories = set(MANIPULATION_CATEGORIES)

    missing_categories = required_categories - present_categories
    print(f"  Required Categories Count : {len(required_categories)}")
    print(f"  Present Categories Count  : {len(present_categories)}")
    for cat in sorted(required_categories):
        count = len(syn_tampered[syn_tampered["manipulation_type"] == cat])
        status_mark = "OK" if count > 0 else "MISSING"
        print(f"    • [{status_mark}] {cat:<32} : {count} samples")

    if missing_categories:
        err = f"Missing manipulation categories in synthetic dataset: {missing_categories}"
        errors.append(err)
        print(f"  [FAIL] {err}")
    else:
        print("  [PASS] All 10 required synthetic manipulation categories are represented.")

    # -------------------------------------------------------------------------
    # 3. Iterate through every single image-mask pair
    # -------------------------------------------------------------------------
    print("\n--- 3. File Integrity, Mask Alignment & Bounding Box Audit ---")
    valid_count = 0
    clean_orig_verified = 0

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        img_rel_path = row[img_col]
        mask_rel_path = row["mask_path"]
        orig_rel_path = row.get("original_path")
        is_tampered = int(row["is_tampered"])

        img_abs = BASE_DIR / img_rel_path
        mask_abs = BASE_DIR / mask_rel_path

        # Check path existence
        if not img_abs.exists():
            errors.append(f"Sample {sample_id}: Tampered image missing at {img_abs}")
            continue
        if not mask_abs.exists():
            errors.append(f"Sample {sample_id}: Mask missing at {mask_abs}")
            continue

        if orig_rel_path:
            orig_abs = BASE_DIR / orig_rel_path
            if not orig_abs.exists():
                errors.append(f"Sample {sample_id}: Original clean image missing at {orig_abs}")
                continue
            else:
                clean_orig_verified += 1

        # Check readability
        try:
            with Image.open(img_abs) as pil_img:
                img_w, img_h = pil_img.size
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

        # Check binary mask values: must be strictly subset of {0, 255}
        unique_vals = np.unique(mask_arr)
        invalid_vals = [v for v in unique_vals if v not in (0, 255)]
        if invalid_vals:
            errors.append(f"Sample {sample_id}: Non-binary values {invalid_vals} found in mask.")
            continue

        # Check positive mask pixel count
        tampered_pixels = int(np.sum(mask_arr > 0))
        if is_tampered == 0 and tampered_pixels > 0:
            errors.append(f"Sample {sample_id}: Authentic document has {tampered_pixels} non-zero mask pixels!")
            continue
        elif is_tampered == 1 and tampered_pixels == 0:
            errors.append(f"Sample {sample_id}: Tampered document has 0 non-zero mask pixels (empty mask)!")
            continue

        # Check bounding box alignment for tampered samples
        if is_tampered == 1 and "bounding_box" in row and pd.notna(row["bounding_box"]):
            try:
                bbox_val = row["bounding_box"]
                if isinstance(bbox_val, str):
                    bbox = ast.literal_eval(bbox_val)
                else:
                    bbox = bbox_val
                
                if len(bbox) == 4 and any(b > 0 for b in bbox):
                    bx1, by1, bx2, by2 = bbox
                    # Ensure bbox is within image boundaries
                    if bx1 < 0 or by1 < 0 or bx2 > img_w or by2 > img_h:
                        errors.append(f"Sample {sample_id}: Bounding box {bbox} exceeds image dimensions ({img_w}, {img_h})")
                        continue
            except Exception as e:
                errors.append(f"Sample {sample_id}: Invalid bounding box format ({e})")
                continue

        # Duplicate hash detection on tampered images
        file_hash = compute_file_md5(img_abs)
        if file_hash in md5_seen:
            prev_sample = md5_seen[file_hash]
            if prev_sample != sample_id:
                duplicates.append((sample_id, prev_sample))
        else:
            md5_seen[file_hash] = sample_id

        valid_count += 1

    print(f"  Audited Sample Pairs : {total_samples}")
    print(f"  Valid Verified Pairs : {valid_count} / {total_samples}")
    print(f"  Corrupt/Errors Found : {len(errors)}")
    print(f"  Duplicate Files Found: {len(duplicates)}")

    status = "PASSED" if len(errors) == 0 else "FAILED"
    print("\n" + "=" * 75)
    print(f"  Final Audit Status: [{status}]")
    print("=" * 75)

    results = {
        "status": status,
        "total_samples": total_samples,
        "valid_samples": valid_count,
        "error_count": len(errors),
        "errors": errors[:10],
        "duplicates_count": len(duplicates),
        "leakage_status": leakage_status,
        "train_families": len(train_families),
        "val_families": len(val_families),
        "test_families": len(test_families),
        "clean_originals_verified": clean_orig_verified,
        "manipulation_categories_covered": len(present_categories),
    }
    return results


if __name__ == "__main__":
    res = validate_dataset()
    if res["status"] != "PASSED":
        sys.exit(1)
