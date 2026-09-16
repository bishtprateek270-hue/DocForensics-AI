"""
DocForensics AI — Comprehensive Phase 3 Dataset & Manipulation Tests
Validates:
1. Master and synthetic metadata file structure
2. Directory organization (data/synthetic/originals, tampered, masks)
3. Full representation of all 10 distinct tampering categories
4. Strict zero document family data leakage across splits
5. Mask dimensions match image dimensions exactly
6. Binary mask values strictly in {0, 255}
7. Non-empty tampered masks and strictly zero authentic masks
8. Bounding box coordinates validity
9. Summary JSON profile validity
"""

import ast
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2

from config import (
    BASE_DIR,
    METADATA_CSV,
    SYNTHETIC_DIR,
    SYNTHETIC_ORIGINALS_DIR,
    SYNTHETIC_TAMPERED_DIR,
    SYNTHETIC_MASKS_DIR,
    SYNTHETIC_METADATA_CSV,
    TRAIN_DATA_DIR,
    VAL_DATA_DIR,
    TEST_DATA_DIR,
    DATASET_SUMMARY_JSON,
)
from src.preprocessing.tampering_engine import MANIPULATION_CATEGORIES
from src.preprocessing.dataset_validator import validate_dataset


def test_metadata_files_exist():
    """Verify metadata files exist for both master dataset and synthetic subset."""
    assert METADATA_CSV.exists(), f"Master metadata file {METADATA_CSV} does not exist"
    assert SYNTHETIC_METADATA_CSV.exists(), f"Synthetic metadata {SYNTHETIC_METADATA_CSV} does not exist"


def test_synthetic_directory_structure():
    """Verify synthetic directory structure is properly created."""
    assert SYNTHETIC_DIR.exists(), f"Missing {SYNTHETIC_DIR}"
    assert SYNTHETIC_ORIGINALS_DIR.exists(), f"Missing {SYNTHETIC_ORIGINALS_DIR}"
    assert SYNTHETIC_TAMPERED_DIR.exists(), f"Missing {SYNTHETIC_TAMPERED_DIR}"
    assert SYNTHETIC_MASKS_DIR.exists(), f"Missing {SYNTHETIC_MASKS_DIR}"
    assert len(list(SYNTHETIC_ORIGINALS_DIR.glob("*.png"))) > 0, "No original clean documents found"
    assert len(list(SYNTHETIC_TAMPERED_DIR.glob("*.png"))) > 0, "No synthetic tampered documents found"
    assert len(list(SYNTHETIC_MASKS_DIR.glob("*.png"))) > 0, "No synthetic masks found"


def test_all_10_manipulation_categories_present():
    """Verify all 10 manipulation categories are generated in the synthetic dataset."""
    df_syn = pd.read_csv(SYNTHETIC_METADATA_CSV)
    syn_tampered = df_syn[df_syn["is_tampered"] == 1]
    present_categories = set(syn_tampered["manipulation_type"].unique())
    
    for cat in MANIPULATION_CATEGORIES:
        assert cat in present_categories, f"Manipulation category '{cat}' missing from synthetic dataset!"
        count = len(syn_tampered[syn_tampered["manipulation_type"] == cat])
        assert count > 0, f"Category '{cat}' has 0 samples!"


def test_dataset_validator_passes():
    """Verify full end-to-end dataset validation suite returns PASSED with 0 errors."""
    res = validate_dataset()
    assert res["status"] == "PASSED", f"Validator failed with errors: {res.get('errors')}"
    assert res["error_count"] == 0, f"Validator found {res['error_count']} errors: {res.get('errors')}"
    assert res["total_samples"] > 0
    assert res["clean_originals_verified"] > 0
    assert res["manipulation_categories_covered"] == 10


def test_zero_data_leakage():
    """Verify zero document family leakage across train, val, and test splits."""
    df = pd.read_csv(METADATA_CSV)
    split_col = "dataset_split" if "dataset_split" in df.columns else "split"
    family_col = "document_family_id" if "document_family_id" in df.columns else "family_id"

    train_families = set(df[df[split_col] == "train"][family_col])
    val_families = set(df[df[split_col] == "val"][family_col])
    test_families = set(df[df[split_col] == "test"][family_col])

    assert len(train_families.intersection(val_families)) == 0, "Leakage between train and val splits!"
    assert len(train_families.intersection(test_families)) == 0, "Leakage between train and test splits!"
    assert len(val_families.intersection(test_families)) == 0, "Leakage between val and test splits!"


def test_mask_properties_and_dimensions():
    """Verify image-mask dimension match, strictly binary {0, 255} mask values, and positive pixels."""
    df = pd.read_csv(METADATA_CSV)
    img_col = "tampered_path" if "tampered_path" in df.columns else "image_path"

    # Sample across synthetic and public
    sample_df = df.sample(min(30, len(df)), random_state=42)
    for _, row in sample_df.iterrows():
        img_p = BASE_DIR / row[img_col]
        mask_p = BASE_DIR / row["mask_path"]
        is_tampered = int(row["is_tampered"])

        with Image.open(img_p) as img:
            w, h = img.size
        
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f"Could not read mask: {mask_p}"
        mh, mw = mask.shape
        assert (w, h) == (mw, mh), f"Dimension mismatch: img={(w, h)}, mask={(mw, mh)}"
        
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 255}), f"Mask contains non-binary values: {unique_vals}"

        non_zero_count = int(np.sum(mask > 0))
        if is_tampered == 1:
            assert non_zero_count > 0, f"Tampered image has empty mask ({img_p})"
        else:
            assert non_zero_count == 0, f"Authentic image has positive mask ({img_p})"


def test_bounding_box_validity():
    """Verify bounding boxes for tampered images are valid [x1, y1, x2, y2] within bounds."""
    df = pd.read_csv(METADATA_CSV)
    tampered_df = df[df["is_tampered"] == 1]

    for _, row in tampered_df.head(30).iterrows():
        bbox_val = row["bounding_box"]
        if isinstance(bbox_val, str):
            bbox = ast.literal_eval(bbox_val)
        else:
            bbox = bbox_val

        assert len(bbox) == 4, f"Invalid bbox format: {bbox}"
        bx1, by1, bx2, by2 = bbox
        assert bx1 >= 0 and by1 >= 0, f"Negative bbox coordinates: {bbox}"
        assert bx2 <= row["width"] and by2 <= row["height"], f"BBox {bbox} exceeds dimensions ({row['width']}, {row['height']})"


def test_dataset_summary_json():
    """Verify dataset summary JSON is populated with valid Phase 3 metrics."""
    assert DATASET_SUMMARY_JSON.exists(), f"Missing {DATASET_SUMMARY_JSON}"
    with open(DATASET_SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["project_name"] == "DocForensics AI"
    assert summary["total_samples"] > 0
    assert summary["clean_source_documents"] > 0
    assert summary["synthetic_dataset"]["manipulation_categories_supported"] == 10
    
    # Check all 10 manipulation categories exist in summary
    for cat in MANIPULATION_CATEGORIES:
        assert cat in summary["synthetic_dataset"]["manipulation_category_counts"]
        assert summary["synthetic_dataset"]["manipulation_category_counts"][cat] > 0
