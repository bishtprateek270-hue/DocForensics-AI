"""
Unit & Integration Tests for Multi-Source Dataset Preparation and Validation
"""
import pytest
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

from config import (
    BASE_DIR,
    METADATA_CSV,
    TRAIN_DATA_DIR,
    VAL_DATA_DIR,
    TEST_DATA_DIR,
    DATASET_SUMMARY_JSON,
)
from src.preprocessing.dataset_validator import validate_dataset


def test_metadata_file_exists():
    assert METADATA_CSV.exists(), "metadata.csv does not exist"


def test_dataset_validator_passes():
    res = validate_dataset()
    assert res["status"] == "PASSED", f"Validator failed with errors: {res.get('errors')}"
    assert res["error_count"] == 0
    assert res["total_samples"] > 0


def test_public_and_synthetic_sources_exist():
    df = pd.read_csv(METADATA_CSV)
    sources = set(df["data_source"].unique())
    assert "public_realtext_v2" in sources, "Missing public_realtext_v2 in dataset sources"
    assert "synthetic_docforensics" in sources, "Missing synthetic_docforensics in dataset sources"


def test_zero_data_leakage():
    df = pd.read_csv(METADATA_CSV)
    train_families = set(df[df["split"] == "train"]["document_family_id"])
    val_families = set(df[df["split"] == "val"]["document_family_id"])
    test_families = set(df[df["split"] == "test"]["document_family_id"])

    assert len(train_families.intersection(val_families)) == 0, "Leakage between train and val!"
    assert len(train_families.intersection(test_families)) == 0, "Leakage between train and test!"
    assert len(val_families.intersection(test_families)) == 0, "Leakage between val and test!"


def test_mask_dimensions_and_binary_values():
    df = pd.read_csv(METADATA_CSV)
    # Sample from both public and synthetic
    sample_df = pd.concat([
        df[df["data_source"] == "public_realtext_v2"].head(10),
        df[df["data_source"] == "synthetic_docforensics"].head(10)
    ])
    for _, row in sample_df.iterrows():
        img_p = BASE_DIR / row["image_path"]
        mask_p = BASE_DIR / row["mask_path"]

        with Image.open(img_p) as img:
            w, h = img.size
        
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f"Could not read mask: {mask_p}"
        mh, mw = mask.shape
        assert (w, h) == (mw, mh), f"Dimension mismatch: img={(w, h)}, mask={(mw, mh)}"
        
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 255}), f"Mask contains non-binary values: {unique_vals}"
