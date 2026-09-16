"""
DocForensics AI — Master Dataset Builder (Phase 3)
Builds a high-quality multi-category synthetic document tampering dataset covering
all 10 distinct manipulation types, alongside public research benchmarks.

Maintains strict directory separation:
data/
├── raw/public/
├── synthetic/
│   ├── originals/
│   ├── tampered/
│   ├── masks/
│   └── metadata.csv
├── train/ (images, masks)
├── val/ (images, masks)
└── test/ (images, masks)
"""

import os
import shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2

from config import (
    BASE_DIR,
    RAW_DATA_DIR,
    RAW_PUBLIC_DIR,
    SYNTHETIC_DIR,
    SYNTHETIC_ORIGINALS_DIR,
    SYNTHETIC_TAMPERED_DIR,
    SYNTHETIC_MASKS_DIR,
    SYNTHETIC_METADATA_CSV,
    TRAIN_DATA_DIR,
    VAL_DATA_DIR,
    TEST_DATA_DIR,
    METADATA_CSV,
    cfg,
)
from src.preprocessing.data_collector import (
    build_raw_authentic_corpus,
    GENERATOR_MAP,
)
from src.preprocessing.tampering_engine import (
    apply_manipulation,
    MANIPULATION_CATEGORIES,
)
from src.preprocessing.public_dataset_downloader import (
    download_realtext_v2_subset,
    DATASET_PROVENANCE,
)


def ensure_clean_directories():
    """Ensure all required dataset directories exist."""
    for d in [
        RAW_PUBLIC_DIR,
        SYNTHETIC_DIR,
        SYNTHETIC_ORIGINALS_DIR,
        SYNTHETIC_TAMPERED_DIR,
        SYNTHETIC_MASKS_DIR,
        TRAIN_DATA_DIR / "images",
        TRAIN_DATA_DIR / "masks",
        VAL_DATA_DIR / "images",
        VAL_DATA_DIR / "masks",
        TEST_DATA_DIR / "images",
        TEST_DATA_DIR / "masks",
    ]:
        d.mkdir(parents=True, exist_ok=True)


def build_synthetic_and_master_dataset(
    synthetic_families_per_type: int = 30,
    synthetic_variants_per_family: int = 4,
    num_public_tampered: int = 30,
    num_public_authentic: int = 20,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build 10-category synthetic tampering dataset, organize into data/synthetic/,
    combine with public data, partition by family, and organize data/train, data/val, data/test.
    """
    random.seed(seed)
    np.random.seed(seed)
    ensure_clean_directories()

    split_dir_map = {
        "train": TRAIN_DATA_DIR,
        "val": VAL_DATA_DIR,
        "test": TEST_DATA_DIR,
    }

    # =========================================================================
    # 1. BUILD SYNTHETIC 10-CATEGORY DATASET
    # =========================================================================
    print("=" * 70)
    print("  PHASE 3: Building High-Quality 10-Category Synthetic Dataset   ")
    print("=" * 70)

    raw_authentic_records = build_raw_authentic_corpus(num_families_per_type=synthetic_families_per_type)
    all_synth_families = sorted(list(set(r["family_id"] for r in raw_authentic_records)))
    random.shuffle(all_synth_families)

    # Group Partition for Synthetic Families
    n_syn = len(all_synth_families)
    n_syn_train = int(round(n_syn * train_ratio))
    n_syn_val = int(round(n_syn * val_ratio))

    syn_train_f = set(all_synth_families[:n_syn_train])
    syn_val_f = set(all_synth_families[n_syn_train:n_syn_train + n_syn_val])
    syn_test_f = set(all_synth_families[n_syn_train + n_syn_val:])

    syn_family_split = {}
    for f in syn_train_f:
        syn_family_split[f] = "train"
    for f in syn_val_f:
        syn_family_split[f] = "val"
    for f in syn_test_f:
        syn_family_split[f] = "test"

    synthetic_records = []
    sample_counter = 1

    # Round-robin assigner for 10 manipulation categories
    cat_cycle_idx = 0

    for auth_rec in raw_authentic_records:
        doc_id = auth_rec["doc_id"]
        family_id = auth_rec["family_id"]
        doc_type = auth_rec["doc_type"]
        split = syn_family_split[family_id]
        split_path = split_dir_map[split]

        auth_img_path = Path(auth_rec["file_path"])
        auth_img = Image.open(auth_img_path).convert("RGB")
        w, h = auth_img.size

        # ---------------------------------------------------------------------
        # Authentic Control in Synthetic Dataset
        # ---------------------------------------------------------------------
        auth_sample_id = f"syn_{doc_id}_authentic"
        
        # Save to synthetic repository
        orig_storage_path = SYNTHETIC_ORIGINALS_DIR / f"{doc_id}.png"
        auth_img.save(orig_storage_path, "PNG")

        syn_auth_img_path = SYNTHETIC_TAMPERED_DIR / f"{auth_sample_id}.png"
        syn_auth_mask_path = SYNTHETIC_MASKS_DIR / f"{auth_sample_id}.png"
        auth_img.save(syn_auth_img_path, "PNG")
        zero_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(str(syn_auth_mask_path), zero_mask)

        # Save to split directory
        split_img_p = split_path / "images" / f"{auth_sample_id}.png"
        split_mask_p = split_path / "masks" / f"{auth_sample_id}.png"
        auth_img.save(split_img_p, "PNG")
        cv2.imwrite(str(split_mask_p), zero_mask)

        synthetic_records.append({
            "sample_id": auth_sample_id,
            "source_document_id": doc_id,
            "document_family_id": family_id,
            "doc_type": doc_type,
            "source_type": "synthetic",
            "manipulation_type": "none",
            "is_tampered": 0,
            "bounding_box": "[0, 0, 0, 0]",
            "tampered_pixel_count": 0,
            "tampered_area_percentage": 0.0,
            "original_path": str(orig_storage_path.relative_to(BASE_DIR)),
            "tampered_path": str(split_img_p.relative_to(BASE_DIR)),
            "mask_path": str(split_mask_p.relative_to(BASE_DIR)),
            "width": w,
            "height": h,
            "dataset_split": split,
            "license": "MIT",
        })
        sample_counter += 1

        # ---------------------------------------------------------------------
        # Multi-Category Tampered Variants for this Document
        # ---------------------------------------------------------------------
        for v_idx in range(1, synthetic_variants_per_family + 1):
            chosen_cat = MANIPULATION_CATEGORIES[cat_cycle_idx % len(MANIPULATION_CATEGORIES)]
            cat_cycle_idx += 1

            tamp_img, mask_arr, bbox, manip_used = apply_manipulation(
                auth_img,
                manipulation_type=chosen_cat,
                seed=seed + sample_counter
            )

            tamp_sample_id = f"syn_{doc_id}_v{v_idx}_{manip_used}"

            # Save in synthetic staging
            syn_tamp_img_path = SYNTHETIC_TAMPERED_DIR / f"{tamp_sample_id}.png"
            syn_tamp_mask_path = SYNTHETIC_MASKS_DIR / f"{tamp_sample_id}.png"
            tamp_img.save(syn_tamp_img_path, "PNG")
            cv2.imwrite(str(syn_tamp_mask_path), mask_arr)

            # Save in split directory
            split_img_p = split_path / "images" / f"{tamp_sample_id}.png"
            split_mask_p = split_path / "masks" / f"{tamp_sample_id}.png"
            tamp_img.save(split_img_p, "PNG")
            cv2.imwrite(str(split_mask_p), mask_arr)

            tampered_pixels = int(np.sum(mask_arr > 0))
            area_pct = float(np.round((tampered_pixels / (w * h)) * 100, 4))

            synthetic_records.append({
                "sample_id": tamp_sample_id,
                "source_document_id": doc_id,
                "document_family_id": family_id,
                "doc_type": doc_type,
                "source_type": "synthetic",
                "manipulation_type": manip_used,
                "is_tampered": 1,
                "bounding_box": str(bbox),
                "tampered_pixel_count": tampered_pixels,
                "tampered_area_percentage": area_pct,
                "original_path": str(orig_storage_path.relative_to(BASE_DIR)),
                "tampered_path": str(split_img_p.relative_to(BASE_DIR)),
                "mask_path": str(split_mask_p.relative_to(BASE_DIR)),
                "width": w,
                "height": h,
                "dataset_split": split,
                "license": "MIT",
            })
            sample_counter += 1

    df_synthetic = pd.DataFrame(synthetic_records)
    df_synthetic.to_csv(SYNTHETIC_METADATA_CSV, index=False)
    print(f"[+] Synthetic dataset successfully constructed: {len(df_synthetic)} samples saved to {SYNTHETIC_DIR}")

    # =========================================================================
    # 2. INTEGRATE PUBLIC BENCHMARK (RealText-V2)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  Integrating Public Research Benchmark (RealText-V2)            ")
    print("=" * 70)

    pub_img_dir = RAW_PUBLIC_DIR / "images"
    pub_mask_dir = RAW_PUBLIC_DIR / "masks"

    if not pub_img_dir.exists() or len(list(pub_img_dir.glob("*.*"))) < (num_public_tampered + num_public_authentic):
        download_realtext_v2_subset(num_tampered=num_public_tampered, num_authentic=num_public_authentic)

    pub_img_files = sorted(list(pub_img_dir.glob("*.*")))
    pub_families = [p.stem for p in pub_img_files]
    random.shuffle(pub_families)

    n_pub = len(pub_families)
    n_pub_tr = int(round(n_pub * train_ratio))
    n_pub_vl = int(round(n_pub * val_ratio))

    pub_tr_f = set(pub_families[:n_pub_tr])
    pub_vl_f = set(pub_families[n_pub_tr:n_pub_tr + n_pub_vl])
    pub_ts_f = set(pub_families[n_pub_tr + n_pub_vl:])

    pub_family_split = {}
    for f in pub_tr_f:
        pub_family_split[f] = "train"
    for f in pub_vl_f:
        pub_family_split[f] = "val"
    for f in pub_ts_f:
        pub_family_split[f] = "test"

    public_records = []
    for img_p in pub_img_files:
        stem = img_p.stem
        mask_p = pub_mask_dir / f"{stem}.png"
        if not mask_p.exists():
            continue

        split = pub_family_split[stem]
        split_path = split_dir_map[split]

        dest_img_path = split_path / "images" / f"public_{stem}.png"
        dest_mask_path = split_path / "masks" / f"public_{stem}.png"

        with Image.open(img_p) as im:
            max_dim = 1500
            if max(im.size) > max_dim:
                im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            im.convert("RGB").save(dest_img_path, "PNG")
            w, h = im.size

        mask_arr = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if mask_arr.shape != (h, w):
            mask_arr = cv2.resize(mask_arr, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(dest_mask_path), mask_arr)

        tampered_pixels = int(np.sum(mask_arr > 0))
        is_tampered = 1 if tampered_pixels > 0 else 0
        area_pct = float(np.round((tampered_pixels / (w * h)) * 100, 4))

        # Compute bounding box if tampered
        if is_tampered:
            y_indices, x_indices = np.where(mask_arr > 0)
            bbox = [int(x_indices.min()), int(y_indices.min()), int(x_indices.max()), int(y_indices.max())]
        else:
            bbox = [0, 0, 0, 0]

        rec = {
            "sample_id": f"public_{stem}",
            "source_document_id": stem,
            "document_family_id": f"pub_family_{stem}",
            "doc_type": "public_document",
            "source_type": "public",
            "manipulation_type": "public_realtext_tampered" if is_tampered else "none",
            "is_tampered": is_tampered,
            "bounding_box": str(bbox),
            "tampered_pixel_count": tampered_pixels,
            "tampered_area_percentage": area_pct,
            "original_path": str(img_p.relative_to(BASE_DIR)),
            "tampered_path": str(dest_img_path.relative_to(BASE_DIR)),
            "mask_path": str(dest_mask_path.relative_to(BASE_DIR)),
            "width": w,
            "height": h,
            "dataset_split": split,
            "license": DATASET_PROVENANCE["license"],
        }
        public_records.append(rec)

    df_public = pd.DataFrame(public_records)
    df_master = pd.concat([df_synthetic, df_public]).reset_index(drop=True)
    df_master.to_csv(METADATA_CSV, index=False)

    print(f"[+] Master dataset compiled: {len(df_master)} total samples indexed in {METADATA_CSV}")
    return df_synthetic, df_master


if __name__ == "__main__":
    syn_df, master_df = build_synthetic_and_master_dataset()
    print(syn_df["manipulation_type"].value_counts())
