"""
DocForensics AI — Master Dataset Builder & Partitioner
Integrates both:
1. Public Research Benchmark: RealText-V2 (ACM MM 2026 GenText-Forensics Challenge, CC-BY-NC 4.0)
2. Synthetic DocForensics Benchmark (MIT License)

Organizes image-mask pairs into data/train, data/val, data/test with strict
family-level group partitioning to prevent source-document leakage across splits.
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
    SYNTHETIC_DATA_DIR,
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
    apply_tampering,
)
from src.preprocessing.public_dataset_downloader import (
    download_realtext_v2_subset,
    PUBLIC_RAW_DIR,
    PUBLIC_IMAGES_DIR,
    PUBLIC_MASKS_DIR,
    DATASET_PROVENANCE,
)


def ensure_clean_split_dirs():
    """Ensure train, val, test directories exist and are clean."""
    for split_dir in [TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR]:
        img_d = split_dir / "images"
        msk_d = split_dir / "masks"
        if img_d.exists():
            shutil.rmtree(img_d)
        if msk_d.exists():
            shutil.rmtree(msk_d)
        img_d.mkdir(parents=True, exist_ok=True)
        msk_d.mkdir(parents=True, exist_ok=True)


def build_full_dataset(
    synthetic_families_per_type: int = 30,
    synthetic_variants_per_family: int = 3,
    num_public_tampered: int = 30,
    num_public_authentic: int = 20,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build integrated dataset combining Public Research Benchmark and Synthetic Benchmark.
    """
    random.seed(seed)
    np.random.seed(seed)
    ensure_clean_split_dirs()

    split_dir_map = {
        "train": TRAIN_DATA_DIR,
        "val": VAL_DATA_DIR,
        "test": TEST_DATA_DIR,
    }

    all_records = []

    # =========================================================================
    # PART 1: PUBLIC DATASET (RealText-V2 Benchmark)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PART 1: Integrating Public Dataset (RealText-V2 ACM MM 2026)")
    print("=" * 70)

    # Check if public files exist, otherwise download
    public_imgs = list(PUBLIC_IMAGES_DIR.glob("*.*")) if PUBLIC_IMAGES_DIR.exists() else []
    if len(public_imgs) < (num_public_tampered + num_public_authentic):
        download_realtext_v2_subset(num_tampered=num_public_tampered, num_authentic=num_public_authentic)

    # Load public dataset samples
    public_img_files = sorted(list(PUBLIC_IMAGES_DIR.glob("*.*")))
    public_families = [p.stem for p in public_img_files]
    random.shuffle(public_families)

    n_pub = len(public_families)
    n_pub_train = int(round(n_pub * train_ratio))
    n_pub_val = int(round(n_pub * val_ratio))

    pub_train_f = set(public_families[:n_pub_train])
    pub_val_f = set(public_families[n_pub_train:n_pub_train + n_pub_val])
    pub_test_f = set(public_families[n_pub_train + n_pub_val:])

    pub_family_split = {}
    for f in pub_train_f:
        pub_family_split[f] = "train"
    for f in pub_val_f:
        pub_family_split[f] = "val"
    for f in pub_test_f:
        pub_family_split[f] = "test"

    for img_p in public_img_files:
        stem = img_p.stem
        mask_p = PUBLIC_MASKS_DIR / f"{stem}.png"
        if not mask_p.exists():
            continue

        split = pub_family_split[stem]
        split_path = split_dir_map[split]

        # Standardize and save to split
        dest_img_path = split_path / "images" / f"public_{stem}.png"
        dest_mask_path = split_path / "masks" / f"public_{stem}.png"

        with Image.open(img_p) as im:
            # Resize very large scans to manageable max dimension (e.g. max 1500px) while maintaining aspect ratio
            max_dim = 1500
            if max(im.size) > max_dim:
                im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            im.convert("RGB").save(dest_img_path, "PNG")
            w, h = im.size

        # Read and resize mask to match
        mask_arr = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if mask_arr.shape != (h, w):
            mask_arr = cv2.resize(mask_arr, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(dest_mask_path), mask_arr)

        tampered_pixels = int(np.sum(mask_arr > 0))
        is_tampered = 1 if tampered_pixels > 0 else 0
        area_ratio = float(tampered_pixels / (w * h))

        all_records.append({
            "sample_id": f"public_{stem}",
            "document_id": stem,
            "document_family_id": f"pub_family_{stem}",
            "doc_type": "public_document",
            "data_source": "public_realtext_v2",
            "is_synthetic": 0,
            "license": DATASET_PROVENANCE["license"],
            "is_tampered": is_tampered,
            "tampering_type": "public_realtext_tampered" if is_tampered else "none",
            "image_path": str(dest_img_path.relative_to(cfg.base_dir)),
            "mask_path": str(dest_mask_path.relative_to(cfg.base_dir)),
            "width": w,
            "height": h,
            "tampered_pixel_count": tampered_pixels,
            "tampered_area_ratio": area_ratio,
            "split": split,
        })

    print(f"[+] Public RealText-V2 dataset processed: {len(public_img_files)} samples partitioned into train/val/test.")

    # =========================================================================
    # PART 2: SYNTHETIC DATASET (DocForensics Benchmark)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PART 2: Integrating Synthetic Dataset (DocForensics-Synthetic)")
    print("=" * 70)

    raw_authentic_records = build_raw_authentic_corpus(num_families_per_type=synthetic_families_per_type)

    all_synth_families = sorted(list(set(r["family_id"] for r in raw_authentic_records)))
    random.shuffle(all_synth_families)

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

    sample_counter = 1
    for auth_rec in raw_authentic_records:
        doc_id = auth_rec["doc_id"]
        family_id = auth_rec["family_id"]
        doc_type = auth_rec["doc_type"]
        split = syn_family_split[family_id]
        split_path = split_dir_map[split]

        auth_img_path = Path(auth_rec["file_path"])
        auth_img = Image.open(auth_img_path).convert("RGB")
        w, h = auth_img.size

        # 1. Authentic Sample
        auth_sample_name = f"synthetic_{doc_id}_authentic"
        dest_img_path = split_path / "images" / f"{auth_sample_name}.png"
        dest_mask_path = split_path / "masks" / f"{auth_sample_name}.png"

        auth_img.save(dest_img_path, "PNG")
        zero_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(str(dest_mask_path), zero_mask)

        all_records.append({
            "sample_id": auth_sample_name,
            "document_id": doc_id,
            "document_family_id": family_id,
            "doc_type": doc_type,
            "data_source": "synthetic_docforensics",
            "is_synthetic": 1,
            "license": "MIT",
            "is_tampered": 0,
            "tampering_type": "none",
            "image_path": str(dest_img_path.relative_to(cfg.base_dir)),
            "mask_path": str(dest_mask_path.relative_to(cfg.base_dir)),
            "width": w,
            "height": h,
            "tampered_pixel_count": 0,
            "tampered_area_ratio": 0.0,
            "split": split,
        })
        sample_counter += 1

        # 2. Tampered Variants
        technique_choices = ["copy_move", "splicing", "erasure_inpainting", "text_alteration"]
        for v_idx in range(1, synthetic_variants_per_family + 1):
            chosen_tech = technique_choices[(v_idx - 1) % len(technique_choices)]
            tamp_img, mask_arr, tech_used = apply_tampering(
                auth_img,
                technique=chosen_tech,
                seed=seed + sample_counter
            )

            tamp_sample_name = f"synthetic_{doc_id}_tampered_v{v_idx}_{tech_used}"
            tamp_img_path = split_path / "images" / f"{tamp_sample_name}.png"
            tamp_mask_path = split_path / "masks" / f"{tamp_sample_name}.png"

            tamp_img.save(tamp_img_path, "PNG")
            cv2.imwrite(str(tamp_mask_path), mask_arr)

            tampered_pixels = int(np.sum(mask_arr > 0))
            area_ratio = float(tampered_pixels / (w * h))

            all_records.append({
                "sample_id": tamp_sample_name,
                "document_id": doc_id,
                "document_family_id": family_id,
                "doc_type": doc_type,
                "data_source": "synthetic_docforensics",
                "is_synthetic": 1,
                "license": "MIT",
                "is_tampered": 1,
                "tampering_type": tech_used,
                "image_path": str(tamp_img_path.relative_to(cfg.base_dir)),
                "mask_path": str(tamp_mask_path.relative_to(cfg.base_dir)),
                "width": w,
                "height": h,
                "tampered_pixel_count": tampered_pixels,
                "tampered_area_ratio": area_ratio,
                "split": split,
            })
            sample_counter += 1

    df = pd.DataFrame(all_records)
    df.to_csv(METADATA_CSV, index=False)
    print(f"\n[+] Master dataset successfully constructed: {len(df)} total samples indexed in {METADATA_CSV}")
    return df


if __name__ == "__main__":
    df = build_full_dataset()
    print(df["data_source"].value_counts())
