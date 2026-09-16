"""
DocForensics AI — Dataset Summary & Provenance Profiler (Phase 3)
Generates comprehensive statistics and metrics across the 10-category synthetic dataset
and public research benchmark, and saves master summary to data/dataset_summary.json.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from config import METADATA_CSV, SYNTHETIC_METADATA_CSV, DATASET_SUMMARY_JSON, cfg
from src.preprocessing.tampering_engine import MANIPULATION_CATEGORIES


def generate_dataset_summary(metadata_path: Path = METADATA_CSV, save_json: bool = True) -> dict:
    """Compute and format comprehensive statistical summary of the dataset."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file {metadata_path} not found.")

    df = pd.read_csv(metadata_path)

    split_col = "dataset_split" if "dataset_split" in df.columns else "split"
    family_col = "document_family_id" if "document_family_id" in df.columns else "family_id"
    source_col = "source_type" if "source_type" in df.columns else "data_source"

    total_samples = len(df)
    genuine_df = df[df["is_tampered"] == 0]
    tampered_df = df[df["is_tampered"] == 1]

    # Clean source documents
    clean_originals_count = len(df["source_document_id"].unique())

    # Synthetic subset
    syn_df = df[df[source_col].str.contains("synthetic", case=False, na=False)]
    syn_tampered = syn_df[syn_df["is_tampered"] == 1]
    syn_genuine = syn_df[syn_df["is_tampered"] == 0]

    # Public subset
    pub_df = df[df[source_col].str.contains("public", case=False, na=False)]
    pub_tampered = pub_df[pub_df["is_tampered"] == 1]
    pub_genuine = pub_df[pub_df["is_tampered"] == 0]

    # Manipulation category counts (all 10 categories)
    manipulation_counts = {}
    for cat in MANIPULATION_CATEGORIES:
        count = int((syn_tampered["manipulation_type"] == cat).sum())
        manipulation_counts[cat] = count

    # Also include public manipulation types
    for p_type in pub_tampered["manipulation_type"].unique():
        count = int((pub_tampered["manipulation_type"] == p_type).sum())
        manipulation_counts[p_type] = count

    # Split distribution
    split_counts = df[split_col].value_counts().to_dict()
    split_families = df.groupby(split_col)[family_col].nunique().to_dict()

    # Mask area statistics
    area_col = "tampered_area_percentage" if "tampered_area_percentage" in df.columns else "tampered_area_ratio"
    if len(tampered_df) > 0:
        area_vals = tampered_df[area_col]
        # If ratio (0-1), scale to %
        if area_vals.max() <= 1.0:
            area_vals = area_vals * 100

        mask_stats = {
            "mean_tampered_area_pct": float(np.round(area_vals.mean(), 4)),
            "median_tampered_area_pct": float(np.round(area_vals.median(), 4)),
            "min_tampered_area_pct": float(np.round(area_vals.min(), 4)),
            "max_tampered_area_pct": float(np.round(area_vals.max(), 4)),
            "std_tampered_area_pct": float(np.round(area_vals.std(), 4)),
        }
    else:
        mask_stats = {}

    summary = {
        "project_name": "DocForensics AI",
        "phase": "Phase 3: High-Quality Synthetic Document Tampering Generation",
        "dataset_name": "DocForensics-Synthetic-Benchmark-v1",
        "total_samples": total_samples,
        "total_masks": total_samples,
        "clean_source_documents": clean_originals_count,
        "total_document_families": int(df[family_col].nunique()),
        "synthetic_dataset": {
            "total_samples": len(syn_df),
            "authentic_samples": len(syn_genuine),
            "tampered_samples": len(syn_tampered),
            "clean_source_originals": int(syn_df["source_document_id"].nunique()),
            "manipulation_categories_supported": len(MANIPULATION_CATEGORIES),
            "manipulation_category_counts": {
                cat: manipulation_counts.get(cat, 0) for cat in MANIPULATION_CATEGORIES
            },
        },
        "public_dataset": {
            "total_samples": len(pub_df),
            "authentic_samples": len(pub_genuine),
            "tampered_samples": len(pub_tampered),
            "source_name": "RealText-V2 (ACM MM 2026 GenText-Forensics Grand Challenge)",
            "license": "CC-BY-NC 4.0",
        },
        "split_distribution": {
            "train": {
                "samples": int(split_counts.get("train", 0)),
                "percentage": float(np.round(split_counts.get("train", 0) / total_samples * 100, 2)),
                "families": int(split_families.get("train", 0)),
            },
            "val": {
                "samples": int(split_counts.get("val", 0)),
                "percentage": float(np.round(split_counts.get("val", 0) / total_samples * 100, 2)),
                "families": int(split_families.get("val", 0)),
            },
            "test": {
                "samples": int(split_counts.get("test", 0)),
                "percentage": float(np.round(split_counts.get("test", 0) / total_samples * 100, 2)),
                "families": int(split_families.get("test", 0)),
            },
        },
        "all_manipulation_counts": manipulation_counts,
        "mask_area_statistics": mask_stats,
    }

    if save_json:
        DATASET_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(DATASET_SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[+] Phase 3 dataset summary saved to {DATASET_SUMMARY_JSON}")

    # Formatted terminal printing
    print("\n" + "=" * 75)
    print("             DOCFORENSICS AI — PHASE 3 DATASET SUMMARY REPORT          ")
    print("=" * 75)
    print(f" Total Master Samples       : {total_samples:,}")
    print(f" Total Clean Source Docs    : {clean_originals_count:,}")
    print(f" Total Generated Masks      : {total_samples:,}")
    print(f" Total Document Families    : {df[family_col].nunique()}")
    print("-" * 75)
    print(" 1. Synthetic Tampering Dataset (10 Categories):")
    print(f"    • Total Synthetic Samples : {len(syn_df)} ({len(syn_genuine)} Clean, {len(syn_tampered)} Tampered)")
    for cat in MANIPULATION_CATEGORIES:
        cnt = manipulation_counts.get(cat, 0)
        print(f"      - {cat:<30} : {cnt:>4} samples")
    print("-" * 75)
    print(" 2. Public Research Benchmark (RealText-V2):")
    print(f"    • Total Public Samples    : {len(pub_df)} ({len(pub_genuine)} Authentic, {len(pub_tampered)} Tampered)")
    print(f"    • License                 : CC-BY-NC 4.0")
    print("-" * 75)
    print(" 3. Split Distribution (Strict Disjoint Document Families):")
    for s_name in ["train", "val", "test"]:
        s_data = summary["split_distribution"][s_name]
        print(f"    • {s_name.upper():<5} Set : {s_data['samples']:>4} images ({s_data['percentage']:>5.1f}%) | {s_data['families']:>3} Document Families")
    print("-" * 75)
    print(" 4. Tampered Mask Area Statistics:")
    print(f"    • Average Tampered Area   : {mask_stats.get('mean_tampered_area_pct', 0):.2f}%")
    print(f"    • Min Tampered Area       : {mask_stats.get('min_tampered_area_pct', 0):.4f}%")
    print(f"    • Max Tampered Area       : {mask_stats.get('max_tampered_area_pct', 0):.2f}%")
    print(f"    • Standard Deviation      : {mask_stats.get('std_tampered_area_pct', 0):.2f}%")
    print("=" * 75 + "\n")

    return summary


if __name__ == "__main__":
    generate_dataset_summary()
