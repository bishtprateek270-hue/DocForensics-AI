"""
DocForensics AI — Dataset Summary & Provenance Profiler
Computes detailed metrics across Public (RealText-V2) and Synthetic (DocForensics) datasets.
Exports master JSON profile to data/dataset_summary.json.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

from config import METADATA_CSV, DATASET_SUMMARY_JSON, cfg


def generate_dataset_summary(metadata_path: Path = METADATA_CSV, save_json: bool = True) -> dict:
    """Compute and format statistical summary of the integrated dataset."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file {metadata_path} not found.")

    df = pd.read_csv(metadata_path)

    total_images = len(df)
    genuine_df = df[df["is_tampered"] == 0]
    tampered_df = df[df["is_tampered"] == 1]

    # Source breakdown
    sources_summary = {}
    for src in df["data_source"].unique():
        sub_df = df[df["data_source"] == src]
        src_gen = sub_df[sub_df["is_tampered"] == 0]
        src_tamp = sub_df[sub_df["is_tampered"] == 1]
        
        sources_summary[src] = {
            "total_samples": len(sub_df),
            "percentage_of_dataset": float(np.round(len(sub_df) / total_images * 100, 2)),
            "genuine_samples": len(src_gen),
            "tampered_samples": len(src_tamp),
            "unique_families": int(sub_df["document_family_id"].nunique()),
            "license": str(sub_df["license"].iloc[0]),
            "splits": sub_df["split"].value_counts().to_dict(),
        }

    # Split statistics
    split_counts = df["split"].value_counts().to_dict()
    split_families = df.groupby("split")["document_family_id"].nunique().to_dict()

    # Document type distribution
    doc_type_counts = df["doc_type"].value_counts().to_dict()

    # Tampering technique distribution
    technique_counts = df["tampering_type"].value_counts().to_dict()

    # Mask statistics
    if len(tampered_df) > 0:
        area_ratios = tampered_df["tampered_area_ratio"] * 100
        mask_stats = {
            "mean_tampered_area_pct": float(np.round(area_ratios.mean(), 2)),
            "median_tampered_area_pct": float(np.round(area_ratios.median(), 2)),
            "min_tampered_area_pct": float(np.round(area_ratios.min(), 2)),
            "max_tampered_area_pct": float(np.round(area_ratios.max(), 2)),
            "std_tampered_area_pct": float(np.round(area_ratios.std(), 2)),
        }
    else:
        mask_stats = {}

    summary = {
        "project_name": "DocForensics AI",
        "dataset_name": "DocForensics-Integrated-Benchmark-v1",
        "total_samples": total_images,
        "total_document_families": int(df["document_family_id"].nunique()),
        "data_sources": sources_summary,
        "class_distribution": {
            "genuine": len(genuine_df),
            "genuine_pct": float(np.round(len(genuine_df) / total_images * 100, 2)),
            "tampered": len(tampered_df),
            "tampered_pct": float(np.round(len(tampered_df) / total_images * 100, 2)),
        },
        "split_distribution": {
            "train_samples": split_counts.get("train", 0),
            "train_pct": float(np.round(split_counts.get("train", 0) / total_images * 100, 2)),
            "train_families": split_families.get("train", 0),
            "val_samples": split_counts.get("val", 0),
            "val_pct": float(np.round(split_counts.get("val", 0) / total_images * 100, 2)),
            "val_families": split_families.get("val", 0),
            "test_samples": split_counts.get("test", 0),
            "test_pct": float(np.round(split_counts.get("test", 0) / total_images * 100, 2)),
            "test_families": split_families.get("test", 0),
        },
        "document_types": doc_type_counts,
        "tampering_techniques": technique_counts,
        "mask_statistics": mask_stats,
    }

    if save_json:
        with open(DATASET_SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[+] Dataset summary saved to {DATASET_SUMMARY_JSON}")

    # Print clean formatted report
    print("\n" + "=" * 70)
    print("             DOCFORENSICS AI — MASTER DATASET PROVENANCE REPORT        ")
    print("=" * 70)
    print(f" Total Master Samples       : {total_images:,}")
    print(f" Total Document Families    : {df['document_family_id'].nunique()}")
    print("-" * 70)
    print(" Data Source Breakdown:")
    for src_name, sinfo in sources_summary.items():
        print(f"  [{src_name}]")
        print(f"    • Samples     : {sinfo['total_samples']} ({sinfo['percentage_of_dataset']}%) | Families: {sinfo['unique_families']}")
        print(f"    • Composition : {sinfo['genuine_samples']} Genuine, {sinfo['tampered_samples']} Tampered")
        print(f"    • License     : {sinfo['license']}")
        print(f"    • Splits      : {sinfo['splits']}")
    print("-" * 70)
    print(f" Overall Genuine Documents  : {summary['class_distribution']['genuine']:,} ({summary['class_distribution']['genuine_pct']}%)")
    print(f" Overall Tampered Documents : {summary['class_distribution']['tampered']:,} ({summary['class_distribution']['tampered_pct']}%)")
    print("-" * 70)
    print(" Split Breakdown (Disjoint Document Families):")
    print(f"  • Train Set               : {summary['split_distribution']['train_samples']:,} images ({summary['split_distribution']['train_pct']}%) across {summary['split_distribution']['train_families']} families")
    print(f"  • Validation Set          : {summary['split_distribution']['val_samples']:,} images ({summary['split_distribution']['val_pct']}%) across {summary['split_distribution']['val_families']} families")
    print(f"  • Test Set                : {summary['split_distribution']['test_samples']:,} images ({summary['split_distribution']['test_pct']}%) across {summary['split_distribution']['test_families']} families")
    print("-" * 70)
    print(" Tampering Techniques:")
    for tech, count in technique_counts.items():
        print(f"  • {tech:<28} : {count:,} samples")
    print("=" * 70 + "\n")

    return summary


if __name__ == "__main__":
    generate_dataset_summary()
