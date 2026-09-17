"""
DocForensics AI — Phase 10 Dataset Auditor & Integrity Validator
Performs comprehensive dataset auditing:
- Samples per split (Train, Val, Test)
- Authentic vs Tampered breakdown
- Synthetic vs Public/Real breakdown
- Manipulation category distribution
- Document family isolation & zero-leakage verification
- Tampered area percentage statistics & quantile buckets
- Image resolution distributions & class imbalance metrics
Outputs: reports/phase10_dataset_audit.json
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import METADATA_CSV, REPORTS_DIR, BASE_DIR


def audit_dataset(metadata_path: Path = METADATA_CSV) -> Dict[str, Any]:
    """Executes full statistical audit and cross-split leakage detection."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    df = pd.read_csv(metadata_path)
    print(f"[*] Loaded metadata with {len(df)} total records from {metadata_path}")

    # 1. Split Distribution
    split_counts = df["dataset_split"].value_counts().to_dict()
    total_samples = len(df)

    # 2. Authentic vs Tampered
    is_tampered_counts = df["is_tampered"].value_counts().to_dict()
    authentic_samples = int(is_tampered_counts.get(0, 0))
    tampered_samples = int(is_tampered_counts.get(1, 0))

    # 3. Source Type Distribution
    source_type_counts = df["source_type"].value_counts().to_dict() if "source_type" in df.columns else {}

    # 4. Manipulation Category Breakdown
    manipulation_counts = df["manipulation_type"].value_counts().to_dict()

    # 5. Per-Split Category Breakdown
    per_split_manipulation = {}
    for split in ["train", "val", "test"]:
        split_df = df[df["dataset_split"] == split]
        per_split_manipulation[split] = {
            "total_samples": len(split_df),
            "authentic_samples": int((split_df["is_tampered"] == 0).sum()),
            "tampered_samples": int((split_df["is_tampered"] == 1).sum()),
            "categories": split_df["manipulation_type"].value_counts().to_dict(),
            "families_count": int(split_df["document_family_id"].nunique()) if "document_family_id" in split_df.columns else 0
        }

    # 6. Document Families & Cross-Split Leakage Check
    train_families: Set[str] = set(df[df["dataset_split"] == "train"]["document_family_id"].dropna().unique())
    val_families: Set[str] = set(df[df["dataset_split"] == "val"]["document_family_id"].dropna().unique())
    test_families: Set[str] = set(df[df["dataset_split"] == "test"]["document_family_id"].dropna().unique())

    train_val_overlap = list(train_families.intersection(val_families))
    train_test_overlap = list(train_families.intersection(test_families))
    val_test_overlap = list(val_families.intersection(test_families))

    has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)

    leakage_status = {
        "zero_leakage_verified": not has_leakage,
        "train_val_overlap_count": len(train_val_overlap),
        "train_test_overlap_count": len(train_test_overlap),
        "val_test_overlap_count": len(val_test_overlap),
        "train_val_leaked_families": train_val_overlap,
        "train_test_leaked_families": train_test_overlap,
        "val_test_leaked_families": val_test_overlap,
    }

    # 7. Tampered Area Statistics (for tampered samples only)
    tampered_df = df[df["is_tampered"] == 1]
    areas = tampered_df["tampered_area_percentage"].values.astype(float) if len(tampered_df) > 0 else np.array([])

    if len(areas) > 0:
        area_stats = {
            "mean_tampered_area_pct": float(round(np.mean(areas), 4)),
            "median_tampered_area_pct": float(round(np.median(areas), 4)),
            "min_tampered_area_pct": float(round(np.min(areas), 4)),
            "max_tampered_area_pct": float(round(np.max(areas), 4)),
            "std_tampered_area_pct": float(round(np.std(areas), 4)),
            "quantiles": {
                "p10": float(round(np.percentile(areas, 10), 4)),
                "p25": float(round(np.percentile(areas, 25), 4)),
                "p50": float(round(np.percentile(areas, 50), 4)),
                "p75": float(round(np.percentile(areas, 75), 4)),
                "p90": float(round(np.percentile(areas, 90), 4)),
            },
            "area_buckets": {
                "tiny_under_0_5pct": int((areas < 0.5).sum()),
                "small_0_5_to_1_5pct": int(((areas >= 0.5) & (areas < 1.5)).sum()),
                "medium_1_5_to_4_0pct": int(((areas >= 1.5) & (areas < 4.0)).sum()),
                "large_above_4_0pct": int((areas >= 4.0).sum()),
            }
        }
    else:
        area_stats = {}

    # 8. Image Resolution Distribution
    resolutions = []
    if "width" in df.columns and "height" in df.columns:
        res_series = df["width"].astype(str) + "x" + df["height"].astype(str)
        resolutions = res_series.value_counts().to_dict()

    # 9. Class Imbalance Ratio
    sample_imbalance_ratio = round(tampered_samples / max(1, authentic_samples), 2)
    # Estimate pixel-level positive ratio
    mean_positive_pixel_ratio = float(round(np.mean(df["tampered_area_percentage"].values) / 100.0, 5))

    # Assemble Audit Report
    audit_report = {
        "audit_version": "Phase 10 Benchmark Audit",
        "total_samples": total_samples,
        "authentic_samples": authentic_samples,
        "tampered_samples": tampered_samples,
        "sample_imbalance_ratio_tampered_to_authentic": sample_imbalance_ratio,
        "mean_positive_pixel_fraction": mean_positive_pixel_ratio,
        "split_distribution": split_counts,
        "source_type_distribution": source_type_counts,
        "manipulation_category_distribution": manipulation_counts,
        "per_split_breakdown": per_split_manipulation,
        "leakage_verification": leakage_status,
        "tampered_area_distribution": area_stats,
        "resolution_distribution": resolutions,
    }

    # Save Audit JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "phase10_dataset_audit.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print(f"[+] Dataset Audit successfully completed!")
    print(f"    - Total Samples: {total_samples}")
    print(f"    - Authentic: {authentic_samples}, Tampered: {tampered_samples}")
    print(f"    - Zero Split Leakage: {leakage_status['zero_leakage_verified']}")
    print(f"    - Report saved to: {report_file}")

    return audit_report


if __name__ == "__main__":
    audit_dataset()
