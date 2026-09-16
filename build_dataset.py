#!/usr/bin/env python3
"""
DocForensics AI — Phase 3 Master Dataset Pipeline Runner
Executes the high-quality synthetic document tampering dataset generation pipeline,
supporting all 10 distinct manipulation categories, integrates public research benchmarks,
validates mask integrity and zero data-leakage, and generates forensic summary and visual grids.
"""

import sys
from config import cfg
from src.preprocessing.dataset_builder import build_synthetic_and_master_dataset
from src.preprocessing.dataset_validator import validate_dataset
from src.preprocessing.dataset_summary import generate_dataset_summary
from src.preprocessing.visualize_dataset import visualize_dataset_samples


def main():
    print("===========================================================================")
    print("  DocForensics AI — Phase 3 Synthetic Generation & Dataset Pipeline        ")
    print("===========================================================================")

    # 1. Build 10-Category Synthetic & Integrated Master Dataset
    df_synthetic, df_master = build_synthetic_and_master_dataset(
        synthetic_families_per_type=30,      # 30 families * 5 doc types = 150 clean source docs
        synthetic_variants_per_family=4,     # 4 tampered variants per clean doc = 600 tampered + 150 authentic = 750 synthetic samples
        num_public_tampered=30,              # 30 real tampered documents from RealText-V2
        num_public_authentic=20,             # 20 real authentic documents from RealText-V2
        train_ratio=cfg.dataset.train_ratio, # 70%
        val_ratio=cfg.dataset.val_ratio,     # 15%
        test_ratio=cfg.dataset.test_ratio,   # 15%
        seed=cfg.dataset.seed,
    )

    # 2. Comprehensive Quality & Zero Data-Leakage Audit
    val_res = validate_dataset()
    if val_res["status"] != "PASSED":
        print(f"[!] Critical Error: Dataset validation failed with {val_res['error_count']} errors!")
        sys.exit(1)

    # 3. Generate Comprehensive Dataset Provenance & Statistical Summary
    summary = generate_dataset_summary()

    # 4. Generate Multi-Panel Visual Verification Grid (ORIGINAL | TAMPERED | MASK | OVERLAY)
    visualize_dataset_samples()

    print("\n[SUCCESS] Phase 3 Synthetic Dataset Pipeline completed and validated 100% successfully!")


if __name__ == "__main__":
    main()
