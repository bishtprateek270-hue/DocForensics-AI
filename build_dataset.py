#!/usr/bin/env python3
"""
DocForensics AI — Master Dataset Pipeline Runner
Integrates both:
1. Public Research Benchmark: RealText-V2 (ACM MM 2026 GenText-Forensics Challenge, CC-BY-NC 4.0)
2. Synthetic DocForensics Benchmark (MIT License)

Partitions data strictly by document family (70% train, 15% val, 15% test),
validates all image-mask pairs, and exports summary and visual audit plots.
"""

from src.preprocessing.dataset_builder import build_full_dataset
from src.preprocessing.dataset_validator import validate_dataset
from src.preprocessing.dataset_summary import generate_dataset_summary
from src.preprocessing.visualize_dataset import visualize_dataset_samples
from config import cfg


def main():
    print("=================================================================")
    print("   DocForensics AI — Master Dataset Preparation & Provenance     ")
    print("=================================================================")

    # 1. Build Integrated Multi-Source Dataset
    df = build_full_dataset(
        synthetic_families_per_type=30,           # 30 families * 5 doc types = 150 families
        synthetic_variants_per_family=3,          # 1 authentic + 3 tampered = 600 synthetic samples
        num_public_tampered=30,                   # 30 real tampered documents from RealText-V2
        num_public_authentic=20,                  # 20 real authentic documents from RealText-V2
        train_ratio=cfg.dataset.train_ratio,      # 70%
        val_ratio=cfg.dataset.val_ratio,          # 15%
        test_ratio=cfg.dataset.test_ratio,        # 15%
        seed=cfg.dataset.seed,
    )

    # 2. Validate Dataset Integrity & Leakage
    val_res = validate_dataset()
    if val_res["status"] != "PASSED":
        print("[!] Dataset validation failed!")
        return

    # 3. Generate Summary Report
    generate_dataset_summary()

    # 4. Generate Visual Verification Grid
    visualize_dataset_samples()

    print("[SUCCESS] Phase 2 Dataset Pipeline completed successfully.")


if __name__ == "__main__":
    main()
