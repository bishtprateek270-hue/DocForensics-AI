"""
DocForensics AI — Phase 8 Forensic Pipeline Demonstration & Benchmarking
Runs the complete End-to-End Forensic Pipeline on representative document samples:
  1. Authentic Control
  2. Text Replacement
  3. Number / Date Replacement
  4. Photo Replacement
  5. Signature Manipulation
  6. Local Inpainting
  7. Intra Copy-Move
  8. Public RealText-V2
Generates structured JSON reports, 3-panel visual composites, and cropped suspicious regions.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BASE_DIR,
    CHECKPOINT_DIR,
    REPORTS_DIR,
    OUTPUT_DIR,
    METADATA_CSV,
    cfg,
    get_target_device,
)
from src.inference.pipeline import DocForensicsPipeline


def run_phase8_comprehensive_demo() -> Dict[str, Any]:
    """
    Executes the Phase 8 pipeline across 8 representative tampering categories on the test set.
    """
    print("=" * 80)
    print("   DocForensics AI — Phase 8 End-to-End Inference & Forensic Reporting Demo   ")
    print("=" * 80)

    # 1. Initialize Pipeline
    ckpt_path = CHECKPOINT_DIR / "dual_stream_best.pth"
    pipeline = DocForensicsPipeline(
        checkpoint_path=ckpt_path,
        threshold=0.50,
        min_region_pixels=64,
        use_ocr=True,
    )

    # 2. Select Representative Test Samples from Metadata
    df = pd.read_csv(METADATA_CSV)
    test_df = df[df["dataset_split"] == "test"].reset_index(drop=True)

    target_categories = [
        ("authentic_control", "none"),
        ("text_replacement", "text_replacement"),
        ("date_replacement", "date_replacement"),
        ("number_replacement", "number_amount_replacement"),
        ("photo_replacement", "photo_replacement"),
        ("signature_manipulation", "signature_manipulation"),
        ("local_inpainting", "local_inpainting"),
        ("copy_move_intra", "copy_move_intra"),
        ("public_realtext", "public_realtext_tampered"),
    ]

    selected_samples = []
    for label_alias, cat in target_categories:
        cat_matches = test_df[test_df["manipulation_type"] == cat]
        if len(cat_matches) > 0:
            sample_row = cat_matches.iloc[0]
            img_rel = sample_row["tampered_path"] if "tampered_path" in sample_row and pd.notna(sample_row["tampered_path"]) else sample_row["original_path"]
            img_abs = BASE_DIR / img_rel
            selected_samples.append({
                "alias": label_alias,
                "category": cat,
                "sample_id": sample_row["sample_id"],
                "image_path": img_abs,
            })

    print(f"[*] Selected {len(selected_samples)} representative document samples for full analysis.\n")

    demo_results = []
    total_pipeline_times = []
    total_model_times = []
    total_ocr_times = []

    for item in selected_samples:
        s_id = item["sample_id"]
        cat = item["category"]
        img_p = item["image_path"]

        print(f"[*] Analyzing [{item['alias'].upper()}] -> {s_id}")
        report = pipeline.analyze_document(
            image_path=img_p,
            document_id=f"{item['alias']}_{s_id[:8]}",
            save_visual_report=True,
        )

        demo_results.append({
            "alias": item["alias"],
            "manipulation_category": cat,
            "sample_id": s_id,
            "analysis_status": report["analysis_status"],
            "assessment_summary": report["assessment_summary"],
            "suspicious_region_count": report["suspicious_region_count"],
            "total_suspicious_area_percent": report["total_suspicious_area_percent"],
            "highest_tampering_score": report["highest_tampering_score"],
            "regions": report["suspicious_regions"],
            "performance_latency": report["performance_latency"],
            "visual_report_path": report["visual_report_path"],
        })

        lat = report["performance_latency"]
        total_model_times.append(lat["model_inference_ms"])
        total_ocr_times.append(lat["ocr_processing_ms"])
        total_pipeline_times.append(lat["total_pipeline_latency_ms"])

        print(f"    Status: {report['analysis_status']} | Regions: {report['suspicious_region_count']} | Area: {report['total_suspicious_area_percent']}% | Latency: {lat['total_pipeline_latency_ms']:.1f}ms")
        if report["suspicious_regions"]:
            for r in report["suspicious_regions"][:2]:
                print(f"      - Region #{r['region_id']} ({r['region_type']}): OCR='{r['ocr_text'][:40]}' | Score={r['mean_tampering_score']:.2f}")

    # Summary Benchmark
    master_summary = {
        "pipeline_version": "phase8_dual_stream_ocr_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_used": str(ckpt_path),
        "total_documents_analyzed": len(demo_results),
        "mean_latencies_ms": {
            "mean_model_inference_ms": round(float(pd.Series(total_model_times).mean()), 2),
            "mean_ocr_processing_ms": round(float(pd.Series(total_ocr_times).mean()), 2),
            "mean_total_pipeline_ms": round(float(pd.Series(total_pipeline_times).mean()), 2),
        },
        "sample_evaluations": demo_results,
    }

    summary_out = REPORTS_DIR / "phase8_pipeline_benchmark.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[+] Phase 8 Multi-Category Demo Complete!")
    print(f"[+] Summary Benchmark JSON saved to: {summary_out}")
    print(f"[+] Visual Reports saved to: {OUTPUT_DIR / 'phase8_visual_reports'}")
    print(f"[+] Average Pipeline Latency: {master_summary['mean_latencies_ms']['mean_total_pipeline_ms']:.2f} ms")
    print("=" * 80)

    return master_summary


if __name__ == "__main__":
    run_phase8_comprehensive_demo()
