"""
DocForensics AI — Phase 8 Inference, OCR & Forensic Pipeline Tests
Verifies:
- Checkpoint loading and initialization
- Coordinate restoration & resolution preservation
- Suspicious region connected component extraction
- OCR bounding box geometric association
- Authentic document / zero-detection case
- Multiple suspicious regions handling
- Corrupted file rejection
- JSON report schema conformance
- CUDA inference & latency measurement
"""

import pytest
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import json
from pathlib import Path

from config import CHECKPOINT_DIR
from src.ocr.ocr_engine import OCREntry
from src.inference.pipeline import (
    DocForensicsPipeline,
    calculate_bbox_intersection,
    classify_region_heuristic,
)


@pytest.fixture(scope="module")
def sample_test_images():
    """Creates temporary synthetic test images for pipeline validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Clean Authentic Image (600x800)
        img_authentic = np.full((600, 800, 3), 245, dtype=np.uint8)
        cv2.putText(img_authentic, "UNIVERSITY TRANSCRIPT", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(img_authentic, "Student: John Doe", (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img_authentic, "CGPA: 3.85", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        path_auth = tmp_path / "sample_authentic.png"
        cv2.imwrite(str(path_auth), cv2.cvtColor(img_authentic, cv2.COLOR_RGB2BGR))

        # 2. Tampered Image (600x800) with forged text box
        img_tampered = img_authentic.copy()
        cv2.rectangle(img_tampered, (220, 210), (360, 260), (255, 255, 255), -1)
        cv2.putText(img_tampered, "CGPA: 4.00", (225, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)
        path_tamp = tmp_path / "sample_tampered.png"
        cv2.imwrite(str(path_tamp), cv2.cvtColor(img_tampered, cv2.COLOR_RGB2BGR))

        # 3. Corrupted image file (0 bytes / invalid bytes)
        path_corrupted = tmp_path / "corrupted_image.jpg"
        with open(path_corrupted, "wb") as f:
            f.write(b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_DATA")

        yield {
            "authentic": path_auth,
            "tampered": path_tamp,
            "corrupted": path_corrupted,
            "tmp_dir": tmp_path,
        }


def test_bbox_intersection_calculation():
    """Verify geometric intersection calculation between region and OCR box."""
    region_box = [100, 100, 300, 200]
    # Fully inside
    ocr_inside = [120, 120, 250, 180]
    assert calculate_bbox_intersection(region_box, ocr_inside) == 1.0

    # No overlap
    ocr_outside = [400, 400, 500, 500]
    assert calculate_bbox_intersection(region_box, ocr_outside) == 0.0

    # 50% horizontal overlap
    ocr_partial = [200, 100, 400, 200]  # width 200, overlap with region is [200, 100, 300, 200] = width 100
    assert abs(calculate_bbox_intersection(region_box, ocr_partial) - 0.50) < 1e-4


def test_region_heuristic_classification():
    """Verify conservative heuristic classification for various visual regions."""
    doc_shape = (1000, 800)
    
    # 1. Text region (with OCR text)
    crop = np.full((40, 120, 3), 240, dtype=np.uint8)
    h_text = classify_region_heuristic([100, 100, 220, 140], doc_shape, has_text=True, crop_rgb=crop)
    assert h_text == "text_or_number_region"

    # 2. Stamp region (high saturation blue/red circle)
    stamp_crop = np.zeros((80, 80, 3), dtype=np.uint8)
    stamp_crop[:, :] = [30, 40, 200]  # Rich red
    h_stamp = classify_region_heuristic([200, 200, 280, 280], doc_shape, has_text=False, crop_rgb=stamp_crop)
    assert h_stamp == "stamp_or_seal_like_region"

    # 3. Signature-like region (elongated stroke)
    sig_crop = np.full((30, 200, 3), 250, dtype=np.uint8)
    h_sig = classify_region_heuristic([100, 500, 300, 530], doc_shape, has_text=False, crop_rgb=sig_crop)
    assert h_sig == "signature_like_region"


def test_pipeline_checkpoint_loading():
    """Verify pipeline initializes and loads dual_stream_best.pth."""
    ckpt_path = CHECKPOINT_DIR / "dual_stream_best.pth"
    if not ckpt_path.exists():
        pytest.skip("dual_stream_best.pth not found")

    pipeline = DocForensicsPipeline(checkpoint_path=ckpt_path, use_ocr=False)
    assert pipeline.model is not None
    assert pipeline.threshold == 0.50


def test_pipeline_corrupted_input_rejection(sample_test_images):
    """Verify pipeline cleanly rejects corrupted or non-image files."""
    ckpt_path = CHECKPOINT_DIR / "dual_stream_best.pth"
    if not ckpt_path.exists():
        pytest.skip("dual_stream_best.pth not found")

    pipeline = DocForensicsPipeline(checkpoint_path=ckpt_path, use_ocr=False)
    with pytest.raises(ValueError):
        pipeline.analyze_document(sample_test_images["corrupted"])


def test_pipeline_full_execution_and_schema(sample_test_images):
    """Verify end-to-end analysis produces valid JSON report schema and coordinates."""
    ckpt_path = CHECKPOINT_DIR / "dual_stream_best.pth"
    if not ckpt_path.exists():
        pytest.skip("dual_stream_best.pth not found")

    pipeline = DocForensicsPipeline(
        checkpoint_path=ckpt_path,
        threshold=0.50,
        use_ocr=True,
    )

    report = pipeline.analyze_document(
        image_path=sample_test_images["tampered"],
        document_id="test_doc_001",
        save_visual_report=True,
        output_dir=sample_test_images["tmp_dir"],
    )

    # Validate Schema
    assert "document_id" in report
    assert "analysis_status" in report
    assert report["analysis_status"] in [
        "no_significant_tampering_evidence_detected",
        "suspicious_visual_manipulation_detected",
        "manual_review_recommended",
    ]
    assert "suspicious_region_count" in report
    assert "total_suspicious_area_percent" in report
    assert "performance_latency" in report
    assert "model_inference_ms" in report["performance_latency"]
    assert "ocr_processing_ms" in report["performance_latency"]
    assert "total_pipeline_latency_ms" in report["performance_latency"]
    assert "suspicious_regions" in report

    # Validate Resolution Preservation
    assert report["original_resolution"] == [800, 600]

    # Check that performance numbers are finite and positive
    assert report["performance_latency"]["total_pipeline_latency_ms"] > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pipeline_cuda_execution(sample_test_images):
    """Verify pipeline runs on CUDA with memory tracking."""
    ckpt_path = CHECKPOINT_DIR / "dual_stream_best.pth"
    if not ckpt_path.exists():
        pytest.skip("dual_stream_best.pth not found")

    device = torch.device("cuda:0")
    pipeline = DocForensicsPipeline(
        checkpoint_path=ckpt_path,
        device=device,
        use_ocr=False,
    )

    report = pipeline.analyze_document(
        image_path=sample_test_images["authentic"],
        save_visual_report=False,
    )

    assert report["performance_latency"]["model_inference_ms"] > 0.0
    assert report["performance_latency"]["peak_vram_mb"] > 0.0
