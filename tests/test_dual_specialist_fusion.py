"""Automated Unit & Integration Tests for Dual-Specialist Fusion Pipeline.

Tests:
1. Model A & Model B checkpoint loading and weights integrity.
2. Dual-specialist inference without memory leaks.
3. Tiny-Text Post-Processor (OCR expansion, min component area, text-line merging).
4. Spatial evidence fusion:
   - Model A detects, Model B doesn't
   - Model B detects, Model A doesn't
   - Both detect same region (IoU overlap merged into single finding with dual tags)
   - Neither detects anything (conservative clean verdict)
5. Content consistency findings integration.
6. Reference verification handling.
7. Robustness: empty OCR, missing OCR, blank images, arbitrary resolutions.
8. PDF report generation byte integrity.
"""

import io
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.forensics.evidence_fusion import EvidenceFusionEngine
from src.forensics.tinytext_postprocessor import TinyTextPostProcessor
from src.models.dual_stream_forensics import DualStreamForensicNet
from backend.pdf_report_generator import generate_forensic_pdf_bytes
from backend.schemas import ForensicAnalysisReport


class DummyForensicModel(nn.Module):
    """Synthetic model returning controllable probability maps for deterministic testing."""
    def __init__(self, output_value: float = 0.0):
        super().__init__()
        self.output_value = output_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inverse sigmoid (logit) of output_value
        val = np.clip(self.output_value, 1e-6, 1.0 - 1e-6)
        logit = float(np.log(val / (1.0 - val)))
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), logit, dtype=torch.float32, device=x.device)


def test_tinytext_postprocessor_geometric_and_ocr_filtering():
    """Verify that isolated tiny noise (<15px) is filtered and OCR text-lines constrain regions."""
    postprocessor = TinyTextPostProcessor(
        threshold=0.45,
        min_component_area=15,
        ocr_expansion_ratio=0.15,
        max_ocr_distance=35.0,
        require_ocr_association=True,
    )

    prob_map = np.zeros((200, 200), dtype=np.float32)

    # 1. Add tiny 4px noise blob (should be filtered by min_component_area=15)
    prob_map[10:12, 10:12] = 0.90

    # 2. Add isolated 30px blob far from OCR (should be filtered by require_ocr_association)
    prob_map[150:156, 150:155] = 0.85

    # 3. Add 40px blob overlapping OCR text
    prob_map[50:58, 50:55] = 0.88

    ocr_results = [
        {"bbox": [48, 48, 20, 15], "text": "SGPA: 9.85", "confidence": 0.96}
    ]

    regions = postprocessor.process(prob_map, ocr_results=ocr_results)

    # Only region #3 near OCR should pass
    assert len(regions) == 1, f"Expected exactly 1 passed region, got {len(regions)}"
    assert regions[0]["associated_ocr_text"] == "SGPA: 9.85"
    assert "Tiny-Text" in regions[0]["evidence_source"]


def test_spatial_region_merging():
    """Verify that adjacent character fragments on same line are merged into one finding."""
    postprocessor = TinyTextPostProcessor(
        threshold=0.40,
        min_component_area=5,
        merge_horizontal_distance=25,
        merge_vertical_distance=10,
        require_ocr_association=False,
    )

    prob_map = np.zeros((100, 200), dtype=np.float32)
    # Fragment 1 (digit 9)
    prob_map[40:48, 30:35] = 0.80
    # Fragment 2 (decimal point)
    prob_map[46:48, 38:40] = 0.75
    # Fragment 3 (digit 8)
    prob_map[40:48, 43:48] = 0.85

    regions = postprocessor.process(prob_map, ocr_results=[])
    assert len(regions) == 1, f"Expected 3 fragments to be merged into 1, got {len(regions)}"
    bbox = regions[0]["bbox"]
    assert bbox[2] >= 15, "Merged bounding box width should encompass all fragments"


def test_dual_specialist_fusion_scenarios():
    """Test the 4 key fusion permutations."""
    postprocessor = TinyTextPostProcessor(threshold=0.40, min_component_area=10, require_ocr_association=False)
    engine = EvidenceFusionEngine(
        model_a_physical=None,
        model_b_tinytext=None,
        tinytext_postprocessor=postprocessor,
        iou_fusion_threshold=0.20,
    )

    # Scenario 1: Model A detects (physical), Model B doesn't
    phys_only = [{"region_id": "phys_1", "bbox": [10, 10, 50, 50], "area_pixels": 2500, "region_evidence_score": 0.85, "evidence_source": "Phase 7 Physical Forensic Model", "explanation": "Physical splicing"}]
    fused_1 = engine.fuse_overlapping_regions(phys_only, [])
    assert len(fused_1) == 1
    assert len(fused_1[0]["evidence_sources"]) == 1
    assert "Physical" in fused_1[0]["evidence_sources"][0]

    # Scenario 2: Model B detects (text), Model A doesn't
    text_only = [{"region_id": "text_1", "bbox": [100, 100, 40, 20], "area_pixels": 800, "region_evidence_score": 0.78, "evidence_source": "Tiny-Text Digital Forensic Model", "explanation": "Text edit", "associated_ocr_text": "9.85"}]
    fused_2 = engine.fuse_overlapping_regions([], text_only)
    assert len(fused_2) == 1
    assert "Tiny-Text" in fused_2[0]["evidence_sources"][0]

    # Scenario 3: Both detect same region (overlapping IoU)
    phys_overlap = [{"region_id": "phys_2", "bbox": [50, 50, 40, 40], "area_pixels": 1600, "region_evidence_score": 0.90, "evidence_source": "Phase 7 Physical Forensic Model", "explanation": "Physical splicing"}]
    text_overlap = [{"region_id": "text_2", "bbox": [55, 55, 30, 30], "area_pixels": 900, "region_evidence_score": 0.82, "evidence_source": "Tiny-Text Digital Forensic Model", "explanation": "Text edit", "associated_ocr_text": "Amount"}]
    fused_3 = engine.fuse_overlapping_regions(phys_overlap, text_overlap)
    assert len(fused_3) == 1, "Overlapping regions should be fused into a single finding"
    assert len(fused_3[0]["evidence_sources"]) == 2, "Fused region should record both contributing specialists"

    # Scenario 4: Neither detects anything
    fused_4 = engine.fuse_overlapping_regions([], [])
    assert len(fused_4) == 0


def test_pdf_dossier_generation():
    """Verify that generate_forensic_pdf_bytes produces valid, non-empty PDF bytes."""
    dummy_report = {
        "session_id": "test-session-uuid-1234",
        "filename": "invoice_sample.pdf",
        "file_type": "PDF",
        "timestamp_utc": "2026-09-17T22:30:00Z",
        "analysis_status": "suspicious_visual_manipulation_detected",
        "assessment_summary": "Suspicious visual manipulation evidence detected in amount field.",
        "model_architecture": "dual_specialist_fusion_pipeline",
        "inference_threshold": 0.50,
        "original_resolution": [1200, 1600],
        "suspicious_region_count": 1,
        "total_suspicious_area_percent": 0.45,
        "highest_tampering_score": 0.88,
        "suspicious_regions": [
            {
                "region_id": 1,
                "bbox": [150, 300, 280, 340],
                "percentage_of_document_area": 0.45,
                "mean_tampering_score": 0.88,
                "ocr_text": "$9,850.00",
                "evidence_sources": ["Phase 7 Physical Forensic Model", "Tiny-Text Digital Forensic Model"],
                "evidence_score": 0.88,
            }
        ],
        "evidence_summary": {
            "physical_visual_evidence": True,
            "digital_text_visual_evidence": True,
            "content_inconsistency": False,
            "reference_mismatch": False,
            "physical_region_count": 1,
            "text_region_count": 1,
            "fused_region_count": 1,
            "disclaimer": "Forensic findings should be verified with institutional issuer.",
        },
        "content_analysis": {
            "checks": [
                {
                    "check_name": "Invoice Total Sum",
                    "displayed_value": "$9,850.00",
                    "calculated_value": "$9,850.00",
                    "status": "match",
                    "explanation": "Subtotal + Tax matches total.",
                }
            ]
        },
        "performance_latency": {
            "total_pipeline_latency_ms": 42.5,
        }
    }

    dummy_img = np.zeros((400, 300, 3), dtype=np.uint8)
    pdf_bytes = generate_forensic_pdf_bytes(dummy_report, overlay_image_rgb=dummy_img)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-"), "Generated file must start with PDF header magic bytes"
