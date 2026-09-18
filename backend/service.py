"""DocForensics AI — Dual-Specialist Backend Forensic Service & Session Cache.

Manages in-memory instances of:
1. Model A (Phase 7 Physical/Image Forensic Model - locked baseline)
2. Model B (DocTamper Tiny-Text Digital Forensic Model)
3. OCR Text Extraction Engine
4. OCR-Constrained Tiny-Text Post-Processor & Hard-Negative Filter
5. Dual-Specialist Evidence Fusion Engine
6. Content Consistency & Reference Verification Pipelines
7. On-demand PDF Forensic Dossier Generator
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR, CHECKPOINT_DIR, get_target_device
from src.consistency.pipeline import ContentConsistencyPipeline
from src.forensics.evidence_fusion import EvidenceFusionEngine
from src.forensics.tinytext_postprocessor import TinyTextPostProcessor
from src.models.dual_stream_forensics import DualStreamForensicNet
from src.ocr.ocr_engine import get_ocr_engine
from backend.pdf_report_generator import generate_forensic_pdf_bytes
from backend.pdf_utils import get_pdf_page_count, render_pdf_page_to_rgb
from backend.schemas import (
    ConsistencyCheckSchema,
    ContentAnalysisSchema,
    EvidenceSummarySchema,
    ForensicAnalysisReport,
    PerformanceLatencySchema,
    RecordVerificationSchema,
    SuspiciousRegionSchema,
    VisualAnalysisSchema,
)


import hashlib

EXPECTED_HASH_A = "376b074999a95c37a33cd0ca9295532ec4b96936919fc84982c4a7230f456d15"
EXPECTED_HASH_B = "58d1456efba3e141041c74ebe392e39339715512155549f3e0bc0eb5249b8238"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 digest of a checkpoint file."""
    if not filepath.exists():
        return ""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class AnalysisSession:
    """Holds artifacts and structured data for a single document analysis session."""

    def __init__(
        self,
        session_id: str,
        filename: str,
        file_type: str,
        page_number: int,
        total_pages: int,
        original_image_rgb: np.ndarray,
        heatmap_rgb: np.ndarray,
        overlay_rgb: np.ndarray,
        report_data: Dict[str, Any],
        crops: Dict[int, np.ndarray],
        pdf_bytes: Optional[bytes] = None,
    ):
        self.session_id = session_id
        self.filename = filename
        self.file_type = file_type
        self.page_number = page_number
        self.total_pages = total_pages
        self.original_image_rgb = original_image_rgb
        self.heatmap_rgb = heatmap_rgb
        self.overlay_rgb = overlay_rgb
        self.report_data = report_data
        self.crops = crops
        self.pdf_bytes = pdf_bytes


class ForensicService:
    """Singleton service managing dual ML specialists, OCR, and active sessions."""

    _instance: Optional["ForensicService"] = None

    def __init__(self):
        self.device = get_target_device()
        self.ckpt_p7_path = CHECKPOINT_DIR / "dual_stream_best.pth"
        self.ckpt_tt_path = CHECKPOINT_DIR / "dual_stream_doctamper_tinytext_best.pth"
        
        self.model_a_physical: Optional[DualStreamForensicNet] = None
        self.model_b_tinytext: Optional[DualStreamForensicNet] = None
        self.ocr_engine = None
        
        self.tinytext_postprocessor = TinyTextPostProcessor(
            threshold=0.45,
            min_component_area=15,
            ocr_expansion_ratio=0.15,
            max_ocr_distance=35.0,
            merge_horizontal_distance=30,
            merge_vertical_distance=10,
            require_ocr_association=True,
        )
        self.fusion_engine: Optional[EvidenceFusionEngine] = None
        self.consistency_pipeline = ContentConsistencyPipeline()
        
        self.sessions: Dict[str, AnalysisSession] = {}
        self.temp_dir = Path(tempfile.mkdtemp(prefix="docforensics_dual_"))

    @classmethod
    def get_instance(cls) -> "ForensicService":
        if cls._instance is None:
            cls._instance = ForensicService()
        return cls._instance

    def initialize(self):
        """Pre-loads Model A, Model B, and OCR Engine into memory at startup with integrity checks."""
        if self.model_a_physical is None:
            print("[*] Initializing Dual-Specialist Forensic Pipeline...")

            # Verify Model A SHA256
            if not self.ckpt_p7_path.exists():
                raise FileNotFoundError(f"Model A checkpoint not found at: {self.ckpt_p7_path}")
            hash_a = compute_sha256(self.ckpt_p7_path)
            if hash_a != EXPECTED_HASH_A:
                raise RuntimeError(
                    f"Model A integrity verification FAILED! Expected {EXPECTED_HASH_A[:12]}..., got {hash_a[:12]}..."
                )
            print(f"[+] Model A SHA256 verified ({hash_a[:12]}...).")

            # Verify Model B SHA256
            if not self.ckpt_tt_path.exists():
                raise FileNotFoundError(f"Model B checkpoint not found at: {self.ckpt_tt_path}")
            hash_b = compute_sha256(self.ckpt_tt_path)
            if hash_b != EXPECTED_HASH_B:
                raise RuntimeError(
                    f"Model B integrity verification FAILED! Expected {EXPECTED_HASH_B[:12]}..., got {hash_b[:12]}..."
                )
            print(f"[+] Model B SHA256 verified ({hash_b[:12]}...).")

            # 1. Load Model A (Phase 7 Physical Forensics Baseline)
            print("[*] Loading Model A (Physical Forensics)...")
            self.model_a_physical = DualStreamForensicNet(pretrained_backbone=False)
            ckpt_a = torch.load(str(self.ckpt_p7_path), map_location="cpu", weights_only=False)
            state_dict_a = ckpt_a.get("model_state_dict", ckpt_a)
            self.model_a_physical.load_state_dict(state_dict_a, strict=False)
            self.model_a_physical.to(self.device)
            self.model_a_physical.eval()

            # 2. Load Model B (DocTamper Tiny-Text Digital Specialist)
            print("[*] Loading Model B (Tiny-Text Forensics)...")
            self.model_b_tinytext = DualStreamForensicNet(pretrained_backbone=False)
            ckpt_b = torch.load(str(self.ckpt_tt_path), map_location="cpu", weights_only=False)
            state_dict_b = ckpt_b.get("model_state_dict", ckpt_b)
            self.model_b_tinytext.load_state_dict(state_dict_b, strict=False)
            self.model_b_tinytext.to(self.device)
            self.model_b_tinytext.eval()

            # 3. Load OCR Engine
            print("[*] Initializing OCR Engine...")
            self.ocr_engine = get_ocr_engine(use_gpu=(self.device.type == "cuda"))

            # 4. Instantiate Evidence Fusion Engine
            self.fusion_engine = EvidenceFusionEngine(
                model_a_physical=self.model_a_physical,
                model_b_tinytext=self.model_b_tinytext,
                tinytext_postprocessor=self.tinytext_postprocessor,
                iou_fusion_threshold=0.20,
            )
            print("[+] Dual-Specialist Forensic Pipeline initialized and verified successfully.")

    def process_document(
        self,
        file_bytes: bytes,
        filename: str,
        page_number: int = 1,
    ) -> ForensicAnalysisReport:
        """Executes full dual-specialist forensic analysis and returns sanitized report."""
        if self.model_a_physical is None or self.model_b_tinytext is None:
            self.initialize()

        t_pipeline_start = time.perf_counter()
        session_id = str(uuid.uuid4())
        ext = Path(filename).suffix.lower() or ".png"
        total_pages = 1

        # 1. Ingest Document
        if ext == ".pdf":
            file_type = "PDF"
            img_rgb, total_pages, w0, h0 = render_pdf_page_to_rgb(file_bytes, page_number=page_number, dpi=150)
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            file_type = ext.replace(".", "").upper()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Could not decode image file; image may be corrupted.")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h0, w0 = img_rgb.shape[:2]
        else:
            raise ValueError(f"Unsupported file type '{ext}'. Supported: JPG, PNG, PDF")

        # 2. OCR Execution
        t_ocr_start = time.perf_counter()
        ocr_entries = []
        if self.ocr_engine is not None:
            ocr_entries, _ = self.ocr_engine.extract_text(img_rgb)
        t_ocr_ms = (time.perf_counter() - t_ocr_start) * 1000.0

        # Format OCR results for post-processor and consistency engine
        ocr_dicts = [
            {"bbox": e.bbox, "text": e.text, "confidence": e.confidence}
            for e in ocr_entries
        ]

        # 3. Content Consistency Pipeline
        content_res = self.consistency_pipeline.analyze_content(ocr_dicts)
        content_findings = [
            c for c in content_res.get("checks", [])
            if c.get("status") in ["mismatch", "impossible_value"]
        ]

        # 4. Neural Model Inference (Dual-Specialist)
        t_infer_start = time.perf_counter()
        img_512 = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_norm = (img_512.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor_in = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                out_a = self.model_a_physical(tensor_in)
                out_b = self.model_b_tinytext(tensor_in)
            prob_a_512 = torch.sigmoid(out_a)[0, 0].cpu().numpy()
            prob_b_512 = torch.sigmoid(out_b)[0, 0].cpu().numpy()
        t_infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

        # 5. Post-Processing & Evidence Fusion
        t_post_start = time.perf_counter()
        prob_a_orig = cv2.resize(prob_a_512, (w0, h0), interpolation=cv2.INTER_LINEAR)
        prob_b_orig = cv2.resize(prob_b_512, (w0, h0), interpolation=cv2.INTER_LINEAR)

        physical_regions = self.fusion_engine.extract_physical_regions(prob_a_orig, threshold=0.50)
        text_regions = self.tinytext_postprocessor.process(prob_b_orig, ocr_results=ocr_dicts)
        fused_regions = self.fusion_engine.fuse_overlapping_regions(physical_regions, text_regions)
        t_post_ms = (time.perf_counter() - t_post_start) * 1000.0

        total_latency_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        # 6. Generate Heatmap & Visual Overlay
        # Combined heatmap (maximum response across specialists)
        combined_prob = np.maximum(prob_a_orig, prob_b_orig)
        heatmap_uint8 = (np.clip(combined_prob, 0.0, 1.0) * 255.0).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        overlay_rgb = img_rgb.copy()
        crops_map: Dict[int, np.ndarray] = {}
        suspicious_regions_schema: List[SuspiciousRegionSchema] = []

        for idx, r in enumerate(fused_regions, start=1):
            rx, ry, rw, rh = r["bbox"]
            x1 = max(0, int(rx))
            y1 = max(0, int(ry))
            x2 = min(w0, int(rx + rw))
            y2 = min(h0, int(ry + rh))
            area_px = (x2 - x1) * (y2 - y1)
            pct_area = (area_px / max(1, w0 * h0)) * 100.0

            sources = r.get("evidence_sources", ["Visual Specialist"])
            is_dual = len(sources) > 1
            is_text_only = ("Tiny-Text" in sources[0]) if not is_dual else False

            # Bounding box color: Blue for Physical, Magenta for Tiny-Text, Purple for Both
            if is_dual:
                box_color = (138, 43, 226)    # BlueViolet
                reg_type = "Multi-Specialist Finding"
            elif is_text_only:
                box_color = (217, 70, 239)   # Fuchsia
                reg_type = "Text-Region Visual Pattern"
            else:
                box_color = (37, 99, 235)     # Blue
                reg_type = "Physical / Splicing Trace"

            cv2.rectangle(overlay_rgb, (x1, y1), (x2, y2), box_color, max(2, int(w0 / 400)))

            # Save Crop
            crop_rgb = img_rgb[y1:y2, x1:x2]
            if crop_rgb.size > 0:
                crops_map[idx] = crop_rgb

            score = r.get("physical_evidence_score") or r.get("text_evidence_score") or 0.50
            if is_dual and r.get("physical_evidence_score") and r.get("text_evidence_score"):
                score = max(r["physical_evidence_score"], r["text_evidence_score"])

            label = f"#{idx} ({score:.2f})"
            font_scale = max(0.4, w0 / 1200)
            thickness = max(1, int(w0 / 600))
            (w_lbl, h_lbl), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(overlay_rgb, (x1, max(0, y1 - h_lbl - 8)), (x1 + w_lbl + 6, max(0, y1)), box_color, -1)
            cv2.putText(
                overlay_rgb,
                label,
                (x1 + 3, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

            suspicious_regions_schema.append(
                SuspiciousRegionSchema(
                    region_id=idx,
                    bbox=[x1, y1, x2, y2],
                    center=[round((x1 + x2) / 2.0, 1), round((y1 + y2) / 2.0, 1)],
                    pixel_area=int(area_px),
                    percentage_of_document_area=round(pct_area, 3),
                    mean_tampering_score=round(float(score), 4),
                    max_tampering_score=round(float(score), 4),
                    ocr_text=r.get("associated_ocr_text", ""),
                    ocr_confidence=0.90 if r.get("associated_ocr_text") else 0.0,
                    has_associated_text=bool(r.get("associated_ocr_text")),
                    region_type=reg_type,
                    evidence_sources=sources,
                    evidence_score=round(float(score), 4),
                    explanation=r.get("explanation"),
                    crop_url=f"/api/analysis/{session_id}/regions/{idx}",
                )
            )

        # 7. Evidence Summary & Synthesis
        has_phys = len(physical_regions) > 0
        has_text = len(text_regions) > 0
        has_incons = len(content_findings) > 0
        has_ref = False

        if has_phys or has_text or has_incons:
            analysis_status = "suspicious_visual_manipulation_detected"
            assessment_summary = "Potential visual manipulation evidence or content inconsistency detected."
        else:
            analysis_status = "no_significant_tampering_evidence_detected"
            assessment_summary = "No significant visual manipulation evidence detected across dual forensic specialists."

        evidence_summary = EvidenceSummarySchema(
            physical_visual_evidence=has_phys,
            digital_text_visual_evidence=has_text,
            content_inconsistency=has_incons,
            reference_mismatch=has_ref,
            physical_region_count=len(physical_regions),
            text_region_count=len(text_regions),
            fused_region_count=len(fused_regions),
            status_message=assessment_summary,
        )

        content_checks_schema = [
            ConsistencyCheckSchema(
                check_id=c["check_id"],
                check_name=c["check_name"],
                displayed_value=str(c.get("displayed_value", "")),
                calculated_value=str(c.get("calculated_value", "")),
                status=c["status"],
                explanation=c["explanation"],
            )
            for c in content_res.get("checks", [])
        ]
        content_analysis_schema = ContentAnalysisSchema(
            document_type=content_res["document_type"],
            document_type_confidence=content_res["document_type_confidence"],
            status=content_res["status"],
            summary=content_res["summary"],
            checks=content_checks_schema,
            extracted_fields=content_res.get("extracted_fields", {}),
        )

        record_verification_schema = RecordVerificationSchema(
            status="not_available",
            source="Local System",
            message="No external database configured for this document.",
        )

        total_susp_area = sum(r.percentage_of_document_area for r in suspicious_regions_schema)
        highest_score = max([r.evidence_score for r in suspicious_regions_schema], default=0.0)

        visual_analysis_schema = VisualAnalysisSchema(
            analysis_status=analysis_status,
            suspicious_region_count=len(suspicious_regions_schema),
            highest_tampering_score=round(highest_score, 4),
            total_suspicious_area_percent=round(total_susp_area, 3),
        )

        latency_schema = PerformanceLatencySchema(
            model_inference_ms=round(t_infer_ms, 2),
            ocr_processing_ms=round(t_ocr_ms, 2),
            post_processing_ms=round(t_post_ms, 2),
            total_pipeline_latency_ms=round(total_latency_ms, 2),
            peak_vram_mb=round(peak_vram_mb, 1),
        )

        report = ForensicAnalysisReport(
            session_id=session_id,
            document_id=session_id,
            filename=filename,
            file_type=file_type,
            page_number=page_number,
            total_pages=total_pages,
            timestamp_utc=datetime.utcnow().isoformat() + "Z",
            analysis_status=analysis_status,
            assessment_summary=assessment_summary,
            model_architecture="dual_specialist_fusion_pipeline",
            inference_threshold=0.50,
            original_resolution=[w0, h0],
            suspicious_region_count=len(suspicious_regions_schema),
            total_suspicious_area_percent=round(total_susp_area, 3),
            highest_tampering_score=round(highest_score, 4),
            suspicious_regions=suspicious_regions_schema,
            visual_analysis=visual_analysis_schema,
            content_analysis=content_analysis_schema,
            record_verification=record_verification_schema,
            evidence_summary=evidence_summary,
            performance_latency=latency_schema,
            image_url=f"/api/analysis/{session_id}/image",
            heatmap_url=f"/api/analysis/{session_id}/heatmap",
            overlay_url=f"/api/analysis/{session_id}/overlay",
            pdf_report_url=f"/api/analysis/{session_id}/pdf-report",
        )

        # 8. Generate PDF Dossier
        pdf_bytes = generate_forensic_pdf_bytes(
            report_data=report.model_dump(),
            overlay_image_rgb=overlay_rgb,
            heatmap_image_rgb=heatmap_rgb,
        )

        # 9. Register in Cache
        session = AnalysisSession(
            session_id=session_id,
            filename=filename,
            file_type=file_type,
            page_number=page_number,
            total_pages=total_pages,
            original_image_rgb=img_rgb,
            heatmap_rgb=heatmap_rgb,
            overlay_rgb=overlay_rgb,
            report_data=report.model_dump(),
            crops=crops_map,
            pdf_bytes=pdf_bytes,
        )
        self.sessions[session_id] = session

        return report

    def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        return self.sessions.get(session_id)
