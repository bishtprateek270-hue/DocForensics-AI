"""
DocForensics AI — Backend Forensic Service & Session Cache (Phase 9)
Manages the single loaded instance of DocForensicsPipeline, coordinates PDF/image ingestion,
caches analysis sessions in memory/temp storage, and generates URLs for frontend asset retrieval.
"""

import os
import sys
import io
import matplotlib
matplotlib.use('Agg')
import uuid
import tempfile
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from config import BASE_DIR, CHECKPOINT_DIR, get_target_device
from src.inference.pipeline import DocForensicsPipeline
from src.consistency.pipeline import ContentConsistencyPipeline
from backend.pdf_utils import render_pdf_page_to_rgb, get_pdf_page_count
from backend.schemas import (
    ForensicAnalysisReport,
    SuspiciousRegionSchema,
    PerformanceLatencySchema,
    VisualAnalysisSchema,
    ContentAnalysisSchema,
    ConsistencyCheckSchema,
    RecordVerificationSchema,
)


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


class ForensicService:
    """
    Singleton service managing the ML pipeline and active analysis sessions.
    """

    _instance: Optional["ForensicService"] = None

    def __init__(self, checkpoint_path: Optional[Path] = None, threshold: float = 0.50):
        self.checkpoint_path = checkpoint_path or (CHECKPOINT_DIR / "dual_stream_best.pth")
        self.threshold = threshold
        self.pipeline: Optional[DocForensicsPipeline] = None
        self.consistency_pipeline = ContentConsistencyPipeline()
        self.sessions: Dict[str, AnalysisSession] = {}
        self.temp_dir = Path(tempfile.mkdtemp(prefix="docforensics_web_"))

    @classmethod
    def get_instance(cls) -> "ForensicService":
        if cls._instance is None:
            cls._instance = ForensicService()
        return cls._instance

    def initialize(self):
        """Loads the Phase 8 DocForensicsPipeline on GPU/CPU once on server startup."""
        if self.pipeline is None:
            print("[*] Initializing ForensicService with Phase 7/8 ML Pipeline...")
            self.pipeline = DocForensicsPipeline(
                checkpoint_path=self.checkpoint_path,
                threshold=self.threshold,
                use_ocr=True,
            )
            print("[+] ForensicService initialized successfully.")

    def process_document(
        self,
        file_bytes: bytes,
        filename: str,
        page_number: int = 1,
    ) -> ForensicAnalysisReport:
        """
        Analyzes an uploaded image or PDF document and registers an analysis session.
        """
        if self.pipeline is None:
            self.initialize()

        session_id = str(uuid.uuid4())
        ext = Path(filename).suffix.lower()
        if not ext:
            ext = ".png"

        total_pages = 1
        # 1. Decode Image or Render PDF
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

        # Save temporary image for pipeline analysis
        temp_doc_path = self.temp_dir / f"{session_id}_input.png"
        Image.fromarray(img_rgb).save(temp_doc_path)

        # 2. Run Forensic Pipeline
        raw_report = self.pipeline.analyze_document(
            image_path=temp_doc_path,
            document_id=session_id,
            save_visual_report=True,
            output_dir=self.temp_dir,
        )

        # 3. Create Heatmap and Overlay Images in Memory
        # Generate Jet Heatmap RGB image
        input_tensor, _ = self.pipeline.preprocess_image(img_rgb)
        with torch.no_grad():
            with torch.amp.autocast(self.pipeline.device.type, enabled=(self.pipeline.device.type == "cuda")):
                logits = self.pipeline.model(input_tensor)
                prob_map_512 = torch.sigmoid(logits)[0, 0].cpu().numpy()
        prob_heatmap = cv2.resize(prob_map_512, (w0, h0), interpolation=cv2.INTER_LINEAR)
        prob_heatmap = np.clip(prob_heatmap, 0.0, 1.0)

        # Heatmap colormap (Jet)
        heatmap_uint8 = (prob_heatmap * 255.0).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Overlay image
        overlay_rgb = img_rgb.copy()
        crops_map: Dict[int, np.ndarray] = {}

        for r in raw_report["suspicious_regions"]:
            x1, y1, x2, y2 = r["bbox"]
            cv2.rectangle(overlay_rgb, (x1, y1), (x2, y2), (255, 0, 0), max(2, int(w0 / 350)))

            # Save crop in crops_map
            crop_rgb = img_rgb[y1:y2, x1:x2]
            if crop_rgb.size > 0:
                crops_map[r["region_id"]] = crop_rgb

            label = f"Region #{r['region_id']} ({r['mean_tampering_score']:.2f})"
            font_scale = max(0.4, w0 / 1200)
            thickness = max(1, int(w0 / 600))
            (w_lbl, h_lbl), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(overlay_rgb, (x1, max(0, y1 - h_lbl - 8)), (x1 + w_lbl + 6, max(0, y1)), (255, 0, 0), -1)
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

        # 4. Register Session in Cache
        session = AnalysisSession(
            session_id=session_id,
            filename=filename,
            file_type=file_type,
            page_number=page_number,
            total_pages=total_pages,
            original_image_rgb=img_rgb,
            heatmap_rgb=heatmap_rgb,
            overlay_rgb=overlay_rgb,
            report_data=raw_report,
            crops=crops_map,
        )
        self.sessions[session_id] = session

        # 5. Build Sanitized Pydantic Response
        regions_schema = []
        for r in raw_report["suspicious_regions"]:
            regions_schema.append(
                SuspiciousRegionSchema(
                    region_id=r["region_id"],
                    bbox=r["bbox"],
                    center=r["center"],
                    pixel_area=r["pixel_area"],
                    percentage_of_document_area=r["percentage_of_document_area"],
                    mean_tampering_score=r["mean_tampering_score"],
                    max_tampering_score=r["max_tampering_score"],
                    ocr_text=r["ocr_text"],
                    ocr_confidence=r["ocr_confidence"],
                    has_associated_text=r["has_associated_text"],
                    region_type=r["region_type"],
                    crop_url=f"/api/analysis/{session_id}/regions/{r['region_id']}",
                )
            )

        latency_schema = PerformanceLatencySchema(**raw_report["performance_latency"])

        # 6. Run Content Consistency Analysis on OCR detections
        ocr_entries_raw = raw_report.get("ocr_entries", [])
        content_res = self.consistency_pipeline.analyze_content(ocr_entries_raw)

        # Build Content Analysis Schema
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

        # Record Verification Schema
        rec_ver = content_res.get("record_verification", {})
        record_verification_schema = RecordVerificationSchema(
            status=rec_ver.get("status", "not_available"),
            source=rec_ver.get("source", "Local System"),
            message=rec_ver.get("message", "No external database configured for this document."),
            mismatches=rec_ver.get("mismatches", None),
        )

        # Visual Analysis Schema
        visual_analysis_schema = VisualAnalysisSchema(
            analysis_status=raw_report["analysis_status"],
            suspicious_region_count=raw_report["suspicious_region_count"],
            highest_tampering_score=raw_report["highest_tampering_score"],
            total_suspicious_area_percent=raw_report["total_suspicious_area_percent"],
        )

        # 7. Conservative Status & Assessment Synthesis
        final_status = raw_report["analysis_status"]
        final_summary = raw_report["assessment_summary"]

        if content_res["status"] == "content_inconsistency_detected":
            if final_status == "no_significant_tampering_evidence_detected":
                final_status = "manual_review_recommended"
                final_summary = f"Content inconsistency detected: {content_res['summary']} (Visual forensic model detected no pixel-level editing artifacts, characteristic of clean screenshot or Inspect-Element manipulation)."
            else:
                final_summary = f"{raw_report['assessment_summary']} Additionally, content inconsistency was detected: {content_res['summary']}"

        report = ForensicAnalysisReport(
            session_id=session_id,
            document_id=session_id,
            filename=filename,
            file_type=file_type,
            page_number=page_number,
            total_pages=total_pages,
            timestamp_utc=raw_report["timestamp_utc"],
            analysis_status=final_status,
            assessment_summary=final_summary,
            model_architecture="dual_stream_rgb_srm_forensic",
            inference_threshold=self.threshold,
            original_resolution=[w0, h0],
            suspicious_region_count=raw_report["suspicious_region_count"],
            total_suspicious_area_percent=raw_report["total_suspicious_area_percent"],
            highest_tampering_score=raw_report["highest_tampering_score"],
            visual_analysis=visual_analysis_schema,
            content_analysis=content_analysis_schema,
            record_verification=record_verification_schema,
            performance_latency=latency_schema,
            suspicious_regions=regions_schema,
            image_url=f"/api/analysis/{session_id}/image",
            heatmap_url=f"/api/analysis/{session_id}/heatmap",
            overlay_url=f"/api/analysis/{session_id}/overlay",
        )

        return report

    def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        return self.sessions.get(session_id)
