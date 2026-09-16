"""
DocForensics AI — FastAPI Backend Application (Phase 9)
Provides production-grade REST API endpoints for document tampering detection,
suspicious region localization, OCR association, and visual asset streaming.
"""

import os
import sys
import io
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from config import BASE_DIR, CHECKPOINT_DIR
from src.training.gpu_utils import get_hardware_diagnostics, get_vram_usage
from backend.schemas import (
    HealthResponse,
    ForensicAnalysisReport,
    ErrorResponse,
)
from backend.service import ForensicService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes ML model and OCR pipeline once on startup."""
    print("=" * 75)
    print("    DocForensics AI — FastAPI Server Starting Up                     ")
    print("=" * 75)
    service = ForensicService.get_instance()
    service.initialize()
    yield
    print("[*] DocForensics AI Server Shutting Down.")


app = FastAPI(
    title="DocForensics AI API",
    description="Production REST API for Document Tampering Detection, Localization, and Forensic Reporting.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def get_health_status():
    """Returns real-time system, GPU, VRAM, and model status."""
    diag = get_hardware_diagnostics()
    vram = get_vram_usage()
    ckpt_path = CHECKPOINT_DIR / "dual_stream_best.pth"

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        cuda_available=diag["cuda_available"],
        gpu_name=diag["gpu_name"] if diag["cuda_available"] else "CPU",
        vram_total_gb=vram.get("total_gb", 0.0),
        vram_allocated_mb=vram.get("allocated_mb", 0.0),
        model_checkpoint_loaded=ckpt_path.name if ckpt_path.exists() else "None",
        ocr_engine="EasyOCR (CRAFT + CRNN with PyTorch CUDA)",
    )


@app.post(
    "/api/analyze",
    response_model=ForensicAnalysisReport,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Forensic Analysis"],
)
async def analyze_document(
    file: UploadFile = File(..., description="Document file (JPG, JPEG, PNG, PDF)"),
    page_number: int = Form(1, description="Page number for PDF documents"),
):
    """
    Accepts an uploaded document file, executes dual-stream forensic inference,
    localizes suspicious regions, associates OCR text evidence, and returns a structured report.
    """
    filename = file.filename or "uploaded_document.png"
    ext = Path(filename).suffix.lower()

    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pdf"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported formats: JPG, JPEG, PNG, PDF",
        )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )

        service = ForensicService.get_instance()
        report = service.process_document(
            file_bytes=content,
            filename=filename,
            page_number=page_number,
        )
        return report

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forensic analysis failed: {str(e)}",
        )


@app.get(
    "/api/analysis/{session_id}",
    response_model=ForensicAnalysisReport,
    responses={404: {"model": ErrorResponse}},
    tags=["Forensic Analysis"],
)
async def get_analysis_report(session_id: str):
    """Retrieves structured JSON analysis report for an existing session."""
    service = ForensicService.get_instance()
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis session not found.")

    # Reconstruct Pydantic report from session
    regions_schema = []
    raw = session.report_data
    for r in raw["suspicious_regions"]:
        regions_schema.append({
            "region_id": r["region_id"],
            "bbox": r["bbox"],
            "center": r["center"],
            "pixel_area": r["pixel_area"],
            "percentage_of_document_area": r["percentage_of_document_area"],
            "mean_tampering_score": r["mean_tampering_score"],
            "max_tampering_score": r["max_tampering_score"],
            "ocr_text": r["ocr_text"],
            "ocr_confidence": r["ocr_confidence"],
            "has_associated_text": r["has_associated_text"],
            "region_type": r["region_type"],
            "crop_url": f"/api/analysis/{session_id}/regions/{r['region_id']}",
        })

    return ForensicAnalysisReport(
        session_id=session_id,
        document_id=session_id,
        filename=session.filename,
        file_type=session.file_type,
        page_number=session.page_number,
        total_pages=session.total_pages,
        timestamp_utc=raw["timestamp_utc"],
        analysis_status=raw["analysis_status"],
        assessment_summary=raw["assessment_summary"],
        model_architecture="dual_stream_rgb_srm_forensic",
        inference_threshold=raw["inference_threshold"],
        original_resolution=raw["original_resolution"],
        suspicious_region_count=raw["suspicious_region_count"],
        total_suspicious_area_percent=raw["total_suspicious_area_percent"],
        highest_tampering_score=raw["highest_tampering_score"],
        performance_latency=raw["performance_latency"],
        suspicious_regions=regions_schema,
        image_url=f"/api/analysis/{session_id}/image",
        heatmap_url=f"/api/analysis/{session_id}/heatmap",
        overlay_url=f"/api/analysis/{session_id}/overlay",
    )


@app.get("/api/analysis/{session_id}/image", tags=["Assets"])
async def get_original_image(session_id: str):
    """Streams original document image or rendered PDF page as PNG."""
    service = ForensicService.get_instance()
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    buf = io.BytesIO()
    Image.fromarray(session.original_image_rgb).save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/analysis/{session_id}/heatmap", tags=["Assets"])
async def get_heatmap_image(session_id: str):
    """Streams probability heatmap PNG."""
    service = ForensicService.get_instance()
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    buf = io.BytesIO()
    Image.fromarray(session.heatmap_rgb).save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/analysis/{session_id}/overlay", tags=["Assets"])
async def get_overlay_image(session_id: str):
    """Streams localization overlay PNG with bounding boxes and region labels."""
    service = ForensicService.get_instance()
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    buf = io.BytesIO()
    Image.fromarray(session.overlay_rgb).save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/analysis/{session_id}/regions/{region_id}", tags=["Assets"])
async def get_region_crop(session_id: str, region_id: int):
    """Streams high-resolution cropped PNG of a specific suspicious region."""
    service = ForensicService.get_instance()
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    crop_rgb = session.crops.get(region_id)
    if crop_rgb is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Region crop not found.")

    buf = io.BytesIO()
    Image.fromarray(crop_rgb).save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
