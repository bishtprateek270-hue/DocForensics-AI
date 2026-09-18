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
    """Initializes ML model and OCR pipeline once on startup with integrity verification."""
    print("=" * 75)
    print("    DocForensics AI — FastAPI Server Starting Up (Phase 11 Production) ")
    print("=" * 75)
    service = ForensicService.get_instance()
    service.initialize()
    yield
    print("[*] DocForensics AI Server Shutting Down.")


app = FastAPI(
    title="DocForensics AI API",
    description="Production REST API for Document Tampering Detection, Localization, and Forensic Reporting.",
    version="1.1.0",
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

MAX_UPLOAD_SIZE = 15 * 1024 * 1024  # 15 MB


def validate_file_bytes(content: bytes, filename: str) -> str:
    """Validates file magic bytes and extensions to prevent arbitrary upload."""
    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty (0 bytes).")
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File exceeds maximum allowed size (15 MB).")

    ext = Path(filename).suffix.lower()
    # Magic bytes check
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    elif content.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    elif content.startswith(b"%PDF"):
        return "pdf"
    elif ext in [".jpg", ".jpeg", ".png", ".pdf", ".bmp", ".tif", ".tiff"]:
        return ext.lstrip(".")
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported or corrupted file format. Allowed formats: PNG, JPG, JPEG, PDF"
        )


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def get_health_status():
    """Returns safe, production-grade health and readiness status."""
    service = ForensicService.get_instance()
    models_ready = bool(service.model_a_physical is not None and service.model_b_tinytext is not None)
    
    from src.ocr.ocr_engine import EASYOCR_AVAILABLE
    ocr_ready = bool(EASYOCR_AVAILABLE or service.ocr_engine is not None)

    return HealthResponse(
        status="healthy" if (models_ready and ocr_ready) else "degraded",
        models_ready=models_ready,
        ocr_ready=ocr_ready,
        version="1.0.0",
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
    raw_filename = Path(file.filename or "document.png").name  # Sanitize against path traversal
    try:
        content = await file.read()
        validate_file_bytes(content, raw_filename)

        service = ForensicService.get_instance()
        report = service.process_document(
            file_bytes=content,
            filename=raw_filename,
            page_number=page_number,
        )
        return report

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        # Sanitize internal error message
        err_msg = str(e).split("\n")[0]
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forensic analysis failed during processing: {err_msg}",
        )


@app.post(
    "/api/reference-verify",
    tags=["Reference Verification"],
)
async def reference_verify_document(
    test_file: UploadFile = File(..., description="Document under analysis"),
    reference_file: UploadFile = File(..., description="Trusted original reference document"),
):
    """
    Optional comparison mode comparing test document against a trusted original reference.
    Labeled explicitly as 'Reference comparison' (NOT 'AI tampering detection').
    """
    from src.forensics.reference_verification import compare_with_reference
    try:
        test_bytes = await test_file.read()
        ref_bytes = await reference_file.read()

        validate_file_bytes(test_bytes, test_file.filename or "test.png")
        validate_file_bytes(ref_bytes, reference_file.filename or "ref.png")

        test_img = Image.open(io.BytesIO(test_bytes)).convert("RGB")
        ref_img = Image.open(io.BytesIO(ref_bytes)).convert("RGB")

        result = compare_with_reference(test_img, ref_img)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        err_msg = str(e).split("\n")[0]
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reference verification failed: {err_msg}",
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

    return ForensicAnalysisReport(**session.report_data)


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


@app.get("/api/analysis/{session_id}/pdf-report", tags=["Assets"])
async def get_pdf_report(session_id: str):
    """Streams authoritative forensic PDF dossier for download."""
    service = ForensicService.get_instance()
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    if session.pdf_bytes is None:
        from backend.pdf_report_generator import generate_forensic_pdf_bytes
        session.pdf_bytes = generate_forensic_pdf_bytes(
            report_data=session.report_data,
            overlay_image_rgb=session.overlay_rgb,
            heatmap_image_rgb=session.heatmap_rgb,
        )

    filename = f"DocForensics_Report_{session.filename.rsplit('.', 1)[0]}_{session_id[:8]}.pdf"
    return Response(
        content=session.pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
