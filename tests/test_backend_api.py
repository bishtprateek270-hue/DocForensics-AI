"""
DocForensics AI — Backend API Test Suite (Phase 9)
Tests:
- GET /api/health endpoint
- POST /api/analyze with valid image (PNG/JPG)
- POST /api/analyze with corrupted file (400 Bad Request)
- POST /api/analyze with PDF rendering
- GET /api/analysis/{id} structured JSON retrieval
- GET /api/analysis/{id}/image, /heatmap, /overlay asset streaming
- Schema conformance & sanitization
"""

import pytest
import io
import numpy as np
import cv2
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    """TestClient fixture with FastAPI lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify /api/health returns 200 with model, GPU, and OCR info."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "cuda_available" in data
    assert "gpu_name" in data
    assert "vram_total_gb" in data
    assert "model_checkpoint_loaded" in data
    assert "EasyOCR" in data["ocr_engine"]


def test_analyze_valid_image(client):
    """Verify /api/analyze processes image and returns full forensic report schema."""
    # Create synthetic test image
    img = np.full((400, 600, 3), 250, dtype=np.uint8)
    cv2.putText(img, "INVOICE #9821", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "AMOUNT: $1,500.00", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("test_invoice.png", buf.getvalue(), "image/png")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 200
    report = response.json()

    # Schema assertions
    assert "session_id" in report
    assert report["filename"] == "test_invoice.png"
    assert report["file_type"] == "PNG"
    assert report["analysis_status"] in [
        "no_significant_tampering_evidence_detected",
        "suspicious_visual_manipulation_detected",
        "manual_review_recommended",
    ]
    assert report["original_resolution"] == [600, 400]
    assert "performance_latency" in report
    assert "image_url" in report
    assert "heatmap_url" in report
    assert "overlay_url" in report

    # Test asset streaming endpoints
    session_id = report["session_id"]
    r_img = client.get(f"/api/analysis/{session_id}/image")
    assert r_img.status_code == 200
    assert r_img.headers["content-type"] == "image/png"

    r_hm = client.get(f"/api/analysis/{session_id}/heatmap")
    assert r_hm.status_code == 200
    assert r_hm.headers["content-type"] == "image/png"

    r_ov = client.get(f"/api/analysis/{session_id}/overlay")
    assert r_ov.status_code == 200
    assert r_ov.headers["content-type"] == "image/png"


def test_analyze_corrupted_file(client):
    """Verify /api/analyze rejects corrupted image with 400 Bad Request."""
    corrupted_bytes = b"CORRUPTED_NON_IMAGE_DATA_12345"
    files = {"file": ("bad_file.png", corrupted_bytes, "image/png")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400
    assert "detail" in response.json()


def test_get_nonexistent_session(client):
    """Verify /api/analysis/{id} returns 404 for unknown session."""
    response = client.get("/api/analysis/nonexistent-session-id-12345")
    assert response.status_code == 404


def test_analyze_pdf_document(client):
    """Verify /api/analyze handles PDF uploads via PyMuPDF rendering."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 72), "CONFIDENTIAL VERIFICATION RECORD", fontsize=16)
    page.insert_text((50, 120), "Date: 2026-09-16", fontsize=12)
    pdf_bytes = doc.write()
    doc.close()

    files = {"file": ("test_doc.pdf", pdf_bytes, "application/pdf")}
    response = client.post("/api/analyze?page_number=1", files=files)
    assert response.status_code == 200
    report = response.json()
    assert report["file_type"] == "PDF"
    assert report["page_number"] == 1
    assert report["total_pages"] == 1
    assert "session_id" in report

