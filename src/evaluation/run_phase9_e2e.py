"""
DocForensics AI — Phase 9 End-to-End System Verification (E2E)
Tests the complete production integration:
FastAPI Backend + Dual-Stream Model + PyMuPDF + EasyOCR + REST APIs + JSON Schemas
"""

import sys
import io
import time
import json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
from fastapi.testclient import TestClient
from backend.main import app
import fitz


def run_phase9_e2e():
    print("=" * 70)
    print("DocForensics AI -- Phase 9 End-to-End Verification Pipeline")
    print("=" * 70)

    with TestClient(app) as client:
        # 1. Check Health & GPU Service
        print("\n[1/6] Verifying API Health & Neural Engine...")
        t0 = time.time()
        res_health = client.get("/api/health")
        assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
        health_data = res_health.json()
        print(f"  [+] Health Status: {health_data['status']}")
        print(f"  [+] CUDA Enabled: {health_data['cuda_available']} ({health_data['gpu_name']})")
        print(f"  [+] VRAM: {health_data['vram_allocated_mb']:.2f} MB / {health_data['vram_total_gb']:.2f} GB")
        print(f"  [+] Model Loaded: {health_data['model_checkpoint_loaded']}")
        print(f"  [+] OCR Engine: {health_data['ocr_engine']}")

        # 2. Test Authentic Document Control
        print("\n[2/6] Analyzing Authentic Document Control...")
        auth_img = np.full((500, 700, 3), 252, dtype=np.uint8)
        cv2.putText(auth_img, "OFFICIAL BILL OF SALE", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
        cv2.putText(auth_img, "SELLER: ACME INDUSTRIAL CORP", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
        cv2.putText(auth_img, "BUYER: GLOBAL LOGISTICS INC", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
        cv2.putText(auth_img, "TRANSACTION TOTAL: $12,450.00", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        cv2.putText(auth_img, "DATE OF ISSUE: 15-JAN-2026", (60, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)

        buf = io.BytesIO()
        Image.fromarray(auth_img).save(buf, format="PNG")
        buf.seek(0)

        t0 = time.time()
        res = client.post("/api/analyze", files={"file": ("authentic_invoice.png", buf.getvalue(), "image/png")})
        assert res.status_code == 200, f"Analysis failed: {res.text}"
        auth_report = res.json()
        print(f"  [+] Session ID: {auth_report['session_id']}")
        print(f"  [+] Status: {auth_report['analysis_status']}")
        print(f"  [+] Max Tampering Score: {auth_report['highest_tampering_score']:.4f}")
        print(f"  [+] Suspicious Regions: {auth_report['suspicious_region_count']}")
        print(f"  [+] Latency: {auth_report['performance_latency']['total_pipeline_latency_ms'] / 1000:.3f}s")

        # 3. Test Text / Amount Manipulation
        print("\n[3/6] Analyzing Spliced / Number Alteration Case...")
        tampered_img = auth_img.copy()
        # Splice tampered amount box with noise artifact
        cv2.rectangle(tampered_img, (55, 235), (450, 275), (240, 240, 240), -1)
        noise = np.random.randint(-35, 35, (40, 395, 3), dtype=np.int16)
        patch = np.clip(tampered_img[235:275, 55:450].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        tampered_img[235:275, 55:450] = patch
        cv2.putText(tampered_img, "TRANSACTION TOTAL: $98,750.00", (60, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        buf = io.BytesIO()
        Image.fromarray(tampered_img).save(buf, format="PNG")
        buf.seek(0)

        res = client.post("/api/analyze", files={"file": ("tampered_amount.png", buf.getvalue(), "image/png")})
        assert res.status_code == 200, f"Analysis failed: {res.text}"
        tamp_report = res.json()
        print(f"  [+] Session ID: {tamp_report['session_id']}")
        print(f"  [+] Status: {tamp_report['analysis_status']}")
        print(f"  [+] Max Tampering Score: {tamp_report['highest_tampering_score']:.4f}")
        print(f"  [+] Suspicious Regions: {tamp_report['suspicious_region_count']}")
        for r in tamp_report['suspicious_regions']:
            print(f"    - Region #{r['region_id']}: Score={r['max_tampering_score']:.3f}, OCR='{r['ocr_text']}' (Conf={r['ocr_confidence']:.2f})")
        print(f"  [+] Latency: {tamp_report['performance_latency']['total_pipeline_latency_ms'] / 1000:.3f}s")

        # 4. Test Inpainting / Erasure Case
        print("\n[4/6] Analyzing Inpainted Erasure Document...")
        inpaint_img = auth_img.copy()
        cv2.rectangle(inpaint_img, (55, 185), (400, 215), (252, 252, 252), -1)

        buf = io.BytesIO()
        Image.fromarray(inpaint_img).save(buf, format="PNG")
        buf.seek(0)

        res = client.post("/api/analyze", files={"file": ("inpainted_erasure.png", buf.getvalue(), "image/png")})
        assert res.status_code == 200, f"Analysis failed: {res.text}"
        inpaint_report = res.json()
        print(f"  [+] Session ID: {inpaint_report['session_id']}")
        print(f"  [+] Status: {inpaint_report['analysis_status']}")
        print(f"  [+] Max Tampering Score: {inpaint_report['highest_tampering_score']:.4f}")
        print(f"  [+] Latency: {inpaint_report['performance_latency']['total_pipeline_latency_ms'] / 1000:.3f}s")

        # 5. Test PDF Page Ingestion & Rasterization
        print("\n[5/6] Analyzing PDF Document Ingestion (PyMuPDF)...")
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 60), "CERTIFICATE OF AUTHENTICITY", fontsize=16)
        page.insert_text((50, 100), "Serial: #DOC-882194-XYZ", fontsize=12)
        page.insert_text((50, 140), "Issued To: Prateek Bisht", fontsize=12)
        page.insert_text((50, 180), "Status: Verified Valid", fontsize=12)
        pdf_bytes = doc.write()
        doc.close()

        res = client.post("/api/analyze?page_number=1", files={"file": ("certificate.pdf", pdf_bytes, "application/pdf")})
        assert res.status_code == 200, f"PDF analysis failed: {res.text}"
        pdf_report = res.json()
        print(f"  [+] Session ID: {pdf_report['session_id']}")
        print(f"  [+] File Type: {pdf_report['file_type']} (Page {pdf_report['page_number']} of {pdf_report['total_pages']})")
        print(f"  [+] Resolution: {pdf_report['original_resolution']}")
        print(f"  [+] Latency: {pdf_report['performance_latency']['total_pipeline_latency_ms'] / 1000:.3f}s")

        # 6. Test Asset Endpoints
        print("\n[6/6] Verifying Visual Asset Streaming Endpoints...")
        session_id = tamp_report["session_id"]
        res_img = client.get(f"/api/analysis/{session_id}/image")
        res_hm = client.get(f"/api/analysis/{session_id}/heatmap")
        res_ov = client.get(f"/api/analysis/{session_id}/overlay")
        assert res_img.status_code == 200 and res_img.headers["content-type"] == "image/png"
        assert res_hm.status_code == 200 and res_hm.headers["content-type"] == "image/png"
        assert res_ov.status_code == 200 and res_ov.headers["content-type"] == "image/png"
        print(f"  [+] Image Stream: {len(res_img.content)} bytes (PNG)")
        print(f"  [+] Heatmap Stream: {len(res_hm.content)} bytes (PNG)")
        print(f"  [+] Overlay Stream: {len(res_ov.content)} bytes (PNG)")

        print("\n" + "=" * 70)
        print("[SUCCESS] PHASE 9 END-TO-END VERIFICATION COMPLETED WITH 100% SUCCESS!")
        print("=" * 70)


if __name__ == "__main__":
    run_phase9_e2e()
