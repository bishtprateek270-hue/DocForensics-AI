# DocForensics AI — Multi-Evidence Document Forensic Analysis System

[![Version](https://img.shields.io/badge/Release-v1.0.0--production-blue.svg)](RELEASE_NOTES_v1.0.0.md)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-83%2F83%20Passing%20(100%25)-success.svg)](tests/)

An enterprise-grade, multi-evidence document forensic analysis system that localizes pixel-level tampering and micro-text modifications using dual deep learning specialists, OCR spatial constraints, deterministic semantic verification, and automated PDF forensic dossier generation.

---

## Architecture Overview

```
                          Uploaded Document (PDF / PNG / JPG)
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         ┌─────────────────────┐                   ┌─────────────────────┐
         │       MODEL A       │                   │       MODEL B       │
         │  Physical Forensics │                   │ Tiny-Text Forensics │
         │   (RGB + SRM Net)   │                   │ (DocTamper Special) │
         └──────────┬──────────┘                   └──────────┬──────────┘
                    │                                         │
                    │                                         ▼
                    │                              ┌─────────────────────┐
                    │                              │  OCR Text Corridors │
                    │                              │   & Post-Processor  │
                    │                              └──────────┬──────────┘
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                              ┌─────────────────────┐
                              │ Spatial OCR Engine  │
                              │ (EasyOCR CRAFT+CRNN)│
                              └──────────┬──────────┘
                                         ▼
                              ┌─────────────────────┐
                              │ Content Consistency │
                              │ & Semantic Formulas │
                              └──────────┬──────────┘
                                         ▼
                              ┌─────────────────────┐
                              │ Reference Verification│
                              │ (Optional Auth DB)  │
                              └──────────┬──────────┘
                                         ▼
                              ┌─────────────────────┐
                              │   Evidence Fusion   │
                              │       Engine        │
                              └──────────┬──────────┘
                                         ▼
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         ┌─────────────────────┐                   ┌─────────────────────┐
         │ Next.js Interactive │                   │ Court-Ready ReportLab│
         │  Results Workspace  │                   │  PDF Forensic Report │
         └─────────────────────┘                   └─────────────────────┘
```

---

## Key Features

- **Dual-Specialist Deep Learning:** Combines an RGB+SRM Dual-Stream model for physical splicing with an asymmetric Focal Tversky specialist for micro-text manipulation.
- **6.85x False-Positive Reduction:** Uses OCR-guided 15% text corridor expansion, component thresholding ($\ge 15$ px), and adjacent fragment merging to reduce false-positive regions from 15.16 down to 2.21 per document.
- **Multi-Evidence Channel Separation:** Strict separation of physical visual traces, digital text visual patterns, content consistency findings, and reference verification without synthetic "authenticity scores".
- **Semantic & Arithmetic Consistency:** Re-evaluates invoice arithmetic and semester SGPA formulas to flag pixel-perfect edits (e.g. Inspect Element alterations).
- **Automated PDF Forensic Dossiers:** Generates downloadable, court-ready PDF reports with region coordinates, metadata, methodology, and legal disclaimers.
- **Cryptographic Model Verification:** Hardened backend verifying SHA256 hashes of all checkpoints on startup to prevent silent model corruption.
- **Real-Time Latency:** End-to-end dual inference in under 55 ms (~19–33 FPS on GPU) with total VRAM usage under 2.2 GB.

---

## Canonical Performance Benchmark

Evaluated across the frozen 122-document production test benchmark and the untouched DocTamper validation split:

### 1. Model A: Physical & Image Manipulation Baseline (122 Test Samples)
| Metric | Macro Dataset Score | Sample-Wise Mean | Status |
|---|---|---|---|
| **Pixel Dice** | **0.6940** | 0.7192 | **0% Regression (PASS)** |
| **Pixel IoU** | **0.5314** | 0.6604 | **0% Regression (PASS)** |
| **Pixel Precision** | **0.6224** | 0.7201 | **0% Regression (PASS)** |
| **Pixel Recall** | **0.7842** | 0.7332 | **0% Regression (PASS)** |
| **Region Recall** | **0.6553** | 0.6553 | **0% Regression (PASS)** |

### 2. Model B: DocTamper Tiny-Text Specialist (Raw vs. Post-Processed)
| Metric | Raw Predictions ($T=0.50$) | Post-Processed Pipeline ($T=0.45$) | Impact |
|---|---|---|---|
| **FP Regions / Document** | **15.16** | **2.21** | **6.85x Reduction** |
| **Overall Region Recall** | 65.70% | **65.52%** | High preservation |
| **Tiny-Region Recall ($<0.5\%$)** | 28.42% | **25.26%** | High micro-text recall |
| **Post-Processing Latency** | 0.0 ms | **5.18 ms** | Real-time |

*See [`reports/final_metric_reconciliation.md`](reports/final_metric_reconciliation.md) for detailed mathematical derivations and evaluation configurations.*

---

## Known System Limitations

1. **Clean Digital Re-Rendering:** Browser 'Inspect Element' edits or vector PDF recreation generate clean rasterization pixels that do not exhibit visual editing noise. In such cases, visual models will correctly report *"No significant visual manipulation evidence detected"*; semantic consistency and reference verification must be used.
2. **Homogeneous Copy-Move:** Splicing text from identical fonts and colors on the same document page remains challenging for visual models without semantic context.
3. **OCR Reliance for Corridor Expansion:** Content consistency and corridor filters depend on OCR character recognition quality on degraded physical scans.
4. **Conservative Legal Disclaimer:** Automated forensic findings indicate visual and mathematical anomalies and should not be interpreted as definitive proof of document authenticity or fraud.

---

## Tech Stack

- **Deep Learning & Forensics:** PyTorch, Torchvision, OpenCV, Spatial Rich Models (SRM), NumPy, Scikit-Image
- **Text & Spatial Extraction:** EasyOCR (CRAFT Detector + CRNN Recognizer)
- **Reporting Engine:** ReportLab, PyMuPDF (Fitz), PIL
- **Backend API:** FastAPI, Uvicorn, Pydantic v2
- **Frontend Workspace:** Next.js 14, React 18, TypeScript, Tailwind CSS, Lucide Icons

---

## Quickstart & Local Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.10, 3.11, 3.12, 3.14)
- Node.js 18+ and npm
- CUDA-compatible GPU recommended (CPU fallback supported automatically)

### 1. Clone the Repository
```bash
git clone https://github.com/bishtprateek270-hue/DocForensics-AI.git
cd DocForensics-AI
```

### 2. Backend Installation & Startup
```bash
# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 3. Frontend Installation & Startup
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Automated Test Suite & Release Verification

Run the complete test suite:
```bash
pytest tests/ -v
```
Run the reproducible release benchmark:
```bash
python -m src.evaluation.run_release_benchmark
```
Build the Next.js production bundle:
```bash
cd frontend && npm run build
```

---

## Project Documentation

- [Release Manifest (v1.0.0)](release_manifest.json)
- [Final Release Benchmark Report](reports/FINAL_RELEASE_REPORT.md)
- [Metric Reconciliation Report](reports/final_metric_reconciliation.md)
- [Resume & Portfolio Summary](docs/resume_project_summary.md)
- [Live Demo Script](docs/demo_script.md)
- [System Architecture](docs/architecture_diagram.md)
- [Release Notes](RELEASE_NOTES_v1.0.0.md)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.  
*Dataset licenses: DocTamper dataset is governed by its respective authors and academic terms.*
