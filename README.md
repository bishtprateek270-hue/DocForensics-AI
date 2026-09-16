# 🛡️ DocForensics AI — Document Tampering Detection & Localization

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**DocForensics AI** is an end-to-end Deep Learning & Computer Vision framework designed to detect, classify, and localize digital document tampering (copy-move, splicing, text alteration, stamp forgery, and erasure/inpainting) across identity documents, financial receipts, contracts, and certificates.

---

## 📁 Project Directory Structure

```text
document-tampering/
├── data/
│   ├── raw/                 # Clean source documents & public benchmark datasets
│   ├── processed/           # Preprocessed tensors & paired dataset masks
│   └── synthetic/           # Synthetically generated tampered documents & ground-truth masks
├── src/
│   ├── __init__.py
│   ├── preprocessing/       # ELA, SRM noise extractors, DCT filters & Albumentations
│   ├── models/              # Neural network architectures (UNet, Two-Stream, TruFor)
│   ├── training/            # PyTorch training loops, loss functions & learning rate schedulers
│   ├── evaluation/          # Pixel-level IoU, F1-score, AUC-ROC & benchmark evaluations
│   ├── inference/           # Single/Batch inference, PDF parser & heatmap overlays
│   └── ocr/                 # OCR-based layout & font inconsistency analysis
├── notebooks/               # Jupyter exploration and visual prototyping
├── tests/                   # Unit & integration test suites
├── backend/                 # FastAPI REST service & PDF processing endpoints
├── frontend/                # Interactive web dashboard & forensic audit visualizer
├── config.py                # Centralized project configuration & hyperparameters
├── requirements.txt         # Core dependencies
├── verify_env.py            # Environment & GPU acceleration verification script
├── .gitignore               # Version control ignore rules
└── README.md                # Project documentation
```

---

## ⚡ Quick Start

### 1. Environment Setup

Ensure you have Python 3.9+ installed. Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Verify Hardware & Dependencies

Run the diagnostic script to verify PyTorch, CUDA/GPU availability, and directory setup:

```bash
python verify_env.py
```

---

## ⚙️ Configuration (`config.py`)

All global paths, image dimensions, preprocessing settings, model choices, and hyperparameters are managed in [config.py](file:///config.py):

- **Image Resolution**: `512x512` default input size.
- **Device**: Automatic CUDA / MPS / CPU fallback.
- **Forensic Preprocessing**: Configurable Error Level Analysis (ELA) quality factors and SRM filter counts.
- **Model Parameters**: Support for Single-Stream (RGB) and Dual-Stream (RGB + Noise/Frequency) architectures.

---

## 🗺️ Project Milestones

- [x] **Phase 1: Project Scaffold & Environment Setup** *(Completed)*
- [ ] **Phase 2: Preprocessing & Synthetic Tampering Generator**
- [ ] **Phase 3: Model Architecture & Training Pipeline**
- [ ] **Phase 4: Evaluation Benchmark & Forensic Heatmap Visualizer**
- [ ] **Phase 5: Full-Stack Web Application (FastAPI + Modern UI)**
