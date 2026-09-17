# 🛡️ DocForensics AI — Document Tampering Detection, Localization & Content Consistency

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14.2%2B-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**DocForensics AI** is an end-to-end Computer Vision, Deep Learning, and Semantic Verification system designed to detect, localize, and explain digital document manipulation across academic transcripts, certificates, invoices, contracts, and identity documents.

---

## 🏛️ System Architecture

DocForensics AI uses a **Three-Channel Evidence Pipeline**:

1. **Visual Forensic Channel (Spatial & Frequency Stream)**:
   - **Model**: Dual-Stream RGB + SRM ResNet34 with ASPP Decoder (`checkpoints/dual_stream_best.pth`).
   - **Features**: Spatial Rich Model (SRM) high-pass directional filters + Error Level Analysis (ELA) + Block-DCT frequency residuals.
   - **Output**: Pixel-level probability heatmap and bounding box localization.
2. **Content Consistency Channel (Semantic & Arithmetic Rules)**:
   - **OCR**: CRAFT text detection + CRNN text recognition (EasyOCR).
   - **Deterministic Rules**: SGPA/CGPA mathematical boundary checks, component marks summations, date sequence monotonicity, and invoice checksums.
   - **Inspect-Element Defense**: Catches clean browser DOM alterations that leave no image-editing pixel noise.
3. **Reference Verification Channel (Trusted Document Comparison)**:
   - Feature-based homography alignment against trusted originals with structural difference mapping.

---

## 📊 Quantitative Benchmark Results

### 1. Canonical Frozen Test Set (122 Unseen Document Samples)
Evaluated deterministically in FP32 via the Canonical Evaluator at threshold $0.50$:

| Model Architecture | Checkpoint | Test Dice | Test IoU | Precision | Recall | Region Recall | Tampered Det. Rate | Authentic FPR | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net Baseline** | `unet_best.pth` | 0.5245 | 0.4580 | 0.7348 | 0.5943 | 0.5820 | 72.16% | 28.00% | 13.5 ms |
| **DeepLabV3+** | `deeplabv3plus_best.pth` | 0.5305 | 0.4651 | 0.8261 | 0.5351 | 0.5410 | 76.29% | 4.00% | 9.59 ms |
| **SegFormer-B0** | `segformer_best.pth` | 0.2627 | 0.2567 | **0.9892** | 0.2666 | 0.2623 | 27.84% | **0.00%** | 26.54 ms |
| **Phase 7 Dual-Stream (Production)** | `dual_stream_best.pth` | **0.7192** | **0.6604** | 0.8605 | **0.7331** | **0.7772** | **84.54%** | **0.00%** | **10.44 ms** |

### 2. External Real-World Benchmark (Controlled Pairs & Real Conditions)
- **Authentic Hard Negatives (Genuine Stamps/Signatures):** **0.00% False Positive Rate** (zero false alarms on pristine documents).
- **Inspect-Element Re-renders:** Visual model produces no false edge artifacts; **100% caught by Content Consistency Layer**.
- **Heavy JPEG Compression ($Q=50$):** High-frequency SRM residuals attenuated; flagged by secondary ELA and OCR checksums.

---

## 🚀 Quick Start & Reproducibility

### 1. Environment Setup
```bash
pip install -r requirements.txt
python verify_env.py
```

### 2. Run Canonical Evaluation Benchmark
```bash
python src/evaluation/run_phase11.py
```
Outputs manifests and tables to `reports/phase11/final_benchmark_table.json`.

### 3. Run Automated Tests
```bash
pytest tests/ -v
```
All **69 unit and integration tests** pass cleanly.

### 4. Start Full-Stack Application
- **Backend (FastAPI)**:
  ```bash
  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
  ```
- **Frontend (Next.js 14)**:
  ```bash
  cd frontend
  npm run dev
  ```

---

## 🔒 Forensic Disclaimer
> *Results indicate potential visual manipulation or content inconsistencies. Absence of detected evidence does not establish document authenticity. The system should be utilized as an investigative decision-support tool alongside authoritative records.*
