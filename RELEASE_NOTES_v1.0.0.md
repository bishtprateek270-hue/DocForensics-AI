# DocForensics AI — Release Notes (v1.0.0)

**Release Version:** `v1.0.0`  
**Release Tag:** `v1.0.0-production-release`  
**Release Date:** September 18, 2026  
**Commit:** Frozen Production Release  

---

## What's New in v1.0.0

### 1. Dual-Specialist Forensic Engine
- **Model A (Physical / Image Forensics):** Locked Phase 7 baseline combining RGB dual-stream with Spatial Rich Model (SRM) filter extractions to detect physical splicing, signature manipulation, photo replacement, stamp manipulation, and sensor noise patterns.
- **Model B (Digital Tiny-Text Forensics):** Dedicated DocTamper specialist fine-tuned with Focal Tversky loss ($\beta=0.7$) to detect font-matched numerical, date, and text modifications spanning $<0.5\%$ of document area.
- **Independent Routing:** Runs both models concurrently without destructive hard-routing switches.

### 2. OCR-Aware False-Positive Reduction
- **6.85x Reduction in False Alarms:** Reduced false-positive regions per document from **15.16 down to 2.21** on the DocTamper validation benchmark.
- **Preserved Recall:** Maintained **65.52% overall region recall** and **25.26% micro-region recall**.
- **Deterministic Post-Processing:** Morphological component thresholding ($\ge 15$ px), OCR 15% text-line corridor expansion, and spatial character fragment merging.

### 3. Multi-Evidence Channel Separation
- Visual forensics, semantic content consistency, and authoritative reference verification are output in strictly separated channels.
- Completely removed synthetic "98% fake" or "authenticity percentage" metrics in favor of auditable localization confidence and conservative forensic findings.

### 4. Production Hardening & PDF Dossier Generation
- **Startup Integrity:** Automatic cryptographic SHA256 verification of both model checkpoints upon server boot.
- **Court-Ready PDF Reports:** ReportLab-powered dossier generator producing structured forensic reports containing region coordinates, OCR context, methodology notes, and legal disclaimers.
- **Real-Time Performance:** Dual-model inference latency of ~30 ms (~33 FPS) with total memory usage under 2.2 GB VRAM.

---

## Canonical Performance Benchmark

| Benchmark Split | Target Domain | Pixel Dice | Region Recall | FP Regions / Doc | Latency (GPU) |
|---|---|---|---|---|---|
| **Frozen 122-Test Baseline** | Physical Splicing & Sensor Artifacts | **0.6940** *(0.7192 sample-wise)* | **65.53%** | 0.00 | 12.73 ms |
| **DocTamper Validation Split** | Digital Font & Micro-Text Editing | **14.02%** | **65.52%** | **2.21** | 17.29 ms |
| **End-to-End Dual Pipeline** | Unified Multi-Specialist Analysis | — | **65.5%** | **2.21** | **30.03 ms** |

---

## Known System Limitations

1. **Clean Digital Re-Rendering:** Inspect Element or vector PDF re-creation produces native raster pixels devoid of editing noise. In such cases, visual models will correctly report *'No significant visual manipulation evidence detected'*; semantic consistency and reference verification must be used.
2. **Copy-Move on Homogeneous Textures:** Duplicate text pasted from identical fonts on the same page remains challenging for pixel-level models without semantic context.
3. **OCR Dependency for Corridors:** Content consistency accuracy depends on OCR character recognition quality on degraded scans.
