# DocForensics AI — Resume & Portfolio Summary

**Project:** DocForensics AI — Multi-Evidence Document Forensic Analysis & Tampering Localization System  
**Release Version:** v1.0.0 (Production Release)  

---

## 1. One-Line Project Summary
> An end-to-end multi-evidence document forensics system combining dual deep learning specialists, OCR spatial constraints, and deterministic consistency engines to localize pixel-level tampering and digital text edits without synthetic confidence claims.

---

## 2. 2-Bullet Resume Version (Engineering & Measurable Impact)

- **Architected Dual-Specialist ML Pipeline:** Designed and deployed a multi-evidence document forensics architecture in PyTorch combining an RGB+SRM Dual-Stream baseline for physical splicing (0.6940 test Dice) with a DocTamper Tiny-Text specialist (65.5% region recall) under a unified FastAPI & Next.js service.
- **Engineered 6.85x False-Positive Reduction:** Formulated OCR-constrained spatial corridor filtering, morphological component thresholding ($\ge 15$ px), and adjacent token merging to reduce false-positive regions per document from 15.16 down to 2.21 while maintaining real-time ~33 FPS inference.

---

## 3. 3-Bullet Technical Version (ML & Applied AI Resumes)

- **Asymmetric Loss & Spatial Patch Sampling:** Formulated Focal Tversky loss ($\alpha=0.3, \beta=0.7, \gamma=1.33$) with positive-aware patch sampling on the DocTamper dataset, elevating micro-text tampering recall from 7.18% to 65.52% on sub-0.5% area alterations.
- **Catastrophic Forgetting Mitigation via Specialized Fusion:** Overcame severe domain degradation (test Dice collapse from 0.6940 to 0.0950 during monolithic fine-tuning) by locking production weights and deploying a spatial IoU evidence fusion engine with separated visual, semantic, and reference verification channels.
- **Production-Hardened System & PDF Dossier Engine:** Built a production-grade inference engine with cryptographic SHA256 checkpoint verification, fail-safe startup, EasyOCR text-line extraction, and ReportLab PDF forensic dossier generation serving <55 ms end-to-end latency.

---

## 4. Interview Explanations

### 30-Second Elevator Pitch
> *"DocForensics AI is an AI-powered document verification system designed to detect both physical document tampering (like forged signatures, photos, and stamps) and digital micro-text edits (like changed bank amounts or grades). Because single neural networks suffer catastrophic forgetting when trying to solve both physical splicing and digital font replacement simultaneously, I engineered a dual-specialist architecture. I combined this with OCR text-line geometric constraints to slash false alarms by 6.85x and generate court-ready forensic PDF reports in under 55 milliseconds."*

### 2-Minute Deep Dive
> *"Document tampering detection in industry faces two massive hurdles: domain divergence and alarm fatigue. 
> 
> First, physical manipulation—like spliced stamps, pasted photos, and signature cut-and-paste—leaves sensor noise and edge discontinuity traces best detected by dual-stream RGB and Spatial Rich Model (SRM) filters. In contrast, digital text edits (such as altering a grade from 8.21 to 9.85) use clean vector fonts where tampered pixels represent less than 0.5% of the page. When we fine-tuned our physical model directly on digital text datasets like DocTamper, test Dice collapsed from 0.6940 down to 0.0950 due to catastrophic forgetting. 
> 
> To solve this without compromising either capability, I locked the physical model and developed a dedicated Tiny-Text specialist trained with an asymmetric Focal Tversky loss ($\beta=0.7$) that penalizes false negatives 1.78x more heavily than false positives. 
> 
> While this boosted region recall to over 76%, raw predictions produced 15 false alarms per document due to regular font boundaries. I resolved this by building an OCR-aware post-processing pipeline: we construct 15% expanded geometric corridors around recognized text lines, filter sub-15px noise, and merge adjacent character fragments. This slashed false-positive regions by 6.85x down to 2.21 per document while preserving 65.5% recall.
> 
> Finally, rather than outputting a deceptive '99% authentic' score, the system routes evidence into four independent channels—Physical Forensics, Digital Text, Content Consistency (e.g. arithmetic checks), and Reference DB verification—delivering conservative, auditable forensic PDF reports in real time."*

---

## 5. Key Technical Questions & Answers

### Q1: Why were dual specialists required instead of a single larger model?
> **Answer:** *Physical splicing and digital text replacement are fundamentally different physical phenomena. Physical tampering disrupts sensor noise patterns, color channels, and background paper texture. Digital text modification involves clean vector rendering where background texture is undisturbed. Monolithic fine-tuning caused catastrophic interference where the network discarded paper noise features to fit font edges. Dual specialists allow each network to maintain peak sensitivity in its respective domain without interference.*

### Q2: How did you identify catastrophic forgetting?
> **Answer:** *We ran continuous regression benchmarks on a frozen 122-document physical tampering test set. As soon as the model was fine-tuned on DocTamper text edits, its test Dice on physical splicing dropped from 0.6940 to 0.0950, despite achieving high precision on text edits. This quantified evidence proved that text fine-tuning overwrote physical forensic filters, mandating frozen dual routing.*

### Q3: How did you reduce false positives without destroying recall?
> **Answer:** *Through component-level profiling of 100+ false alarms, we observed that 80%+ of false positives came from isolated sub-15px edge noise and standard printed text boundaries. We introduced three deterministic filters: (1) min component thresholding ($\ge 15$ px), (2) OCR-guided text corridor constraints (suppressing isolated non-text background noise by 3.5x), and (3) spatial merging of adjacent character components on the same text line. This reduced FP regions from 15.16 to 2.21 while retaining 65.5% recall.*

### Q4: What is the primary limitation of pixel-level document forensics?
> **Answer:** *Clean digital re-rendering (such as browser 'Inspect Element' edits re-rendered natively to PDF) leaves no pixel-level manipulation artifacts because the rendered pixels are generated directly by the browser's layout engine. In such scenarios, visual models correctly report 'No significant visual manipulation evidence detected.' This is why our system integrates a second layer: Content Consistency & Semantic Arithmetic (detecting impossible GPA calculations or subtotal mismatches) and an Authoritative Reference Verification interface.*
