# DocForensics AI — 2 to 3 Minute Live Demo Script

**Target Audience:** Recruiters, Technical Interviewers, Hackathon Judges, Document Security Engineers  
**Duration:** 2 Minutes 30 Seconds  

---

## 1. Problem Introduction (30 Seconds)
- **Say:** *"In banking, legal, and academic verification, document fraud is shifting rapidly from crude cut-and-paste jobs to micro-text digital replacements—like modifying a single number on an invoice or changing a grade on a transcript. Traditional verification relies either on naive visual models that drown in false alarms, or simple rule engines that miss physical photo and signature swaps. DocForensics AI solves this with a multi-evidence dual-specialist pipeline."*
- **Action:** Open application landing page (`http://localhost:3000`).

---

## 2. Document Upload & Ingestion (20 Seconds)
- **Say:** *"The system accepts standard PDF, PNG, and JPEG documents. Let's upload a tampered marksheet where a student's SGPA was digitally altered from 8.21 to 9.85 alongside a physical stamp modification."*
- **Action:** Select or drag-and-drop the sample marksheet into the Upload Zone.
- **Show:** The progressive analysis state machine activating:
  1. *Preparing Document*
  2. *Analyzing Physical & Sensor Traces (Model A)*
  3. *Analyzing Tiny-Text Modifications (Model B)*
  4. *Extracting Text & Spatial OCR (EasyOCR)*
  5. *Verifying Semantic & Arithmetic Consistency*
  6. *Synthesizing Multi-Evidence Findings*

---

## 3. Results Workspace & Localization Overlay (40 Seconds)
- **Say:** *"In under 55 milliseconds, the system produces a unified evidence workspace. Rather than displaying a fake '98% Authentic' score, the interface breaks findings down across independent, auditable evidence channels."*
- **Highlight on Screen:**
  - **Visual Localization Overlay:** Point out the dual-colored bounding boxes:
    - *Blue box* on the physical stamp boundary (flagged by Model A - Physical Specialist).
    - *Purple box* on the SGPA amount (flagged by Model B - Tiny-Text Specialist).
  - **Heatmap Toggle:** Switch to the Jet probability heatmap view to show localized high-confidence activation.

---

## 4. Region Inspector & OCR Association (30 Seconds)
- **Say:** *"Clicking on Region #1 reveals the component breakdown. The post-processor expanded OCR text lines to confirm that this anomaly is attached to the string 'SGPA: 9.85', filtering out adjacent character noise."*
- **Action:** Click Region #1 in the Region Inspector to zoom in and display:
  - Contributing Specialist Model tag
  - Associated OCR Text (`"SGPA: 9.85"`)
  - Localization Confidence Score (`0.88`)
  - Document Area Coverage (`0.32%`)

---

## 5. Content Consistency & Inspect Element Handling (20 Seconds)
- **Say:** *"Next, look at the Content Consistency tab. Even if a bad actor cleanly re-renders text using browser 'Inspect Element'—which produces clean raster pixels without visual editing noise—our semantic engine recalculates the weighted semester GPA from extracted course grades and flags the mathematical mismatch."*
- **Action:** Switch to Content Consistency tab to show the failed GPA formula check.

---

## 6. PDF Forensic Report Download & Conclusion (10 Seconds)
- **Say:** *"Finally, examiners can export a complete, court-ready Forensic PDF Dossier with a single click, complete with region coordinates, metadata, methodology notes, and legal disclaimers."*
- **Action:** Click **"Download PDF Report"**; show the generated ReportLab dossier.
- **Closing Statement:** *"DocForensics AI provides verifiable, conservative, real-time document forensic intelligence ready for enterprise deployment."*
