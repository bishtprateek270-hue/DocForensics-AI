# DocForensics AI — System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Document Ingestion & Preprocessing"]
        Doc[Uploaded Document<br/>PDF / PNG / JPEG] --> Render[Resolution Normalizer &<br/>PDF PyMuPDF Rasterizer]
        Render --> RGB[Normalized RGB Tensor<br/>512x512]
        Render --> Orig[Original Resolution Array<br/>H0 x W0]
    end

    subgraph Specialists ["2. Dual Neural Specialists"]
        RGB --> ModelA["MODEL A: Physical Forensics<br/>RGB + SRM Dual-Stream<br/>(checkpoints/dual_stream_best.pth)"]
        RGB --> ModelB["MODEL B: Tiny-Text Forensics<br/>Focal Tversky Specialist<br/>(checkpoints/dual_stream_doctamper_tinytext_best.pth)"]
        
        ModelA --> ProbA[Physical Tampering<br/>Probability Map]
        ModelB --> ProbB[Digital Text Tampering<br/>Probability Map]
    end

    subgraph TextPipeline ["3. Spatial OCR & Post-Processing"]
        Orig --> OCR[EasyOCR Engine<br/>CRAFT + CRNN]
        OCR --> Tokens[Recognized Text Lines &<br/>Bounding Corridors]
        
        ProbB --> PP["Tiny-Text Post-Processor<br/>- Min Component Size (>=15px)<br/>- 15% OCR Corridor Constraint<br/>- Adjacent Fragment Merging"]
        Tokens -.-> PP
        
        PP --> TextRegions[Suspicious Text Regions]
        ProbA --> PhysRegions[Physical Splicing Regions]
    end

    subgraph Consistency ["4. Semantic & Arithmetic Consistency"]
        Tokens --> Rules["Content Consistency Engine<br/>- Academic SGPA/Credit Formulas<br/>- Invoice Subtotal/Tax Math<br/>- Format Regularity"]
        Rules --> Inconsistencies[Content Consistency Findings]
        
        Doc -.-> RefAuth["Optional Reference Verification<br/>(Authoritative Database Comparison)"]
        RefAuth -.-> RefFindings[Reference Mismatches]
    end

    subgraph FusionEngine ["5. Evidence Fusion & Reporting"]
        PhysRegions --> Fusion[Evidence Fusion Engine<br/>Spatial IoU Overlap Deduplication]
        TextRegions --> Fusion
        Inconsistencies --> Fusion
        RefFindings --> Fusion
        
        Fusion --> StructReport["Structured Forensic Dossier<br/>- Physical Visual Evidence Channel<br/>- Digital Text Evidence Channel<br/>- Content Consistency Channel<br/>- Reference Verification Channel"]
        
        StructReport --> WebUI[Next.js Interactive Workspace]
        StructReport --> PDF[ReportLab Forensic PDF Dossier]
    end

    style ModelA fill:#1E293B,stroke:#3B82F6,stroke-width:2px,color:#FFFFFF
    style ModelB fill:#1E293B,stroke:#A855F7,stroke-width:2px,color:#FFFFFF
    style Fusion fill:#0F172A,stroke:#10B981,stroke-width:2px,color:#FFFFFF
    style StructReport fill:#0F172A,stroke:#E2E8F0,stroke-width:1.5px,color:#FFFFFF
```

---

## Data Flow Summary

1. **Ingestion:** Supports PNG, JPEG, and multi-page PDF documents. All pages are rasterized and normalized to $512 \times 512$ ImageNet-standard tensors while maintaining full original resolution coordinates.
2. **Dual Inference:** Model A (physical splicing, signature/photo copy-move, stamps) and Model B (micro-text and numerical alterations) execute concurrently in GPU inference mode.
3. **OCR-Aware Suppression:** EasyOCR extracts bounding boxes around text tokens; Model B predictions outside expanded text corridors or below 15 pixels are suppressed, reducing false-positive regions by 6.85x.
4. **Consistency Evaluation:** Deterministic rule engines recalculate arithmetic and semester GPA formulas to catch pixel-perfect Inspect-Element edits.
5. **Channelized Fusion:** Detections are structured into separated channels without synthetic probability averaging, generating both web workspace interactions and authoritative PDF dossiers.
