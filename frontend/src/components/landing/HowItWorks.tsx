import React from "react";
import { UploadCloud, Cpu, Crosshair, Type, FileSpreadsheet } from "lucide-react";

const steps = [
  {
    step: "01",
    icon: UploadCloud,
    title: "Document Ingestion",
    description:
      "Upload image formats (JPG, PNG) or multi-page PDFs. Vector PDF documents are rasterized at high DPI preserving pixel alignment and coordinate mapping.",
  },
  {
    step: "02",
    icon: Cpu,
    title: "Dual-Stream Extraction",
    description:
      "Simultaneously extracts RGB spatial semantics via ResNet50 and noise residual inconsistencies via 30 Steganographic Residual Analysis (SRM) high-pass kernels.",
  },
  {
    step: "03",
    icon: Crosshair,
    title: "Tampering Localization",
    description:
      "An adaptive convolutional gating module fuses spatial and noise features to generate a high-resolution pixel-level probability map (0.0 – 1.0).",
  },
  {
    step: "04",
    icon: Type,
    title: "OCR Evidence Extraction",
    description:
      "PyTorch-based CRAFT text detection and CRNN recognition map localized tampering bounding boxes to actual document text, amounts, and dates.",
  },
  {
    step: "05",
    icon: FileSpreadsheet,
    title: "Structured Forensic Report",
    description:
      "Generates an interactive verification workspace with heatmap cross-fading, per-region statistics, OCR evidence tables, and exportable JSON artifacts.",
  },
];

export const HowItWorks: React.FC = () => {
  return (
    <section id="how-it-works" className="py-16 md:py-20 border-b border-surface-200 bg-surface-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-2xl mb-12">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
            Verification Pipeline
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold text-surface-900 tracking-tight">
            How DocForensics AI Analyzes Documents
          </h2>
          <p className="text-sm sm:text-base text-slate-600 mt-2">
            A deterministic, multi-modal forensic inspection process designed to expose subtle digital alterations without relying on black-box claims.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6">
          {steps.map((s) => {
            const Icon = s.icon;
            return (
              <div
                key={s.step}
                className="bg-white rounded-lg border border-surface-200 p-5 shadow-subtle flex flex-col justify-between relative hover:border-surface-300 transition-colors"
              >
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <span className="font-mono text-xs font-bold text-slate-400">
                      {s.step}
                    </span>
                    <div className="w-8 h-8 rounded bg-surface-100 flex items-center justify-center text-surface-900">
                      <Icon className="w-4 h-4 text-slate-700" />
                    </div>
                  </div>
                  <h3 className="font-semibold text-sm text-surface-900 mb-2">
                    {s.title}
                  </h3>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    {s.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
