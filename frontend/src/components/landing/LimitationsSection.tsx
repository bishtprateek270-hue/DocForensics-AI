import React from "react";
import { AlertCircle, Scale, Eye, FileWarning, Cpu } from "lucide-react";

export const LimitationsSection: React.FC = () => {
  return (
    <section id="limitations" className="py-16 md:py-20 border-b border-surface-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-2xl mb-12">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
            Forensic Integrity &amp; Transparency
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold text-surface-900 tracking-tight">
            Known Boundaries &amp; Operating Constraints
          </h2>
          <p className="text-sm sm:text-base text-slate-600 mt-2">
            Responsible digital forensics requires explicit disclosure of assumptions and limitations. DocForensics AI provides statistical evidence, not judicial proof.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg border border-surface-200 p-6 shadow-subtle flex items-start gap-4">
            <div className="w-9 h-9 rounded-md bg-surface-100 flex items-center justify-center text-slate-700 shrink-0">
              <Scale className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-semibold text-sm text-surface-900 mb-1">
                Visual Evidence &ne; Legal Authenticity
              </h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                Localization masks indicate high probability of inconsistent pixel distributions. The model does not verify document validity, legal signing authority, or intent.
              </p>
            </div>
          </div>

          <div className="bg-white rounded-lg border border-surface-200 p-6 shadow-subtle flex items-start gap-4">
            <div className="w-9 h-9 rounded-md bg-surface-100 flex items-center justify-center text-slate-700 shrink-0">
              <FileWarning className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-semibold text-sm text-surface-900 mb-1">
                Heavy JPEG &amp; Rescan Degradation
              </h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                Repeated lossy re-compression or low-resolution physical photo scans can blur high-pass noise residuals, potentially raising false negatives on faint alterations.
              </p>
            </div>
          </div>

          <div className="bg-white rounded-lg border border-surface-200 p-6 shadow-subtle flex items-start gap-4">
            <div className="w-9 h-9 rounded-md bg-surface-100 flex items-center justify-center text-slate-700 shrink-0">
              <Eye className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-semibold text-sm text-surface-900 mb-1">
                OCR Character Recognition Limits
              </h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                OCR extraction operates on CRAFT/CRNN bounding boxes. Cursive signatures, distorted text, or non-Latin scripts may exhibit lower text confidence or empty readings.
              </p>
            </div>
          </div>

          <div className="bg-white rounded-lg border border-surface-200 p-6 shadow-subtle flex items-start gap-4">
            <div className="w-9 h-9 rounded-md bg-surface-100 flex items-center justify-center text-slate-700 shrink-0">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-semibold text-sm text-surface-900 mb-1">
                Human-in-the-Loop Protocol
              </h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                All suspicious regions flagged with moderate scores (0.35 &le; score &le; 0.65) should be routed to a qualified forensic document examiner for corroborating review.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
