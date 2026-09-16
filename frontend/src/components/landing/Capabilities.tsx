import React from "react";
import { CheckCircle2, AlertTriangle, Hash, Calendar, PenTool, Image as ImageIcon, Stamp, Eraser } from "lucide-react";

const capabilities = [
  {
    icon: Hash,
    title: "Financial & Number Replacement",
    description: "Detects modified invoice amounts, bank balances, tax totals, and salary figures spliced into existing layout backgrounds.",
    supported: true,
  },
  {
    icon: Calendar,
    title: "Date & Period Alteration",
    description: "Identifies replaced issue dates, expiration periods, and timeline stamps exhibiting localized noise frequency mismatches.",
    supported: true,
  },
  {
    icon: PenTool,
    title: "Signature Splicing & Forgery",
    description: "Localizes pasted, erased, or re-positioned signatures and endorsement stamps on legal contracts and affidavits.",
    supported: true,
  },
  {
    icon: ImageIcon,
    title: "Photo & Portrait Substitution",
    description: "Pinpoints spliced portrait photographs on identification documents, badges, driver licenses, and credentials.",
    supported: true,
  },
  {
    icon: Stamp,
    title: "Official Stamp / Seal Modification",
    description: "Exposes altered, digitally superimposed, or transplanted government and institutional rubber stamps.",
    supported: true,
  },
  {
    icon: Eraser,
    title: "Content Inpainting & Erasure",
    description: "Detects algorithmic or manual brush-based deletion of text clauses, disclaimers, or watermarks.",
    supported: true,
  },
];

export const Capabilities: React.FC = () => {
  return (
    <section id="capabilities" className="py-16 md:py-20 border-b border-surface-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 gap-4">
          <div className="max-w-2xl">
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
              Scope of Analysis
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold text-surface-900 tracking-tight">
              Tampering Modalities & Forensic Scope
            </h2>
            <p className="text-sm sm:text-base text-slate-600 mt-2">
              The dual-stream neural engine is calibrated for forensic indicators commonly found in forged digital records.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {capabilities.map((c, i) => {
            const Icon = c.icon;
            return (
              <div
                key={i}
                className="bg-white rounded-lg border border-surface-200 p-5 shadow-subtle hover:shadow-card transition-shadow"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="w-9 h-9 rounded-md bg-surface-100 flex items-center justify-center text-slate-800">
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                    <CheckCircle2 className="w-3 h-3" />
                    Supported
                  </span>
                </div>
                <h3 className="font-semibold text-sm text-surface-900 mb-1.5">
                  {c.title}
                </h3>
                <p className="text-xs text-slate-500 leading-relaxed">
                  {c.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* Boundary Notice */}
        <div className="bg-amber-50/70 border border-amber-200/80 rounded-lg p-4 flex items-start gap-3 text-xs text-amber-900">
          <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <div className="leading-relaxed">
            <span className="font-semibold">Known Boundary Condition:</span> While the system excels at detecting splicing, text replacement, and inpainting, pristine intra-document copy-move duplicates with matched noise characteristics remain challenging for single-frame residual models. Dedicated keypoint-matching modules should be referenced for purely identical self-cloned regions.
          </div>
        </div>
      </div>
    </section>
  );
};
