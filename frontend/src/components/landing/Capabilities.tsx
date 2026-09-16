import React from "react";
import { Hash, Calendar, PenTool, Image as ImageIcon, Stamp, Eraser, AlertCircle } from "lucide-react";

const capabilities = [
  {
    icon: Hash,
    title: "Financial & Number Edits",
    description: "Detects modified invoice amounts, changed bank account numbers, altered prices, and replaced tax figures.",
  },
  {
    icon: Calendar,
    title: "Date & Period Alterations",
    description: "Identifies replaced issue dates, modified expiry dates, and altered timeline stamps across official forms.",
  },
  {
    icon: PenTool,
    title: "Signature Splicing",
    description: "Pinpoints signatures that have been digitally cut, copied, or superimposed onto contracts and agreements.",
  },
  {
    icon: ImageIcon,
    title: "Photo & ID Substitution",
    description: "Localizes replaced photos on identity cards, badges, certificates, passports, and credential documents.",
  },
  {
    icon: Stamp,
    title: "Stamp & Seal Modifications",
    description: "Exposes digitally altered or transplanted institutional seals, approval stamps, and notary marks.",
  },
  {
    icon: Eraser,
    title: "Content Erasure & Inpainting",
    description: "Highlights deleted text clauses, erased disclaimers, or painted-over document sections.",
  },
];

export const Capabilities: React.FC = () => {
  return (
    <section id="capabilities" className="py-16 md:py-24 border-b border-slate-200 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl mb-14">
          <div className="text-xs font-bold uppercase tracking-wider text-blue-600 mb-2">
            Inspection Capabilities
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
            Types of Document Manipulation We Inspect
          </h2>
          <p className="text-base sm:text-lg text-slate-600 mt-3">
            DocForensics AI is calibrated to detect common forms of visual and textual forgery in digital paperwork.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {capabilities.map((c, i) => {
            const Icon = c.icon;
            return (
              <div
                key={i}
                className="bg-slate-50/70 rounded-xl border border-slate-200 p-6 shadow-sm hover:shadow-md hover:bg-white transition-all"
              >
                <div className="w-11 h-11 rounded-lg bg-white border border-slate-200 flex items-center justify-center text-slate-900 mb-4 shadow-sm">
                  <Icon className="w-5 h-5 text-blue-600" />
                </div>
                <h3 className="font-bold text-base text-slate-900 mb-2">
                  {c.title}
                </h3>
                <p className="text-sm text-slate-600 leading-relaxed">
                  {c.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* Short Unobtrusive Disclaimer */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 flex items-start gap-3.5 text-xs text-slate-600 max-w-4xl">
          <AlertCircle className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <strong className="text-slate-800">Notice:</strong> Results indicate potential visual manipulation and should not be treated as proof of document authenticity. A qualified forensic document examiner should review any flagged items for critical decisions.
          </p>
        </div>
      </div>
    </section>
  );
};
