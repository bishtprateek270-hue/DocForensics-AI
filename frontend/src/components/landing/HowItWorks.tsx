import React from "react";
import { UploadCloud, Search, CheckCircle2, FileText } from "lucide-react";

const steps = [
  {
    step: "01",
    icon: UploadCloud,
    title: "Upload Document",
    description:
      "Upload your document in PNG, JPG, or PDF format. Vector PDFs are automatically converted at high resolution for precise inspection.",
  },
  {
    step: "02",
    icon: Search,
    title: "AI Forensic Analysis",
    description:
      "The system inspects sub-pixel visual patterns and noise consistency to spot signs of splicing, inpainting, or digital tampering.",
  },
  {
    step: "03",
    icon: CheckCircle2,
    title: "Inspect Suspicious Regions",
    description:
      "View exact highlighted bounding boxes, interactive heatmaps with opacity controls, and localized tampering scores.",
  },
  {
    step: "04",
    icon: FileText,
    title: "View Forensic Report",
    description:
      "Review extracted OCR text evidence, inspect affected document areas, and export a complete forensic audit report.",
  },
];

export const HowItWorks: React.FC = () => {
  return (
    <section id="how-it-works" className="py-16 md:py-24 border-b border-slate-200 bg-slate-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl mb-14">
          <div className="text-xs font-bold uppercase tracking-wider text-blue-600 mb-2">
            Simple 4-Step Process
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
            How DocForensics AI Works
          </h2>
          <p className="text-base sm:text-lg text-slate-600 mt-3">
            An automated forensic verification pipeline designed to spot subtle document alterations quickly and accurately.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {steps.map((s) => {
            const Icon = s.icon;
            return (
              <div
                key={s.step}
                className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm flex flex-col justify-between hover:shadow-md hover:border-slate-300 transition-all"
              >
                <div>
                  <div className="flex items-center justify-between mb-5">
                    <span className="font-mono text-sm font-extrabold text-slate-400 bg-slate-100 px-2.5 py-1 rounded">
                      {s.step}
                    </span>
                    <div className="w-10 h-10 rounded-lg bg-blue-50 flex items-center justify-center text-blue-600">
                      <Icon className="w-5 h-5" />
                    </div>
                  </div>
                  <h3 className="font-bold text-base text-slate-900 mb-2">
                    {s.title}
                  </h3>
                  <p className="text-sm text-slate-600 leading-relaxed">
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
