"use client";

import React, { useEffect, useState } from "react";
import { RefreshCw, FileSearch, Layers, CheckCircle2, Type, Eye } from "lucide-react";

interface AnalysisProgressProps {
  previewUrl: string | null;
  filename: string;
}

const pipelineStages = [
  { label: "Preparing document and verifying format...", icon: Layers },
  { label: "Scanning visual patterns and texture consistency...", icon: Eye },
  { label: "Detecting localized manipulation anomalies...", icon: FileSearch },
  { label: "Extracting readable text and numbers from regions...", icon: Type },
  { label: "Compiling forensic analysis report...", icon: CheckCircle2 },
];

export const AnalysisProgress: React.FC<AnalysisProgressProps> = ({
  previewUrl,
  filename,
}) => {
  const [currentStageIdx, setCurrentStageIdx] = useState<number>(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStageIdx((prev) => (prev < pipelineStages.length - 1 ? prev + 1 : prev));
    }, 700);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-8 shadow-md max-w-2xl mx-auto my-8">
      <div className="flex items-center justify-between border-b border-slate-200 pb-4 mb-6">
        <div className="flex items-center gap-3">
          <RefreshCw className="w-5 h-5 text-blue-600 animate-spin" />
          <h3 className="font-bold text-base text-slate-900">
            Analyzing Document...
          </h3>
        </div>
        <span className="font-mono text-xs text-slate-500 truncate max-w-[200px] bg-slate-100 px-2 py-1 rounded">
          {filename}
        </span>
      </div>

      {/* Document Scanner Preview */}
      {previewUrl && (
        <div className="relative aspect-[16/9] max-h-56 bg-slate-100 rounded-xl border border-slate-200 overflow-hidden flex items-center justify-center mb-6 shadow-inner">
          <img
            src={previewUrl}
            alt="Analyzing document"
            className="max-h-full max-w-full object-contain filter grayscale contrast-125 opacity-75"
          />
          {/* Animated Scanning Line */}
          <div className="absolute inset-x-0 h-1.5 bg-gradient-to-r from-transparent via-red-500 to-transparent shadow-[0_0_12px_rgba(239,68,68,0.9)] animate-scan-line pointer-events-none" />
          {/* Grid overlay */}
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#00000008_1px,transparent_1px),linear-gradient(to_bottom,#00000008_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />
        </div>
      )}

      {/* Procedural Pipeline Sequence */}
      <div className="space-y-3">
        {pipelineStages.map((stage, idx) => {
          const isDone = idx < currentStageIdx;
          const isCurrent = idx === currentStageIdx;

          return (
            <div
              key={idx}
              className={`flex items-center gap-3.5 p-3 rounded-xl text-sm transition-all ${
                isCurrent
                  ? "bg-blue-50 border border-blue-200 text-blue-950 font-semibold"
                  : isDone
                  ? "bg-slate-50 text-slate-700 opacity-90"
                  : "text-slate-400 opacity-50"
              }`}
            >
              <div className="shrink-0">
                {isDone ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                ) : isCurrent ? (
                  <RefreshCw className="w-4 h-4 text-blue-600 animate-spin" />
                ) : (
                  <div className="w-4 h-4 rounded-full border border-slate-300" />
                )}
              </div>
              <span className="truncate">{stage.label}</span>
            </div>
          );
        })}
      </div>

      <div className="mt-6 pt-4 border-t border-slate-200 text-center">
        <p className="text-xs text-slate-500">
          This usually takes only 1–3 seconds. Please do not close the window.
        </p>
      </div>
    </div>
  );
};
