"use client";

import React, { useEffect, useState } from "react";
import { Cpu, RefreshCw, Layers, Crosshair, Type, CheckCircle2 } from "lucide-react";

interface AnalysisProgressProps {
  previewUrl: string | null;
  filename: string;
}

const pipelineStages = [
  { label: "Rasterizing and normalizing document tensor...", icon: Layers },
  { label: "Computing 30 Steganographic Residual (SRM) noise kernels...", icon: Cpu },
  { label: "Executing dual-stream spatial & residual inference...", icon: Crosshair },
  { label: "Extracting 8-connected suspicious regions & bounding boxes...", icon: Crosshair },
  { label: "Running PyTorch CRAFT & CRNN text evidence extraction...", icon: Type },
  { label: "Generating structured forensic audit report...", icon: CheckCircle2 },
];

export const AnalysisProgress: React.FC<AnalysisProgressProps> = ({
  previewUrl,
  filename,
}) => {
  const [currentStageIdx, setCurrentStageIdx] = useState<number>(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStageIdx((prev) => (prev < pipelineStages.length - 1 ? prev + 1 : prev));
    }, 650);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-white rounded-xl border border-surface-200 p-6 shadow-card max-w-2xl mx-auto my-6">
      <div className="flex items-center justify-between border-b border-surface-200 pb-4 mb-5">
        <div className="flex items-center gap-2.5">
          <RefreshCw className="w-4 h-4 text-blue-600 animate-spin" />
          <h3 className="font-semibold text-sm text-surface-900">
            Forensic Analysis in Progress
          </h3>
        </div>
        <span className="font-mono text-xs text-slate-500 truncate max-w-[200px]">
          {filename}
        </span>
      </div>

      {/* Document Scanner Preview */}
      {previewUrl && (
        <div className="relative aspect-[16/9] max-h-52 bg-surface-100 rounded-lg border border-surface-200 overflow-hidden flex items-center justify-center mb-6">
          <img
            src={previewUrl}
            alt="Analyzing document"
            className="max-h-full max-w-full object-contain filter grayscale contrast-125 opacity-70"
          />
          {/* Animated Scanning Line */}
          <div className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-red-500 to-transparent shadow-[0_0_12px_rgba(239,68,68,0.9)] animate-scan-line pointer-events-none" />
          {/* Grid overlay */}
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#00000008_1px,transparent_1px),linear-gradient(to_bottom,#00000008_1px,transparent_1px)] bg-[size:16px_16px] pointer-events-none" />
        </div>
      )}

      {/* Procedural Pipeline Sequence */}
      <div className="space-y-2.5">
        {pipelineStages.map((stage, idx) => {
          const Icon = stage.icon;
          const isDone = idx < currentStageIdx;
          const isCurrent = idx === currentStageIdx;
          const isPending = idx > currentStageIdx;

          return (
            <div
              key={idx}
              className={`flex items-center gap-3 p-2.5 rounded-lg text-xs transition-all ${
                isCurrent
                  ? "bg-blue-50/70 border border-blue-200 text-blue-900 font-medium"
                  : isDone
                  ? "bg-surface-50 text-slate-700 opacity-90"
                  : "text-slate-400 opacity-50"
              }`}
            >
              <div className="shrink-0">
                {isDone ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                ) : isCurrent ? (
                  <RefreshCw className="w-3.5 h-3.5 text-blue-600 animate-spin" />
                ) : (
                  <div className="w-3.5 h-3.5 rounded-full border border-slate-300" />
                )}
              </div>
              <span className="truncate">{stage.label}</span>
            </div>
          );
        })}
      </div>

      <div className="mt-4 pt-3 border-t border-surface-200 text-center">
        <p className="text-[11px] text-slate-400">
          Running on CUDA-accelerated dual-stream neural engine. Please do not refresh.
        </p>
      </div>
    </div>
  );
};
