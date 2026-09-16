import React from "react";
import { ShieldAlert, Cpu, FileCheck2, Terminal } from "lucide-react";

export const Footer: React.FC = () => {
  return (
    <footer className="w-full border-t border-surface-200 bg-surface-50 text-slate-600 text-xs py-10 mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded bg-surface-900 text-white flex items-center justify-center font-bold text-xs">
                DF
              </div>
              <span className="font-semibold text-sm text-surface-900">
                DocForensics AI
              </span>
            </div>
            <p className="text-slate-500 leading-relaxed">
              Research & production document tampering localization suite combining RGB Spatial Analysis and SRM Noise Residual Fusion.
            </p>
          </div>

          <div>
            <h4 className="font-semibold text-surface-900 mb-2">Model Specifications</h4>
            <ul className="space-y-1.5 text-slate-500 font-mono text-[11px]">
              <li>Dual-Stream DeepLabV3+ (ResNet50)</li>
              <li>Spatial: 3-ch RGB Image</li>
              <li>Forensic: 30-filter SRM Kernels</li>
              <li>Fusion: Adaptive Conv Gate + Decoder</li>
              <li>Checkpoint: Phase 7 (Val Dice 0.719)</li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-surface-900 mb-2">Inference & OCR</h4>
            <ul className="space-y-1.5 text-slate-500 text-[11px]">
              <li>PyTorch CUDA 12.8 / RTX 5050 8GB</li>
              <li>EasyOCR PyTorch (CRAFT + CRNN)</li>
              <li>Spatial IoU Region-Text Association</li>
              <li>PyMuPDF High-DPI Page Rendering</li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-surface-900 mb-2">Responsible Usage</h4>
            <p className="text-slate-500 leading-relaxed text-[11px]">
              DocForensics AI identifies pixel-level manipulation artifacts. It does not certify legal document authenticity or prove legal intent. All suspicious findings should undergo human forensic verification.
            </p>
          </div>
        </div>

        <div className="pt-6 border-t border-surface-200 flex flex-col sm:flex-row items-center justify-between text-slate-400 text-[11px] gap-2">
          <span>&copy; {new Date().getFullYear()} DocForensics AI. Production Forensic Pipeline (Phase 9).</span>
          <div className="flex items-center gap-4">
            <span>FastAPI 0.115</span>
            <span>&bull;</span>
            <span>Next.js 14</span>
            <span>&bull;</span>
            <span>PyTorch 2.11+cu128</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
