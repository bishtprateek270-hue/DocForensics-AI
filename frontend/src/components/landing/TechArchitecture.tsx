import React from "react";
import { Layers, Cpu, GitMerge, FileCode, CheckCircle2 } from "lucide-react";

export const TechArchitecture: React.FC = () => {
  return (
    <section id="architecture" className="py-16 md:py-20 border-b border-surface-200 bg-surface-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-2xl mb-12">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
            Model Architecture
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold text-surface-900 tracking-tight">
            Dual-Stream Feature Fusion Network
          </h2>
          <p className="text-sm sm:text-base text-slate-600 mt-2">
            Why single-stream CNNs miss subtle document forgeries: normal RGB channels focus on visual semantics, while the SRM stream exposes high-frequency boundary traces.
          </p>
        </div>

        {/* Architecture Grid / Diagram */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          {/* Stream 1 */}
          <div className="lg:col-span-4 bg-white rounded-lg border border-surface-200 p-6 shadow-subtle flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-mono font-bold text-blue-600 uppercase bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                  Stream 1 &bull; Spatial
                </span>
                <span className="text-xs font-mono text-slate-400">3 Channels</span>
              </div>
              <h3 className="font-semibold text-base text-surface-900 mb-2">
                RGB Spatial Semantics
              </h3>
              <p className="text-xs text-slate-500 leading-relaxed mb-4">
                Processes full-color document imagery through a deep ResNet-50 feature extractor to capture font geometry, paragraph layout, table structures, and visual alignment.
              </p>
            </div>
            <div className="bg-surface-50 p-3 rounded border border-surface-200 font-mono text-[11px] text-slate-600 space-y-1">
              <div>Backbone: ResNet-50</div>
              <div>Input: 512 &times; 512 &times; 3</div>
              <div>Output: 2048-dim Feature Maps</div>
            </div>
          </div>

          {/* Stream 2 */}
          <div className="lg:col-span-4 bg-white rounded-lg border border-surface-200 p-6 shadow-subtle flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-mono font-bold text-teal-700 uppercase bg-teal-50 px-2 py-0.5 rounded border border-teal-200">
                  Stream 2 &bull; Forensic
                </span>
                <span className="text-xs font-mono text-slate-400">30 Filters</span>
              </div>
              <h3 className="font-semibold text-base text-surface-900 mb-2">
                SRM Noise Residuals
              </h3>
              <p className="text-xs text-slate-500 leading-relaxed mb-4">
                Applies 30 Steganographic Residual Analysis (SRM) high-pass convolution kernels. Suppresses image content to reveal sub-pixel interpolation anomalies, resampling artifacts, and splicing boundaries.
              </p>
            </div>
            <div className="bg-surface-50 p-3 rounded border border-surface-200 font-mono text-[11px] text-slate-600 space-y-1">
              <div>Filters: 1st, 2nd, 3rd Order SRM</div>
              <div>Weights: Fixed Steganographic</div>
              <div>Output: Noise Residual Tensor</div>
            </div>
          </div>

          {/* Stream Fusion & Head */}
          <div className="lg:col-span-4 bg-white rounded-lg border border-surface-200 p-6 shadow-subtle flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-mono font-bold text-indigo-700 uppercase bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200">
                  Fusion &amp; Decoder
                </span>
                <span className="text-xs font-mono text-slate-400">DeepLabV3+</span>
              </div>
              <h3 className="font-semibold text-base text-surface-900 mb-2">
                Gated Fusion &amp; Localization
              </h3>
              <p className="text-xs text-slate-500 leading-relaxed mb-4">
                An adaptive convolutional gate dynamically balances spatial features against noise residuals before passing through an Atrous Spatial Pyramid Pooling (ASPP) decoder to output the final probability map.
              </p>
            </div>
            <div className="bg-surface-50 p-3 rounded border border-surface-200 font-mono text-[11px] text-slate-600 space-y-1">
              <div>ASPP Rates: [6, 12, 18]</div>
              <div>Loss: BCE + Soft Dice Loss</div>
              <div>Metrics: 0.719 Dice / 0.660 IoU</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
