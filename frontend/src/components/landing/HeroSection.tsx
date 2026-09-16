"use client";

import React from "react";
import { motion } from "framer-motion";
import { ArrowRight, ShieldCheck, FileSearch, Layers, Sparkles } from "lucide-react";

export const HeroSection: React.FC = () => {
  return (
    <section className="relative overflow-hidden pt-12 pb-16 md:pt-20 md:pb-24 border-b border-surface-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          {/* Left Text Column */}
          <div className="lg:col-span-7 space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-md text-xs font-semibold bg-surface-100 text-surface-800 border border-surface-200">
              <span className="w-2 h-2 rounded-full bg-blue-600" />
              <span>Dual-Stream RGB + SRM Noise Residual Neural Pipeline</span>
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-surface-900 tracking-tight leading-[1.1]">
              Detect visual manipulation.{" "}
              <span className="text-slate-500 font-normal">
                See exactly where it happened.
              </span>
            </h1>

            <p className="text-base sm:text-lg text-slate-600 leading-relaxed max-w-2xl font-normal">
              DocForensics AI inspects digital documents, certificates, and invoices for localized pixel-level tampering. By fusing RGB visual features with high-pass SRM noise residuals, the system pinpoints modified characters, dates, amounts, and signatures with bounding-box precision and OCR text evidence.
            </p>

            {/* CTAs */}
            <div className="flex flex-wrap items-center gap-4 pt-2">
              <a
                href="#workspace"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-lg text-sm font-semibold bg-surface-900 text-white hover:bg-surface-800 active:scale-[0.99] transition-all shadow-subtle group"
              >
                <span>Analyze a Document</span>
                <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
              </a>

              <a
                href="#how-it-works"
                className="inline-flex items-center gap-2 px-5 py-3 rounded-lg text-sm font-medium bg-white text-surface-800 border border-surface-200 hover:bg-surface-50 active:scale-[0.99] transition-all shadow-subtle"
              >
                <span>See How It Works</span>
              </a>
            </div>

            {/* Key Metric Badges */}
            <div className="grid grid-cols-3 gap-4 pt-6 border-t border-surface-200 max-w-lg">
              <div>
                <div className="text-xl font-bold font-mono text-surface-900">0.719</div>
                <div className="text-xs text-slate-500">Test Dice Score</div>
              </div>
              <div>
                <div className="text-xl font-bold font-mono text-surface-900">30 Filters</div>
                <div className="text-xs text-slate-500">SRM Noise Kernels</div>
              </div>
              <div>
                <div className="text-xl font-bold font-mono text-surface-900">&lt; 350ms</div>
                <div className="text-xs text-slate-500">GPU Inference Latency</div>
              </div>
            </div>
          </div>

          {/* Right Visual Column (Forensic Scanning Simulation) */}
          <div className="lg:col-span-5">
            <div className="relative rounded-xl border border-surface-300 bg-white p-4 shadow-elevation">
              {/* Document Header Bar */}
              <div className="flex items-center justify-between border-b border-surface-200 pb-3 mb-3 text-xs text-slate-500">
                <div className="flex items-center gap-2 font-mono">
                  <span className="w-2.5 h-2.5 rounded-full bg-slate-300" />
                  <span>INVOICE_SAMPLE_984.PNG</span>
                </div>
                <span className="text-[11px] font-semibold text-tamper-dark bg-tamper-light px-2 py-0.5 rounded border border-tamper-border">
                  1 Suspicious Region
                </span>
              </div>

              {/* Document Canvas with Scanning Ray */}
              <div className="relative aspect-[4/3] bg-surface-50 rounded-lg border border-surface-200 p-4 overflow-hidden font-mono text-xs select-none">
                {/* Simulated Document Content */}
                <div className="space-y-2 opacity-80">
                  <div className="h-4 bg-slate-300 rounded w-1/3 mb-4" />
                  <div className="flex justify-between border-b border-slate-200 pb-1">
                    <span className="text-slate-400">Vendor:</span>
                    <span className="text-slate-800">Apex Global Logistics</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-200 pb-1">
                    <span className="text-slate-400">Invoice Date:</span>
                    <span className="text-slate-800">14-Aug-2026</span>
                  </div>
                  <div className="flex justify-between items-center border-b border-slate-200 pb-1 relative">
                    <span className="text-slate-400">Total Billed:</span>
                    {/* Tampered Value with Region Bounding Box */}
                    <div className="relative px-2 py-0.5 rounded border-2 border-red-500 bg-red-50 text-red-700 font-bold">
                      $84,500.00
                      {/* Badge tag */}
                      <span className="absolute -top-3 -right-2 bg-red-600 text-white text-[9px] px-1 rounded uppercase tracking-wider font-sans">
                        R#1 (0.89)
                      </span>
                    </div>
                  </div>
                  <div className="flex justify-between border-b border-slate-200 pb-1">
                    <span className="text-slate-400">Tax ID:</span>
                    <span className="text-slate-800">US-882194-A</span>
                  </div>
                  <div className="h-3 bg-slate-200 rounded w-2/3 mt-3" />
                </div>

                {/* Animated Forensic Scanning Laser */}
                <div className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-red-500 to-transparent shadow-[0_0_12px_rgba(239,68,68,0.8)] animate-scan-line pointer-events-none" />

                {/* Forensic Heatmap Ghost Highlight */}
                <div className="absolute top-[48%] right-[10%] w-28 h-8 rounded bg-red-500/20 blur-sm pointer-events-none animate-forensic-pulse" />
              </div>

              {/* Bottom Forensic Inspector Note */}
              <div className="mt-3 pt-2 border-t border-surface-200 flex items-center justify-between text-[11px] text-slate-500">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-red-500" />
                  <span className="font-medium text-slate-700">OCR Evidence:</span>
                  <span className="font-mono text-slate-800 font-semibold">&quot;$84,500.00&quot;</span>
                </div>
                <span className="font-mono text-slate-400">Score: 0.892</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
