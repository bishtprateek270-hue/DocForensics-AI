"use client";

import React from "react";
import { ArrowRight, Search, CheckCircle2, ShieldCheck } from "lucide-react";

export const HeroSection: React.FC = () => {
  return (
    <section className="relative overflow-hidden pt-12 pb-16 md:pt-20 md:pb-24 border-b border-slate-200 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-16 items-center">
          {/* Left Text Column */}
          <div className="lg:col-span-7 space-y-6">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold bg-slate-100 text-slate-800 border border-slate-200">
              <span className="w-2 h-2 rounded-full bg-blue-600 animate-pulse" />
              <span>AI-Powered Document Forensics</span>
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-slate-900 tracking-tight leading-[1.1]">
              Detect visual manipulation.{" "}
              <span className="text-slate-500 font-normal">
                See exactly where it happened.
              </span>
            </h1>

            <p className="text-lg sm:text-xl text-slate-600 leading-relaxed font-normal max-w-2xl">
              DocForensics AI analyzes visual patterns in a document to locate areas that may have been digitally modified. Upload invoices, IDs, contracts, or receipts to pinpoint altered amounts, modified dates, substituted photos, and forged signatures.
            </p>

            {/* Main Action CTAs */}
            <div className="flex flex-wrap items-center gap-4 pt-2">
              <a
                href="#workspace"
                className="inline-flex items-center gap-2.5 px-7 py-3.5 rounded-xl text-base font-semibold bg-slate-900 text-white hover:bg-slate-800 active:scale-[0.99] transition-all shadow-md group"
              >
                <span>Analyze a Document</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </a>

              <a
                href="#how-it-works"
                className="inline-flex items-center gap-2 px-6 py-3.5 rounded-xl text-base font-medium bg-slate-50 text-slate-800 border border-slate-200 hover:bg-slate-100 active:scale-[0.99] transition-all"
              >
                <span>See How It Works</span>
              </a>
            </div>

            {/* Key Benefits */}
            <div className="grid grid-cols-3 gap-6 pt-8 border-t border-slate-200">
              <div>
                <div className="text-sm font-bold text-slate-900">Precise Localization</div>
                <div className="text-xs text-slate-500 mt-0.5">Exact bounding-box regions</div>
              </div>
              <div>
                <div className="text-sm font-bold text-slate-900">Text Evidence</div>
                <div className="text-xs text-slate-500 mt-0.5">Automated OCR extraction</div>
              </div>
              <div>
                <div className="text-sm font-bold text-slate-900">Instant Reports</div>
                <div className="text-xs text-slate-500 mt-0.5">Detailed forensic audit sheet</div>
              </div>
            </div>
          </div>

          {/* Right Visual Column (Document Inspection Mockup) */}
          <div className="lg:col-span-5">
            <div className="relative rounded-2xl border border-slate-300 bg-white p-5 shadow-lg">
              {/* Document Header Bar */}
              <div className="flex items-center justify-between border-b border-slate-200 pb-3 mb-4 text-xs text-slate-500">
                <div className="flex items-center gap-2 font-mono">
                  <span className="w-2.5 h-2.5 rounded-full bg-slate-300" />
                  <span className="font-semibold text-slate-700">Sample_Invoice_884.pdf</span>
                </div>
                <span className="text-[11px] font-semibold text-red-700 bg-red-50 px-2.5 py-1 rounded-md border border-red-200">
                  1 Suspicious Region
                </span>
              </div>

              {/* Document Canvas with Scan Line */}
              <div className="relative aspect-[4/3] bg-slate-50 rounded-xl border border-slate-200 p-5 overflow-hidden font-mono text-xs select-none shadow-inner">
                {/* Simulated Document Body */}
                <div className="space-y-3 opacity-90">
                  <div className="h-4 bg-slate-300 rounded w-2/5 mb-5" />
                  <div className="flex justify-between border-b border-slate-200 pb-1.5">
                    <span className="text-slate-400">Vendor:</span>
                    <span className="text-slate-800 font-medium">Apex Global Logistics</span>
                  </div>
                  <div className="flex justify-between border-b border-slate-200 pb-1.5">
                    <span className="text-slate-400">Issue Date:</span>
                    <span className="text-slate-800 font-medium">14-Aug-2026</span>
                  </div>
                  <div className="flex justify-between items-center border-b border-slate-200 pb-1.5 relative">
                    <span className="text-slate-400">Total Amount:</span>
                    {/* Highlighted Altered Amount */}
                    <div className="relative px-2.5 py-1 rounded border-2 border-red-500 bg-red-50 text-red-700 font-bold text-sm">
                      $84,500.00
                      {/* Region Tag */}
                      <span className="absolute -top-3.5 -right-2.5 bg-red-600 text-white text-[9px] px-1.5 py-0.5 rounded font-sans uppercase font-bold shadow-sm">
                        Altered
                      </span>
                    </div>
                  </div>
                  <div className="flex justify-between border-b border-slate-200 pb-1.5">
                    <span className="text-slate-400">Tax ID:</span>
                    <span className="text-slate-800 font-medium">US-882194-A</span>
                  </div>
                  <div className="h-3 bg-slate-200 rounded w-3/4 mt-4" />
                </div>

                {/* Subtle Scanning Line Animation */}
                <div className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-red-500 to-transparent shadow-[0_0_12px_rgba(239,68,68,0.9)] animate-scan-line pointer-events-none" />

                {/* Heatmap Ghost */}
                <div className="absolute top-[48%] right-[8%] w-32 h-10 rounded bg-red-500/20 blur-sm pointer-events-none animate-forensic-pulse" />
              </div>

              {/* Bottom Card Summary */}
              <div className="mt-4 pt-3 border-t border-slate-200 flex items-center justify-between text-xs text-slate-600">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-red-500" />
                  <span className="font-semibold text-slate-800">Detected Text:</span>
                  <span className="font-mono text-slate-900 bg-slate-100 px-1.5 py-0.5 rounded">&ldquo;$84,500.00&rdquo;</span>
                </div>
                <span className="text-slate-500 font-medium">High Confidence</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
