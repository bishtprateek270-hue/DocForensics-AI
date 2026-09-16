"use client";

import React, { useState } from "react";
import { ForensicReport } from "../../types/forensic";
import { X, Copy, Check, Download, FileText } from "lucide-react";

interface ForensicReportModalProps {
  report: ForensicReport;
  isOpen: boolean;
  onClose: () => void;
}

export const ForensicReportModal: React.FC<ForensicReportModalProps> = ({
  report,
  isOpen,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<"formatted" | "raw">("formatted");
  const [copied, setCopied] = useState<boolean>(false);

  if (!isOpen) return null;

  const jsonString = JSON.stringify(report, null, 2);

  const handleCopyJson = () => {
    navigator.clipboard.writeText(jsonString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadJson = () => {
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `docforensics_${report.session_id.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-white rounded-xl border border-surface-200 shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden">
        {/* Modal Header */}
        <div className="p-4 sm:p-5 border-b border-surface-200 bg-surface-50 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-surface-900 text-white flex items-center justify-center">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <h2 className="font-bold text-base text-surface-900">
                Forensic Analysis Report
              </h2>
              <p className="font-mono text-xs text-slate-500">
                Session ID: {report.session_id}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* View Switcher Tabs */}
            <div className="inline-flex rounded-lg p-0.5 bg-surface-200 border border-surface-300 text-xs">
              <button
                onClick={() => setActiveTab("formatted")}
                className={`px-3 py-1 rounded-md font-medium transition-all ${
                  activeTab === "formatted"
                    ? "bg-white text-surface-900 shadow-subtle font-semibold"
                    : "text-slate-600 hover:text-surface-900"
                }`}
              >
                Audit Report
              </button>
              <button
                onClick={() => setActiveTab("raw")}
                className={`px-3 py-1 rounded-md font-medium transition-all ${
                  activeTab === "raw"
                    ? "bg-white text-surface-900 shadow-subtle font-semibold"
                    : "text-slate-600 hover:text-surface-900"
                }`}
              >
                Raw JSON
              </button>
            </div>

            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-surface-200 rounded-md transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {activeTab === "formatted" ? (
            <div className="space-y-6 text-xs text-slate-700">
              {/* Executive Summary Card */}
              <div className="p-4 rounded-lg border border-surface-200 bg-surface-50">
                <h3 className="font-bold text-sm text-surface-900 mb-2 uppercase tracking-wide">
                  Executive Summary
                </h3>
                <p className="text-sm font-medium text-slate-800 leading-relaxed mb-3">
                  {report.assessment_summary}
                </p>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-surface-200 font-mono text-xs">
                  <div>
                    <span className="text-slate-400 block text-[10px] uppercase">Document</span>
                    <strong className="text-slate-900">{report.filename}</strong>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[10px] uppercase">Status</span>
                    <strong className="text-slate-900 uppercase">
                      {report.analysis_status.replace(/_/g, " ")}
                    </strong>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[10px] uppercase">Max Score</span>
                    <strong className="text-red-700">{report.highest_tampering_score.toFixed(3)}</strong>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[10px] uppercase">Latency</span>
                    <strong className="text-slate-900">
                      {(report.performance_latency.total_pipeline_latency_ms / 1000).toFixed(3)}s
                    </strong>
                  </div>
                </div>
              </div>

              {/* Suspicious Regions Table */}
              <div>
                <h3 className="font-bold text-sm text-surface-900 mb-3 uppercase tracking-wide">
                  Detected Suspicious Regions ({report.suspicious_regions.length})
                </h3>
                {report.suspicious_regions.length === 0 ? (
                  <div className="p-4 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-900 text-xs">
                    No suspicious pixel-level manipulation regions were localized exceeding detection thresholds.
                  </div>
                ) : (
                  <div className="border border-surface-200 rounded-lg overflow-hidden">
                    <table className="w-full text-left border-collapse font-mono text-xs">
                      <thead>
                        <tr className="bg-surface-100 border-b border-surface-200 text-slate-600">
                          <th className="p-2.5">Region ID</th>
                          <th className="p-2.5">Type</th>
                          <th className="p-2.5">Tamper Score (Max/Mean)</th>
                          <th className="p-2.5">Area (%)</th>
                          <th className="p-2.5 font-sans">OCR Extracted Text</th>
                          <th className="p-2.5">OCR Conf</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-surface-200">
                        {report.suspicious_regions.map((r) => (
                          <tr key={r.region_id} className="hover:bg-surface-50">
                            <td className="p-2.5 font-bold">R#{r.region_id}</td>
                            <td className="p-2.5 capitalize">{r.region_type.replace(/_/g, " ")}</td>
                            <td className="p-2.5 text-red-700 font-bold">
                              {r.max_tampering_score.toFixed(3)} / {r.mean_tampering_score.toFixed(3)}
                            </td>
                            <td className="p-2.5">{r.percentage_of_document_area.toFixed(3)}%</td>
                            <td className="p-2.5 font-sans font-medium text-slate-900">
                              {r.has_associated_text && r.ocr_text ? `"${r.ocr_text}"` : "—"}
                            </td>
                            <td className="p-2.5">
                              {r.has_associated_text && r.ocr_confidence !== null
                                ? `${(r.ocr_confidence * 100).toFixed(1)}%`
                                : "N/A"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              {/* Model & System Specifications */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="p-4 rounded-lg border border-surface-200 bg-surface-50 space-y-2">
                  <h4 className="font-bold text-xs text-surface-900 uppercase tracking-wide">
                    Neural Pipeline Metadata
                  </h4>
                  <ul className="space-y-1 font-mono text-[11px] text-slate-600">
                    <li>Architecture: {report.model_architecture}</li>
                    <li>Threshold: {report.inference_threshold}</li>
                    <li>Timestamp (UTC): {report.timestamp_utc}</li>
                    <li>VRAM Peak: {report.performance_latency.peak_vram_mb.toFixed(2)} MB</li>
                  </ul>
                </div>

                <div className="p-4 rounded-lg border border-surface-200 bg-surface-50 space-y-2">
                  <h4 className="font-bold text-xs text-surface-900 uppercase tracking-wide">
                    Forensic Caveats
                  </h4>
                  <ul className="space-y-1 text-[11px] text-slate-500 list-disc list-inside">
                    <li>Results indicate statistical pixel manipulation evidence, not legal authenticity.</li>
                    <li>Heavily degraded or re-compressed documents may produce attenuated noise residuals.</li>
                    <li>Manual review by a certified document examiner is recommended for legal proceedings.</li>
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            /* Raw JSON View */
            <div className="relative">
              <pre className="bg-slate-950 text-slate-50 font-mono text-xs p-4 rounded-lg overflow-x-auto max-h-[500px] leading-relaxed border border-slate-800">
                {jsonString}
              </pre>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-surface-200 bg-surface-50 flex items-center justify-between">
          <div className="text-[11px] text-slate-500">
            Generated deterministically by DocForensics AI Phase 9 Pipeline
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopyJson}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-white text-surface-800 border border-surface-200 hover:bg-surface-100 transition-colors shadow-subtle"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? "Copied!" : "Copy JSON"}</span>
            </button>

            <button
              onClick={handleDownloadJson}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold bg-surface-900 text-white hover:bg-surface-800 transition-colors shadow-subtle"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download JSON</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
