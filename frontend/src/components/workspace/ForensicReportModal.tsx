"use client";

import React, { useState } from "react";
import { ForensicReport } from "../../types/forensic";
import { X, Copy, Check, Download, FileText, CheckCircle, AlertTriangle } from "lucide-react";

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
    a.download = `docforensics_report_${report.session_id.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-white rounded-2xl border border-slate-200 shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden">
        {/* Modal Header */}
        <div className="p-5 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-slate-900 text-white flex items-center justify-center shadow-sm">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <h2 className="font-bold text-lg text-slate-900">
                Document Forensic Report
              </h2>
              <p className="font-mono text-xs text-slate-500">
                {report.filename} &bull; ID: {report.session_id.slice(0, 8)}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Tab Switcher */}
            <div className="inline-flex rounded-xl p-1 bg-slate-200 border border-slate-300 text-xs">
              <button
                onClick={() => setActiveTab("formatted")}
                className={`px-3.5 py-1.5 rounded-lg font-semibold transition-all ${
                  activeTab === "formatted"
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Audit Report
              </button>
              <button
                onClick={() => setActiveTab("raw")}
                className={`px-3.5 py-1.5 rounded-lg font-semibold transition-all ${
                  activeTab === "raw"
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Raw Data (JSON)
              </button>
            </div>

            <button
              onClick={onClose}
              className="p-2 text-slate-400 hover:text-slate-700 hover:bg-slate-200 rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {activeTab === "formatted" ? (
            <div className="space-y-6 text-sm text-slate-700">
              {/* Executive Summary Card */}
              <div className="p-5 rounded-xl border border-slate-200 bg-slate-50">
                <h3 className="font-bold text-xs text-slate-500 uppercase tracking-wider mb-2">
                  Executive Summary
                </h3>
                <p className="text-base font-semibold text-slate-900 leading-relaxed mb-4">
                  {report.assessment_summary}
                </p>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-4 border-t border-slate-200 text-xs">
                  <div>
                    <span className="text-slate-400 block mb-0.5">Document File</span>
                    <strong className="text-slate-900">{report.filename}</strong>
                  </div>
                  <div>
                    <span className="text-slate-400 block mb-0.5">Verdict</span>
                    <strong className="text-slate-900 uppercase">
                      {report.analysis_status.replace(/_/g, " ")}
                    </strong>
                  </div>
                  <div>
                    <span className="text-slate-400 block mb-0.5">Tampering Confidence</span>
                    <strong className="text-red-700 font-mono text-sm">
                      {(report.highest_tampering_score * 100).toFixed(1)}%
                    </strong>
                  </div>
                  <div>
                    <span className="text-slate-400 block mb-0.5">Analysis Time</span>
                    <strong className="text-slate-900 font-mono">
                      {(report.performance_latency.total_pipeline_latency_ms / 1000).toFixed(2)}s
                    </strong>
                  </div>
                </div>
              </div>

              {/* Suspicious Regions Table */}
              <div>
                <h3 className="font-bold text-xs text-slate-500 uppercase tracking-wider mb-3">
                  Localized Suspicious Areas ({report.suspicious_regions.length})
                </h3>
                {report.suspicious_regions.length === 0 ? (
                  <div className="p-5 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-950 text-sm flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-emerald-600 shrink-0" />
                    <span>No localized visual manipulation evidence was detected on this document.</span>
                  </div>
                ) : (
                  <div className="border border-slate-200 rounded-xl overflow-hidden shadow-sm">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="bg-slate-100 border-b border-slate-200 text-slate-700 font-bold">
                          <th className="p-3">Area ID</th>
                          <th className="p-3">Type</th>
                          <th className="p-3">Tampering Probability</th>
                          <th className="p-3">Document Coverage</th>
                          <th className="p-3">Extracted Text Evidence</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 font-mono">
                        {report.suspicious_regions.map((r) => (
                          <tr key={r.region_id} className="hover:bg-slate-50">
                            <td className="p-3 font-bold">Area #{r.region_id}</td>
                            <td className="p-3 font-sans capitalize">{r.region_type.replace(/_/g, " ")}</td>
                            <td className="p-3 text-red-700 font-bold">
                              {(r.max_tampering_score * 100).toFixed(1)}%
                            </td>
                            <td className="p-3">{r.percentage_of_document_area.toFixed(2)}%</td>
                            <td className="p-3 font-sans font-bold text-slate-900">
                              {r.has_associated_text && r.ocr_text ? `"${r.ocr_text}"` : "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              {/* Notice */}
              <div className="p-4 rounded-xl border border-slate-200 bg-slate-50 text-xs text-slate-500">
                <strong className="text-slate-700">Notice:</strong> Results indicate statistical visual manipulation patterns. For official legal or forensic proceedings, a certified document examiner should conduct corroborating manual analysis.
              </div>
            </div>
          ) : (
            /* Raw JSON View */
            <div className="relative">
              <pre className="bg-slate-950 text-slate-50 font-mono text-xs p-5 rounded-xl overflow-x-auto max-h-[500px] leading-relaxed border border-slate-800">
                {jsonString}
              </pre>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="p-4 sm:p-5 border-t border-slate-200 bg-slate-50 flex items-center justify-between">
          <div className="text-xs text-slate-500 hidden sm:block">
            DocForensics AI Verification Report
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleCopyJson}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-white text-slate-800 border border-slate-200 hover:bg-slate-100 transition-colors shadow-sm"
            >
              {copied ? <Check className="w-4 h-4 text-emerald-600" /> : <Copy className="w-4 h-4" />}
              <span>{copied ? "Copied!" : "Copy JSON"}</span>
            </button>

            <button
              onClick={handleDownloadJson}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-slate-900 text-white hover:bg-slate-800 transition-colors shadow-sm"
            >
              <Download className="w-4 h-4" />
              <span>Download Report</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
