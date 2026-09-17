"use client";

import React from "react";
import { ForensicReport } from "../../types/forensic";
import { getAssetUrl } from "../../lib/api";
import {
  X,
  Download,
  FileText,
  CheckCircle,
  AlertTriangle,
  Calculator,
  Layers,
  ShieldCheck,
  ShieldAlert,
  Info,
  ExternalLink,
} from "lucide-react";

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
  if (!isOpen) return null;

  const contentAnalysis = report.content_analysis;
  const checks = contentAnalysis?.checks || [];
  const extractedSubjects = contentAnalysis?.extracted_fields?.subjects || [];
  const evSummary = report.evidence_summary;

  const handleDownloadPdf = () => {
    const pdfUrl = getAssetUrl(report.pdf_report_url || `/api/analysis/${report.session_id}/pdf-report`);
    window.open(pdfUrl, "_blank");
  };

  const handleDownloadText = () => {
    const reportText = `======================================================================
DOCFORENSICS AI - DUAL-SPECIALIST FORENSIC REPORT
======================================================================
Document File:   ${report.filename}
Session ID:      ${report.session_id}
Date & Time:     ${report.timestamp_utc}
Verdict:         ${report.assessment_summary}

----------------------------------------------------------------------
1. MULTI-EVIDENCE CHANNEL SUMMARY
----------------------------------------------------------------------
- Physical / Image Visual Evidence: ${evSummary?.physical_visual_evidence ? "Visual Traces Present" : "No Significant Evidence"}
- Tiny-Text Visual Pattern Evidence: ${evSummary?.digital_text_visual_evidence ? "Visual Patterns Present" : "No Significant Evidence"}
- Content Consistency:              ${evSummary?.content_inconsistency ? "Inconsistency Detected" : "Consistent"}
- Reference Verification:           ${report.record_verification?.status || "Not Available"}

----------------------------------------------------------------------
2. LOCALIZED SUSPICIOUS REGIONS (${report.suspicious_regions.length})
----------------------------------------------------------------------
${report.suspicious_regions.length === 0 ? "No suspicious regions detected." : report.suspicious_regions.map((r) => `
[Region #${r.region_id}]
- Contributing Sources: ${(r.evidence_sources || ["Visual Specialist"]).join(", ")}
- Bounding Box [x1,y1,x2,y2]: [${r.bbox.join(", ")}]
- Evidence Score:       ${(r.evidence_score ?? r.mean_tampering_score).toFixed(3)}
- Associated Text:      ${r.has_associated_text && r.ocr_text ? `"${r.ocr_text}"` : "None"}
- Explanation:          ${r.explanation || "Visual anomaly detected"}
`).join("")}

----------------------------------------------------------------------
3. CONTENT & MATHEMATICAL CHECKS
----------------------------------------------------------------------
${checks.length === 0 ? "No deterministic checks executed." : checks.map((c) => `
[Check: ${c.check_name}]
- Status:     ${c.status.toUpperCase()}
- Displayed:  ${c.displayed_value || "—"}
- Calculated: ${c.calculated_value || "—"}
- Detail:     ${c.explanation}
`).join("")}

----------------------------------------------------------------------
DISCLAIMER:
These findings indicate potential visual or content inconsistencies and
should not be interpreted as definitive proof of document authenticity or fraud.
======================================================================
`;

    const blob = new Blob([reportText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `docforensics_report_${report.session_id.slice(0, 8)}.txt`;
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
                Dual-Specialist Forensic Dossier
              </h2>
              <p className="font-mono text-xs text-slate-500">
                {report.filename} &bull; ID: {report.session_id.slice(0, 8)}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleDownloadPdf}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-blue-600 text-white hover:bg-blue-700 transition-colors shadow-sm"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download PDF Report</span>
            </button>
            <button
              onClick={handleDownloadText}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-200 text-slate-700 hover:bg-slate-300 transition-colors"
            >
              <span>TXT</span>
            </button>
            <button
              onClick={onClose}
              className="p-2 text-slate-400 hover:text-slate-700 hover:bg-slate-200 rounded-lg transition-colors ml-1"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 text-sm text-slate-700">
          {/* Executive Summary Card */}
          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-2">
            <div className="flex items-center gap-2 font-bold text-slate-900">
              <Info className="w-4 h-4 text-blue-600" />
              <span>Forensic Assessment</span>
            </div>
            <p className="text-xs leading-relaxed text-slate-700">
              {report.assessment_summary}
            </p>
          </div>

          {/* Evidence Channels Grid */}
          <div>
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">
              1. Multi-Evidence Channel Status
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <div className="text-[11px] text-slate-500 font-medium">Physical Splicing</div>
                <div className="text-xs font-bold mt-1 text-slate-900">
                  {evSummary?.physical_visual_evidence ? (
                    <span className="text-red-600">Traces Present</span>
                  ) : (
                    <span className="text-emerald-600">No Significant Evidence</span>
                  )}
                </div>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <div className="text-[11px] text-slate-500 font-medium">Tiny-Text Digital</div>
                <div className="text-xs font-bold mt-1 text-slate-900">
                  {evSummary?.digital_text_visual_evidence ? (
                    <span className="text-purple-600">Patterns Present</span>
                  ) : (
                    <span className="text-emerald-600">No Significant Evidence</span>
                  )}
                </div>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <div className="text-[11px] text-slate-500 font-medium">Content Consistency</div>
                <div className="text-xs font-bold mt-1 text-slate-900">
                  {evSummary?.content_inconsistency ? (
                    <span className="text-amber-600">Inconsistency</span>
                  ) : (
                    <span className="text-emerald-600">Consistent</span>
                  )}
                </div>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <div className="text-[11px] text-slate-500 font-medium">Reference Match</div>
                <div className="text-xs font-bold mt-1 text-slate-500">
                  {report.record_verification?.status === "verified_match" ? (
                    <span className="text-emerald-600">Verified</span>
                  ) : report.record_verification?.status === "authoritative_mismatch" ? (
                    <span className="text-red-600">Mismatch</span>
                  ) : (
                    <span>Not Configured</span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Localized Suspicious Regions */}
          <div>
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">
              2. Suspicious Visual Findings ({report.suspicious_regions.length})
            </h3>
            {report.suspicious_regions.length === 0 ? (
              <p className="text-xs italic text-slate-400">
                No suspicious regions detected above decision threshold.
              </p>
            ) : (
              <div className="space-y-2">
                {report.suspicious_regions.map((r) => (
                  <div
                    key={r.region_id}
                    className="p-3 bg-slate-50 rounded-xl border border-slate-200 flex items-center justify-between gap-4 text-xs"
                  >
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold px-1.5 py-0.5 bg-slate-200 text-slate-800 rounded">
                          #{r.region_id}
                        </span>
                        <span className="font-semibold text-slate-900">
                          {(r.evidence_sources || ["Visual Specialist"]).join(" + ")}
                        </span>
                      </div>
                      {r.has_associated_text && r.ocr_text && (
                        <div className="font-mono text-slate-600">
                          OCR: &ldquo;{r.ocr_text}&rdquo;
                        </div>
                      )}
                      <div className="text-slate-500 text-[11px]">
                        BBox: [{r.bbox.join(", ")}] &bull; Area: {r.percentage_of_document_area.toFixed(2)}%
                      </div>
                    </div>
                    <div className="text-right shrink-0">
                      <div className="text-[11px] text-slate-400">Evidence Score</div>
                      <div className="font-mono font-bold text-sm text-slate-900">
                        {(r.evidence_score ?? r.mean_tampering_score).toFixed(3)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Content Consistency Table */}
          {checks.length > 0 && (
            <div>
              <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">
                3. Content Consistency &amp; Semantic Rules
              </h3>
              <div className="border border-slate-200 rounded-xl overflow-hidden text-xs">
                <table className="w-full text-left">
                  <thead className="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
                    <tr>
                      <th className="p-2.5">Rule / Check</th>
                      <th className="p-2.5">Extracted Value</th>
                      <th className="p-2.5">Calculated</th>
                      <th className="p-2.5 text-right">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {checks.map((c) => (
                      <tr key={c.check_id} className="bg-white hover:bg-slate-50">
                        <td className="p-2.5 font-medium">{c.check_name}</td>
                        <td className="p-2.5 font-mono">{c.displayed_value || "—"}</td>
                        <td className="p-2.5 font-mono">{c.calculated_value || "—"}</td>
                        <td className="p-2.5 text-right font-bold">
                          {c.status === "match" ? (
                            <span className="text-emerald-600">MATCH</span>
                          ) : (
                            <span className="text-red-600">{c.status.toUpperCase()}</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Conservative Disclaimer */}
          <div className="p-3.5 bg-slate-100 rounded-xl border border-slate-200 text-[11px] text-slate-500 leading-relaxed">
            <strong>Disclaimer:</strong> {evSummary?.disclaimer || "These findings indicate potential visual or content inconsistencies and should not be interpreted as definitive proof of document authenticity or fraud."}
          </div>
        </div>
      </div>
    </div>
  );
};
