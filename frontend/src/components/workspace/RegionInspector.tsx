"use client";

import React, { useState } from "react";
import { ForensicReport } from "../../types/forensic";
import { getAssetUrl } from "../../lib/api";
import {
  AlertTriangle,
  CheckCircle,
  Type,
  ChevronRight,
  Layers,
  Calculator,
  FileSpreadsheet,
  Database,
  ShieldAlert,
  Info,
} from "lucide-react";

interface RegionInspectorProps {
  report: ForensicReport;
  selectedRegionId: number | null;
  onSelectRegion: (id: number | null) => void;
  onOpenReportModal: () => void;
}

export const RegionInspector: React.FC<RegionInspectorProps> = ({
  report,
  selectedRegionId,
  onSelectRegion,
  onOpenReportModal,
}) => {
  const [activeTab, setActiveTab] = useState<"visual" | "content">("visual");

  const isVisualClean = report.suspicious_region_count === 0;
  const contentAnalysis = report.content_analysis;
  const hasContentInconsistency =
    contentAnalysis?.status === "content_inconsistency_detected";
  const isInspectElementScenario =
    isVisualClean && hasContentInconsistency;

  const checks = contentAnalysis?.checks || [];
  const extractedSubjects =
    contentAnalysis?.extracted_fields?.subjects || [];
  const recordVer = report.record_verification;

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm flex flex-col h-full overflow-hidden">
      {/* Inspector Header */}
      <div className="p-4 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
        <div>
          <h3 className="font-bold text-base text-slate-900">
            Forensic Inspection
          </h3>
          <p className="text-xs text-slate-500">
            Multi-layer tampering &amp; content analysis
          </p>
        </div>

        <button
          onClick={onOpenReportModal}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-900 text-white hover:bg-slate-800 transition-colors shadow-sm"
        >
          <span>Full Report</span>
          <ChevronRight className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Layer Navigation Tabs */}
      <div className="flex border-b border-slate-200 bg-white px-4 pt-2 gap-2">
        <button
          onClick={() => setActiveTab("visual")}
          className={`flex items-center gap-1.5 px-3 py-2 text-xs font-bold border-b-2 transition-all ${
            activeTab === "visual"
              ? "border-blue-600 text-blue-600 bg-blue-50/50 rounded-t-lg"
              : "border-transparent text-slate-600 hover:text-slate-900"
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Visual Forensics</span>
          {report.suspicious_region_count > 0 && (
            <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-red-100 text-red-700 font-mono">
              {report.suspicious_region_count}
            </span>
          )}
        </button>

        <button
          onClick={() => setActiveTab("content")}
          className={`flex items-center gap-1.5 px-3 py-2 text-xs font-bold border-b-2 transition-all ${
            activeTab === "content"
              ? "border-blue-600 text-blue-600 bg-blue-50/50 rounded-t-lg"
              : "border-transparent text-slate-600 hover:text-slate-900"
          }`}
        >
          <Calculator className="w-3.5 h-3.5" />
          <span>Content Consistency</span>
          {hasContentInconsistency ? (
            <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-amber-100 text-amber-800 font-mono font-bold">
              Mismatch
            </span>
          ) : contentAnalysis?.status === "content_consistent" ? (
            <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-emerald-100 text-emerald-700 font-mono">
              Pass
            </span>
          ) : null}
        </button>
      </div>

      {/* Main Tab Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Inspect-Element Alert Banner (Special Cross-Layer Flag) */}
        {isInspectElementScenario && (
          <div className="p-3.5 rounded-xl border border-amber-300 bg-amber-50/90 text-amber-950 text-xs shadow-sm">
            <div className="flex items-start gap-2.5">
              <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
              <div>
                <strong className="block text-amber-900 font-bold mb-0.5">
                  Inspect-Element / Mathematical Discrepancy Found
                </strong>
                <p className="text-amber-800 leading-relaxed">
                  The visual model detected clean pixels with no boundary noise, but Content Consistency discovered mathematical or grading contradictions.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* TAB 1: VISUAL FORENSICS */}
        {activeTab === "visual" && (
          <div className="space-y-4">
            {/* Status Banner */}
            <div
              className={`p-3.5 rounded-xl border text-xs leading-relaxed ${
                isVisualClean
                  ? "bg-emerald-50 border-emerald-200 text-emerald-950"
                  : "bg-red-50 border-red-200 text-red-950"
              }`}
            >
              <div className="flex items-center gap-2 font-bold mb-1">
                {isVisualClean ? (
                  <>
                    <CheckCircle className="w-4 h-4 text-emerald-600 shrink-0" />
                    <span>No Pixel-Level Tampering Detected</span>
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-4 h-4 text-red-600 shrink-0" />
                    <span>Suspicious Visual Manipulation Detected</span>
                  </>
                )}
              </div>
              <p className="opacity-90 leading-relaxed">
                {report.assessment_summary}
              </p>
            </div>

            {/* Aggregate Visual Scores */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-200">
                <div className="text-[11px] text-slate-500 font-medium mb-0.5">
                  Tampering Probability
                </div>
                <div className="text-lg font-bold font-mono text-slate-900">
                  {(report.highest_tampering_score * 100).toFixed(1)}%
                </div>
                <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden mt-1.5">
                  <div
                    className="bg-red-500 h-full rounded-full"
                    style={{
                      width: `${Math.min(
                        report.highest_tampering_score * 100,
                        100
                      )}%`,
                    }}
                  />
                </div>
              </div>

              <div className="bg-slate-50 p-3 rounded-xl border border-slate-200">
                <div className="text-[11px] text-slate-500 font-medium mb-0.5">
                  Suspicious Area
                </div>
                <div className="text-lg font-bold font-mono text-slate-900">
                  {report.total_suspicious_area_percent.toFixed(2)}%
                </div>
                <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden mt-1.5">
                  <div
                    className="bg-amber-500 h-full rounded-full"
                    style={{
                      width: `${Math.min(
                        report.total_suspicious_area_percent * 5,
                        100
                      )}%`,
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Flagged Regions List */}
            {!isVisualClean && (
              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                    Flagged Regions ({report.suspicious_regions.length})
                  </span>
                  {selectedRegionId && (
                    <button
                      onClick={() => onSelectRegion(null)}
                      className="text-xs text-blue-600 font-medium hover:underline"
                    >
                      Clear Selection
                    </button>
                  )}
                </div>

                <div className="space-y-2.5">
                  {report.suspicious_regions.map((region) => {
                    const isSelected = selectedRegionId === region.region_id;

                    return (
                      <div
                        key={region.region_id}
                        onClick={() =>
                          onSelectRegion(
                            isSelected ? null : region.region_id
                          )
                        }
                        className={`p-3 rounded-xl border transition-all cursor-pointer ${
                          isSelected
                            ? "bg-red-50/70 border-red-400 shadow-md ring-2 ring-red-200"
                            : "bg-slate-50/70 border-slate-200 hover:border-slate-300 hover:bg-slate-100/70 shadow-sm"
                        }`}
                      >
                        {/* Region Header */}
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span
                              className={`font-mono text-xs font-bold px-2 py-0.5 rounded-md ${
                                isSelected
                                  ? "bg-red-600 text-white"
                                  : "bg-slate-200 text-slate-800"
                              }`}
                            >
                              Area #{region.region_id}
                            </span>
                            <span className="text-xs text-slate-600 font-semibold capitalize">
                              {region.region_type.replace(/_/g, " ")}
                            </span>
                          </div>

                          <span className="font-mono text-xs font-bold text-red-700 bg-red-100 px-2 py-0.5 rounded">
                            {(region.max_tampering_score * 100).toFixed(0)}% Match
                          </span>
                        </div>

                        {/* OCR Text Box */}
                        <div className="bg-white p-2.5 rounded-lg border border-slate-200 mb-2">
                          <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                            <span className="flex items-center gap-1 font-bold uppercase tracking-wider text-slate-700">
                              <Type className="w-3 h-3 text-blue-600" />
                              Detected Text
                            </span>
                            {region.has_associated_text && (
                              <span className="font-mono text-slate-500 font-medium">
                                Conf: {(region.ocr_confidence * 100).toFixed(0)}%
                              </span>
                            )}
                          </div>
                          {region.has_associated_text && region.ocr_text ? (
                            <div className="font-mono text-xs font-bold text-slate-900 bg-slate-50 px-2 py-1 rounded border border-slate-200 break-words">
                              &ldquo;{region.ocr_text}&rdquo;
                            </div>
                          ) : (
                            <div className="text-xs italic text-slate-400">
                              No text detected in this area
                            </div>
                          )}
                        </div>

                        {/* Region Crop */}
                        {isSelected && (
                          <div className="pt-2 border-t border-red-200 mt-2 flex items-center gap-3">
                            {region.crop_url && (
                              <div className="w-24 h-16 bg-slate-100 rounded border border-slate-200 overflow-hidden shrink-0 flex items-center justify-center">
                                <img
                                  src={getAssetUrl(region.crop_url)}
                                  alt={`Area #${region.region_id} crop`}
                                  className="max-h-full max-w-full object-contain"
                                />
                              </div>
                            )}
                            <div className="text-xs text-slate-600 space-y-0.5">
                              <div>
                                <span className="text-slate-400">Area Fraction:</span>{" "}
                                <strong>
                                  {region.percentage_of_document_area.toFixed(2)}%
                                </strong>
                              </div>
                              <div>
                                <span className="text-slate-400">Status:</span>{" "}
                                <strong className="text-red-700">Flagged For Review</strong>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: CONTENT CONSISTENCY */}
        {activeTab === "content" && (
          <div className="space-y-4">
            {/* Document Classification */}
            <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
                  <FileSpreadsheet className="w-3.5 h-3.5 text-blue-600" />
                  Document Category
                </span>
                <span className="font-mono text-xs text-slate-600">
                  Confidence: {((contentAnalysis?.document_type_confidence || 0.8) * 100).toFixed(0)}%
                </span>
              </div>
              <div className="text-sm font-bold text-slate-900 capitalize">
                {contentAnalysis?.document_type?.replace(/_/g, " ") || "Generic Document"}
              </div>
            </div>

            {/* Consistency Summary Banner */}
            <div
              className={`p-3.5 rounded-xl border text-xs leading-relaxed ${
                hasContentInconsistency
                  ? "bg-amber-50 border-amber-200 text-amber-950"
                  : contentAnalysis?.status === "content_consistent"
                  ? "bg-emerald-50 border-emerald-200 text-emerald-950"
                  : "bg-slate-50 border-slate-200 text-slate-800"
              }`}
            >
              <div className="flex items-center gap-2 font-bold mb-1">
                {hasContentInconsistency ? (
                  <>
                    <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                    <span>Content Inconsistency Detected</span>
                  </>
                ) : contentAnalysis?.status === "content_consistent" ? (
                  <>
                    <CheckCircle className="w-4 h-4 text-emerald-600 shrink-0" />
                    <span>Content Consistency Verified</span>
                  </>
                ) : (
                  <>
                    <Info className="w-4 h-4 text-slate-500 shrink-0" />
                    <span>Document Structure Analysis</span>
                  </>
                )}
              </div>
              <p className="opacity-90 leading-relaxed">
                {contentAnalysis?.summary || "No mathematical relationships configured for this document format."}
              </p>
            </div>

            {/* Checks List */}
            {checks.length > 0 && (
              <div className="space-y-2.5">
                <div className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                  Deterministic Checks ({checks.length})
                </div>

                <div className="space-y-2.5">
                  {checks.map((c) => {
                    const isMismatch =
                      c.status === "mismatch" || c.status === "impossible_value";

                    return (
                      <div
                        key={c.check_id}
                        className={`p-3 rounded-xl border text-xs shadow-sm ${
                          isMismatch
                            ? "bg-red-50/70 border-red-300"
                            : "bg-slate-50/70 border-slate-200"
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-bold text-slate-900">
                            {c.check_name}
                          </span>
                          <span
                            className={`px-2 py-0.5 rounded font-mono text-[11px] font-bold ${
                              c.status === "match"
                                ? "bg-emerald-100 text-emerald-800"
                                : c.status === "impossible_value"
                                ? "bg-purple-100 text-purple-800"
                                : "bg-red-100 text-red-800"
                            }`}
                          >
                            {c.status.replace(/_/g, " ").toUpperCase()}
                          </span>
                        </div>

                        {/* Comparison Box */}
                        <div className="grid grid-cols-2 gap-2 bg-white p-2 rounded-lg border border-slate-200 mb-2 font-mono">
                          <div>
                            <span className="text-[10px] text-slate-400 block font-sans">
                              Displayed
                            </span>
                            <strong className="text-slate-900 text-xs">
                              {c.displayed_value || "—"}
                            </strong>
                          </div>
                          <div>
                            <span className="text-[10px] text-slate-400 block font-sans">
                              Calculated
                            </span>
                            <strong
                              className={`text-xs ${
                                isMismatch ? "text-red-600 font-bold" : "text-emerald-700 font-bold"
                              }`}
                            >
                              {c.calculated_value || "—"}
                            </strong>
                          </div>
                        </div>

                        <p className="text-slate-600 text-[11px] leading-relaxed">
                          {c.explanation}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Extracted Subjects Table (if available) */}
            {extractedSubjects.length > 0 && (
              <div className="space-y-2">
                <div className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                  Extracted Course Table ({extractedSubjects.length} subjects)
                </div>

                <div className="border border-slate-200 rounded-xl overflow-x-auto shadow-sm">
                  <table className="w-full text-left border-collapse text-[11px]">
                    <thead>
                      <tr className="bg-slate-100 border-b border-slate-200 text-slate-700 font-bold">
                        <th className="p-2">Subject</th>
                        <th className="p-2">Credit</th>
                        <th className="p-2">Grade</th>
                        <th className="p-2">GP</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 font-mono">
                      {extractedSubjects.map((s: any, idx: number) => (
                        <tr key={idx} className="hover:bg-slate-50">
                          <td className="p-2 font-sans font-medium text-slate-800">
                            {s.code || s.name || `Subject ${idx + 1}`}
                          </td>
                          <td className="p-2">{s.credits ?? "—"}</td>
                          <td className="p-2 font-bold text-slate-900">{s.grade ?? "—"}</td>
                          <td className="p-2 text-blue-600">{s.grade_point ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Authoritative Record Verification */}
            {recordVer && (
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-xs">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-bold text-slate-600 flex items-center gap-1.5">
                    <Database className="w-3.5 h-3.5 text-slate-500" />
                    Record Verification
                  </span>
                  <span className="font-mono text-[10px] text-slate-500 uppercase px-1.5 py-0.5 bg-slate-200 rounded">
                    {recordVer.status.replace(/_/g, " ")}
                  </span>
                </div>
                <p className="text-slate-600 text-[11px] leading-relaxed">
                  {recordVer.message}
                </p>
              </div>
            )}
          </div>
        )}

        {/* Processing Performance Pill */}
        <div className="p-2.5 bg-slate-50 rounded-xl border border-slate-200 text-xs text-slate-500 flex items-center justify-between">
          <span>Inspection Latency:</span>
          <strong className="font-mono text-slate-800">
            {(report.performance_latency.total_pipeline_latency_ms / 1000).toFixed(2)}s
          </strong>
        </div>
      </div>
    </div>
  );
};
