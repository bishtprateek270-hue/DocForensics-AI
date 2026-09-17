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
  Download,
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
  const evSummary = report.evidence_summary;

  const checks = contentAnalysis?.checks || [];
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
            Dual-Specialist Evidence &amp; Verification
          </p>
        </div>

        <button
          onClick={onOpenReportModal}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-900 text-white hover:bg-slate-800 transition-colors shadow-sm"
        >
          <span>Full Dossier</span>
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
            <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-blue-100 text-blue-800 font-mono">
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
          ) : (
            <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-emerald-100 text-emerald-800 font-mono">
              OK
            </span>
          )}
        </button>
      </div>

      {/* Tab Content */}
      <div className="p-4 flex-1 overflow-y-auto space-y-4">
        {/* TAB 1: VISUAL FORENSICS */}
        {activeTab === "visual" && (
          <div className="space-y-4">
            {/* Multi-Evidence Status Banner */}
            <div
              className={`p-3.5 rounded-xl border text-xs leading-relaxed ${
                isVisualClean
                  ? "bg-emerald-50 border-emerald-200 text-emerald-950"
                  : "bg-blue-50/70 border-blue-200 text-slate-900"
              }`}
            >
              <div className="flex items-center gap-2 font-bold mb-1">
                {isVisualClean ? (
                  <>
                    <CheckCircle className="w-4 h-4 text-emerald-600 shrink-0" />
                    <span>No Significant Visual Manipulation Evidence Detected</span>
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-4 h-4 text-blue-600 shrink-0" />
                    <span>Suspicious Visual Forensic Findings Detected</span>
                  </>
                )}
              </div>
              <p className="opacity-90 leading-relaxed">
                {report.assessment_summary}
              </p>
            </div>

            {/* Specialist Breakdown Cards */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-200">
                <div className="text-[11px] text-slate-500 font-medium mb-0.5">
                  Physical / Splicing
                </div>
                <div className="text-sm font-bold text-slate-900 flex items-center justify-between">
                  <span>{evSummary?.physical_region_count ?? 0} Regions</span>
                  {evSummary?.physical_visual_evidence ? (
                    <span className="text-[11px] font-bold text-blue-600 bg-blue-100 px-1.5 py-0.5 rounded">
                      Traces
                    </span>
                  ) : (
                    <span className="text-[11px] text-emerald-600">Clean</span>
                  )}
                </div>
              </div>

              <div className="bg-slate-50 p-3 rounded-xl border border-slate-200">
                <div className="text-[11px] text-slate-500 font-medium mb-0.5">
                  Tiny-Text Digital
                </div>
                <div className="text-sm font-bold text-slate-900 flex items-center justify-between">
                  <span>{evSummary?.text_region_count ?? 0} Regions</span>
                  {evSummary?.digital_text_visual_evidence ? (
                    <span className="text-[11px] font-bold text-purple-600 bg-purple-100 px-1.5 py-0.5 rounded">
                      Patterns
                    </span>
                  ) : (
                    <span className="text-[11px] text-emerald-600">Clean</span>
                  )}
                </div>
              </div>
            </div>

            {/* Flagged Regions List */}
            {!isVisualClean && (
              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                    Suspicious Regions ({report.suspicious_regions.length})
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
                    const sources = region.evidence_sources || ["Visual Specialist"];
                    const isDual = sources.length > 1;

                    return (
                      <div
                        key={region.region_id}
                        onClick={() =>
                          onSelectRegion(isSelected ? null : region.region_id)
                        }
                        className={`p-3 rounded-xl border transition-all cursor-pointer ${
                          isSelected
                            ? "bg-blue-50 border-blue-400 shadow-md ring-2 ring-blue-200"
                            : "bg-slate-50/70 border-slate-200 hover:border-slate-300 hover:bg-slate-100/70 shadow-sm"
                        }`}
                      >
                        {/* Region Header */}
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span
                              className={`font-mono text-xs font-bold px-2 py-0.5 rounded-md ${
                                isSelected
                                  ? "bg-blue-600 text-white"
                                  : "bg-slate-200 text-slate-800"
                              }`}
                            >
                              #{region.region_id}
                            </span>
                            <span className="text-xs text-slate-700 font-semibold">
                              {sources.join(" + ")}
                            </span>
                          </div>

                          <span className="font-mono text-xs font-bold text-slate-900 bg-slate-200 px-2 py-0.5 rounded">
                            Score: {(region.evidence_score ?? region.mean_tampering_score).toFixed(2)}
                          </span>
                        </div>

                        {/* OCR Text Box */}
                        {region.has_associated_text && region.ocr_text && (
                          <div className="bg-white p-2 rounded-lg border border-slate-200 mb-2 text-xs">
                            <div className="flex items-center gap-1 font-bold text-slate-500 text-[10px] uppercase mb-0.5">
                              <Type className="w-3 h-3 text-blue-600" />
                              Associated OCR Text
                            </div>
                            <div className="font-mono font-bold text-slate-900">
                              &ldquo;{region.ocr_text}&rdquo;
                            </div>
                          </div>
                        )}

                        {/* Explanation */}
                        {region.explanation && (
                          <div className="text-[11px] text-slate-600 leading-snug">
                            {region.explanation}
                          </div>
                        )}

                        {/* Region Crop on Selection */}
                        {isSelected && (
                          <div className="pt-2 border-t border-blue-200 mt-2 flex items-center gap-3">
                            {region.crop_url && (
                              <div className="w-24 h-16 bg-slate-100 rounded border border-slate-200 overflow-hidden shrink-0 flex items-center justify-center">
                                <img
                                  src={getAssetUrl(region.crop_url)}
                                  alt={`Region #${region.region_id} crop`}
                                  className="max-h-full max-w-full object-contain"
                                />
                              </div>
                            )}
                            <div className="text-xs text-slate-600 space-y-0.5">
                              <div>
                                <span className="text-slate-400">Area Coverage:</span>{" "}
                                <strong>
                                  {region.percentage_of_document_area.toFixed(2)}%
                                </strong>
                              </div>
                              <div>
                                <span className="text-slate-400">Coordinates:</span>{" "}
                                <strong className="font-mono">
                                  [{region.bbox.join(", ")}]
                                </strong>
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
            {/* Category Banner */}
            <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200">
              <div className="flex items-center justify-between mb-1">
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

            {/* Consistency Summary */}
            <div
              className={`p-3.5 rounded-xl border text-xs leading-relaxed ${
                hasContentInconsistency
                  ? "bg-amber-50 border-amber-200 text-amber-950"
                  : "bg-emerald-50 border-emerald-200 text-emerald-950"
              }`}
            >
              <div className="flex items-center gap-2 font-bold mb-1">
                {hasContentInconsistency ? (
                  <>
                    <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                    <span>Content Inconsistency Detected</span>
                  </>
                ) : (
                  <>
                    <CheckCircle className="w-4 h-4 text-emerald-600 shrink-0" />
                    <span>Deterministic Content Rules Consistent</span>
                  </>
                )}
              </div>
              <p className="opacity-90 leading-relaxed">
                {contentAnalysis?.summary || "Deterministic checks complete."}
              </p>
            </div>

            {/* Checks List */}
            {checks.length > 0 && (
              <div className="space-y-2">
                <span className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                  Verification Checks ({checks.length})
                </span>
                {checks.map((c) => (
                  <div
                    key={c.check_id}
                    className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-xs space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-slate-900">{c.check_name}</span>
                      <span
                        className={`font-bold font-mono text-[10px] px-2 py-0.5 rounded ${
                          c.status === "match"
                            ? "bg-emerald-100 text-emerald-800"
                            : "bg-red-100 text-red-800"
                        }`}
                      >
                        {c.status.toUpperCase()}
                      </span>
                    </div>
                    <div className="text-slate-600 text-[11px]">{c.explanation}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
