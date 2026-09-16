"use client";

import React from "react";
import { ForensicReport, SuspiciousRegion } from "../../types/forensic";
import { getAssetUrl } from "../../lib/api";
import {
  AlertTriangle,
  CheckCircle,
  HelpCircle,
  Type,
  ChevronRight,
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
  const isClean = report.suspicious_region_count === 0;
  const activeRegion = report.suspicious_regions.find((r) => r.region_id === selectedRegionId) || null;

  return (
    <div className="bg-white rounded-xl border border-surface-200 shadow-card flex flex-col h-full overflow-hidden">
      {/* Inspector Header */}
      <div className="p-4 border-b border-surface-200 bg-surface-50 flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-sm text-surface-900">
            Forensic Inspector
          </h3>
          <p className="text-[11px] text-slate-500">
            Localization analysis &amp; OCR text evidence
          </p>
        </div>

        <button
          onClick={onOpenReportModal}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-surface-900 text-white hover:bg-surface-800 transition-colors shadow-subtle"
        >
          <span>Full Report</span>
          <ChevronRight className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Status Banner */}
        <div
          className={`p-3.5 rounded-lg border text-xs leading-relaxed ${
            isClean
              ? "bg-emerald-50/70 border-emerald-200 text-emerald-950"
              : "bg-red-50/70 border-red-200 text-red-950"
          }`}
        >
          <div className="flex items-center gap-2 font-semibold mb-1">
            {isClean ? (
              <>
                <CheckCircle className="w-4 h-4 text-emerald-600 shrink-0" />
                <span>No Significant Tampering Evidence</span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-4 h-4 text-red-600 shrink-0" />
                <span>Suspicious Visual Manipulation Detected</span>
              </>
            )}
          </div>
          <p className="text-[11px] opacity-90">{report.assessment_summary}</p>
        </div>

        {/* Aggregate Forensic Scores */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-surface-50 p-3 rounded-lg border border-surface-200">
            <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1">
              <span>Max Tampering Score</span>
              <span title="Peak model probability for pixel manipulation in suspicious regions.">
                <HelpCircle className="w-3 h-3 text-slate-400" />
              </span>
            </div>
            <div className="text-lg font-bold font-mono text-surface-900">
              {report.highest_tampering_score.toFixed(3)}
            </div>
            <div className="w-full bg-surface-200 h-1.5 rounded-full overflow-hidden mt-1.5">
              <div
                className="bg-red-500 h-full rounded-full"
                style={{ width: `${Math.min(report.highest_tampering_score * 100, 100)}%` }}
              />
            </div>
          </div>

          <div className="bg-surface-50 p-3 rounded-lg border border-surface-200">
            <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1">
              <span>Suspicious Area</span>
              <span title="Total fraction of document pixels flagged as manipulated.">
                <HelpCircle className="w-3 h-3 text-slate-400" />
              </span>
            </div>
            <div className="text-lg font-bold font-mono text-surface-900">
              {report.total_suspicious_area_percent.toFixed(2)}%
            </div>
            <div className="w-full bg-surface-200 h-1.5 rounded-full overflow-hidden mt-1.5">
              <div
                className="bg-amber-500 h-full rounded-full"
                style={{ width: `${Math.min(report.total_suspicious_area_percent * 5, 100)}%` }}
              />
            </div>
          </div>
        </div>

        {/* Region List & Details */}
        {!isClean && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-surface-900 uppercase tracking-wide">
                Detected Regions ({report.suspicious_regions.length})
              </span>
              {selectedRegionId && (
                <button
                  onClick={() => onSelectRegion(null)}
                  className="text-[11px] text-blue-600 hover:underline"
                >
                  Clear Selection
                </button>
              )}
            </div>

            {/* Region Cards */}
            <div className="space-y-2.5">
              {report.suspicious_regions.map((region) => {
                const isSelected = selectedRegionId === region.region_id;

                return (
                  <div
                    key={region.region_id}
                    onClick={() => onSelectRegion(isSelected ? null : region.region_id)}
                    className={`p-3 rounded-lg border transition-all cursor-pointer ${
                      isSelected
                        ? "bg-red-50/50 border-red-400 shadow-sm ring-1 ring-red-300"
                        : "bg-surface-50/60 border-surface-200 hover:border-surface-300 hover:bg-surface-100/50"
                    }`}
                  >
                    {/* Region Header */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span
                          className={`font-mono text-xs font-bold px-1.5 py-0.5 rounded ${
                            isSelected
                              ? "bg-red-600 text-white"
                              : "bg-surface-200 text-surface-800"
                          }`}
                        >
                          Region #{region.region_id}
                        </span>
                        <span className="text-[11px] text-slate-500 font-medium capitalize">
                          {region.region_type.replace(/_/g, " ")}
                        </span>
                      </div>

                      <span className="font-mono text-xs font-bold text-red-700">
                        Score: {region.max_tampering_score.toFixed(3)}
                      </span>
                    </div>

                    {/* OCR Evidence Box */}
                    <div className="bg-white p-2.5 rounded border border-surface-200 mb-2">
                      <div className="flex items-center justify-between text-[10px] text-slate-400 mb-1">
                        <span className="flex items-center gap-1 font-semibold uppercase tracking-wider text-slate-600">
                          <Type className="w-3 h-3 text-slate-500" />
                          OCR Evidence
                        </span>
                        {region.has_associated_text && (
                          <span className="font-mono text-slate-600 font-medium">
                            Conf: {(region.ocr_confidence * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                      {region.has_associated_text && region.ocr_text ? (
                        <div className="font-mono text-xs font-semibold text-slate-900 bg-surface-50 px-2 py-1 rounded border border-surface-200 break-words">
                          &ldquo;{region.ocr_text}&rdquo;
                        </div>
                      ) : (
                        <div className="text-[11px] italic text-slate-400">
                          No text associated with this region
                        </div>
                      )}
                    </div>

                    {/* Region Crop & Coordinates (Expanded if selected) */}
                    {isSelected && (
                      <div className="pt-2 border-t border-red-200/60 mt-2 space-y-2">
                        <div className="flex items-center gap-3">
                          {region.crop_url && (
                            <div className="w-24 h-16 bg-surface-100 rounded border border-surface-200 overflow-hidden shrink-0 flex items-center justify-center">
                              <img
                                src={getAssetUrl(region.crop_url)}
                                alt={`Region #${region.region_id} crop`}
                                className="max-h-full max-w-full object-contain"
                              />
                            </div>
                          )}
                          <div className="text-[11px] text-slate-600 space-y-1 font-mono">
                            <div>
                              <span className="text-slate-400">Mean Score:</span>{" "}
                              {region.mean_tampering_score.toFixed(3)}
                            </div>
                            <div>
                              <span className="text-slate-400">Area:</span>{" "}
                              {region.percentage_of_document_area.toFixed(3)}% ({region.pixel_area} px)
                            </div>
                            <div>
                              <span className="text-slate-400">BBox:</span> [
                              {region.bbox.join(", ")}]
                            </div>
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

        {/* Inference Latency Breakdown */}
        <div className="bg-surface-50 p-3 rounded-lg border border-surface-200 text-xs text-slate-600">
          <div className="font-semibold text-surface-900 mb-1.5 text-[11px] uppercase tracking-wide">
            Latency Breakdown
          </div>
          <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
            <div>
              <span className="text-slate-400">Dual-Stream:</span>{" "}
              {(report.performance_latency.model_inference_ms / 1000).toFixed(3)}s
            </div>
            <div>
              <span className="text-slate-400">OCR Engine:</span>{" "}
              {(report.performance_latency.ocr_processing_ms / 1000).toFixed(3)}s
            </div>
            <div>
              <span className="text-slate-400">Localization:</span>{" "}
              {(report.performance_latency.post_processing_ms / 1000).toFixed(3)}s
            </div>
            <div>
              <span className="text-slate-400">Total Pipeline:</span>{" "}
              <strong className="text-surface-900">
                {(report.performance_latency.total_pipeline_latency_ms / 1000).toFixed(3)}s
              </strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
