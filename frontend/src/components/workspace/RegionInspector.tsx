"use client";

import React from "react";
import { ForensicReport, SuspiciousRegion } from "../../types/forensic";
import { getAssetUrl } from "../../lib/api";
import {
  AlertTriangle,
  CheckCircle,
  Type,
  ChevronRight,
  Sparkles,
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

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm flex flex-col h-full overflow-hidden">
      {/* Inspector Header */}
      <div className="p-5 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
        <div>
          <h3 className="font-bold text-base text-slate-900">
            Inspection Results
          </h3>
          <p className="text-xs text-slate-500">
            Detected regions &amp; extracted text
          </p>
        </div>

        <button
          onClick={onOpenReportModal}
          className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold bg-slate-900 text-white hover:bg-slate-800 transition-colors shadow-sm"
        >
          <span>Full Report</span>
          <ChevronRight className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-5 space-y-5">
        {/* Status Banner */}
        <div
          className={`p-4 rounded-xl border text-sm leading-relaxed ${
            isClean
              ? "bg-emerald-50 border-emerald-200 text-emerald-950"
              : "bg-red-50 border-red-200 text-red-950"
          }`}
        >
          <div className="flex items-center gap-2.5 font-bold mb-1.5">
            {isClean ? (
              <>
                <CheckCircle className="w-5 h-5 text-emerald-600 shrink-0" />
                <span>No Tampering Detected</span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-5 h-5 text-red-600 shrink-0" />
                <span>Suspicious Visual Manipulation Detected</span>
              </>
            )}
          </div>
          <p className="text-xs opacity-90 leading-relaxed">{report.assessment_summary}</p>
        </div>

        {/* Aggregate Forensic Scores */}
        <div className="grid grid-cols-2 gap-3.5">
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <div className="text-xs text-slate-500 font-medium mb-1">
              Tampering Probability
            </div>
            <div className="text-xl font-bold font-mono text-slate-900">
              {(report.highest_tampering_score * 100).toFixed(1)}%
            </div>
            <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden mt-2">
              <div
                className="bg-red-500 h-full rounded-full"
                style={{ width: `${Math.min(report.highest_tampering_score * 100, 100)}%` }}
              />
            </div>
          </div>

          <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
            <div className="text-xs text-slate-500 font-medium mb-1">
              Suspicious Area
            </div>
            <div className="text-xl font-bold font-mono text-slate-900">
              {report.total_suspicious_area_percent.toFixed(2)}%
            </div>
            <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden mt-2">
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
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                Flagged Areas ({report.suspicious_regions.length})
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

            {/* Region Cards */}
            <div className="space-y-3">
              {report.suspicious_regions.map((region) => {
                const isSelected = selectedRegionId === region.region_id;

                return (
                  <div
                    key={region.region_id}
                    onClick={() => onSelectRegion(isSelected ? null : region.region_id)}
                    className={`p-3.5 rounded-xl border transition-all cursor-pointer ${
                      isSelected
                        ? "bg-red-50/70 border-red-400 shadow-md ring-2 ring-red-200"
                        : "bg-slate-50/70 border-slate-200 hover:border-slate-300 hover:bg-slate-100/70 shadow-sm"
                    }`}
                  >
                    {/* Region Header */}
                    <div className="flex items-center justify-between mb-2.5">
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

                    {/* OCR Evidence Box */}
                    <div className="bg-white p-3 rounded-lg border border-slate-200 mb-2 shadow-inner">
                      <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1.5">
                        <span className="flex items-center gap-1.5 font-bold uppercase tracking-wider text-slate-700">
                          <Type className="w-3.5 h-3.5 text-blue-600" />
                          Detected Text Evidence
                        </span>
                        {region.has_associated_text && (
                          <span className="font-mono text-slate-500 font-medium">
                            Conf: {(region.ocr_confidence * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                      {region.has_associated_text && region.ocr_text ? (
                        <div className="font-mono text-sm font-bold text-slate-900 bg-slate-50 px-2.5 py-1.5 rounded border border-slate-200 break-words">
                          &ldquo;{region.ocr_text}&rdquo;
                        </div>
                      ) : (
                        <div className="text-xs italic text-slate-400 py-0.5">
                          No text detected in this visual area
                        </div>
                      )}
                    </div>

                    {/* Region Crop & Info (Expanded if selected) */}
                    {isSelected && (
                      <div className="pt-3 border-t border-red-200 mt-2 space-y-2.5">
                        <div className="flex items-center gap-3.5">
                          {region.crop_url && (
                            <div className="w-28 h-18 bg-slate-100 rounded-lg border border-slate-200 overflow-hidden shrink-0 flex items-center justify-center shadow-inner">
                              <img
                                src={getAssetUrl(region.crop_url)}
                                alt={`Area #${region.region_id} crop`}
                                className="max-h-full max-w-full object-contain"
                              />
                            </div>
                          )}
                          <div className="text-xs text-slate-600 space-y-1">
                            <div>
                              <span className="text-slate-400">Area Fraction:</span>{" "}
                              <strong>{region.percentage_of_document_area.toFixed(2)}%</strong>
                            </div>
                            <div>
                              <span className="text-slate-400">Status:</span>{" "}
                              <strong className="text-red-700">Flagged For Review</strong>
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

        {/* Processing Performance Pill */}
        <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-xs text-slate-500 flex items-center justify-between">
          <span>Inspection Speed:</span>
          <strong className="font-mono text-slate-800">
            {(report.performance_latency.total_pipeline_latency_ms / 1000).toFixed(2)}s
          </strong>
        </div>
      </div>
    </div>
  );
};
