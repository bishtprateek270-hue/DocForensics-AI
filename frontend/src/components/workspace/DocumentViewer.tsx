"use client";

import React, { useState, useRef } from "react";
import { ForensicReport, SuspiciousRegion } from "../../types/forensic";
import { getAssetUrl } from "../../lib/api";
import { ZoomIn, ZoomOut, RotateCcw, Eye, Layers, Flame } from "lucide-react";

interface DocumentViewerProps {
  report: ForensicReport;
  selectedRegionId: number | null;
  onSelectRegion: (id: number | null) => void;
}

type ViewMode = "overlay" | "heatmap" | "original";

export const DocumentViewer: React.FC<DocumentViewerProps> = ({
  report,
  selectedRegionId,
  onSelectRegion,
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>("overlay");
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(65);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const containerRef = useRef<HTMLDivElement>(null);

  const [origW, origH] = report.original_resolution;

  const handleZoomIn = () => setZoomLevel((prev) => Math.min(prev + 0.25, 3));
  const handleZoomOut = () => setZoomLevel((prev) => Math.max(prev - 0.25, 0.5));
  const handleResetZoom = () => setZoomLevel(1);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden shadow-sm flex flex-col h-full">
      {/* Top Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 border-b border-slate-200 bg-slate-50 text-xs">
        {/* Segmented View Mode Controls */}
        <div className="inline-flex rounded-xl p-1 bg-slate-200 border border-slate-300">
          <button
            onClick={() => setViewMode("overlay")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-semibold transition-all ${
              viewMode === "overlay"
                ? "bg-white text-slate-900 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Layers className="w-4 h-4" />
            <span>Inspection Overlay</span>
          </button>
          <button
            onClick={() => setViewMode("heatmap")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-semibold transition-all ${
              viewMode === "heatmap"
                ? "bg-white text-slate-900 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Flame className="w-4 h-4 text-red-500" />
            <span>Heatmap</span>
          </button>
          <button
            onClick={() => setViewMode("original")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-semibold transition-all ${
              viewMode === "original"
                ? "bg-white text-slate-900 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Eye className="w-4 h-4" />
            <span>Original</span>
          </button>
        </div>

        {/* Heatmap Opacity Slider (Only in heatmap mode) */}
        {viewMode === "heatmap" && (
          <div className="flex items-center gap-2.5 bg-white px-3.5 py-1.5 rounded-xl border border-slate-200 shadow-sm">
            <span className="text-slate-600 font-medium text-xs">Heatmap Intensity:</span>
            <input
              type="range"
              min="0"
              max="100"
              value={heatmapOpacity}
              onChange={(e) => setHeatmapOpacity(parseInt(e.target.value))}
              className="w-28 h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-red-600"
            />
            <span className="font-mono text-xs text-slate-800 font-semibold w-8 text-right">
              {heatmapOpacity}%
            </span>
          </div>
        )}

        {/* Zoom Controls */}
        <div className="flex items-center gap-1 bg-white rounded-xl border border-slate-200 p-1 shadow-sm">
          <button
            onClick={handleZoomOut}
            title="Zoom Out"
            className="p-1.5 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="px-2 font-mono text-xs text-slate-700 font-semibold">
            {Math.round(zoomLevel * 100)}%
          </span>
          <button
            onClick={handleZoomIn}
            title="Zoom In"
            className="p-1.5 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={handleResetZoom}
            title="Reset Zoom"
            className="p-1.5 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg border-l border-slate-200 ml-1 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Document Canvas Viewport */}
      <div
        ref={containerRef}
        onClick={() => onSelectRegion(null)}
        className="relative flex-1 min-h-[500px] max-h-[680px] bg-slate-900/5 overflow-auto p-6 flex items-center justify-center select-none cursor-default"
      >
        <div
          className="relative inline-block transition-transform duration-100 ease-out origin-center"
          style={{ transform: `scale(${zoomLevel})` }}
        >
          {/* Base Original Image */}
          <img
            src={getAssetUrl(report.image_url)}
            alt="Analyzed document"
            className="block max-h-[600px] max-w-full w-auto object-contain rounded-lg border border-slate-300 shadow-md bg-white"
          />

          {/* Heatmap Overlay (Faded on top of Original) */}
          {viewMode === "heatmap" && (
            <img
              src={getAssetUrl(report.heatmap_url)}
              alt="Tampering Heatmap"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none rounded-lg transition-opacity duration-150"
              style={{ opacity: heatmapOpacity / 100 }}
            />
          )}

          {/* Interactive Bounding Boxes Overlay */}
          {viewMode === "overlay" &&
            report.suspicious_regions.map((region) => {
              const [x1, y1, x2, y2] = region.bbox;
              const leftPct = (x1 / origW) * 100;
              const topPct = (y1 / origH) * 100;
              const widthPct = ((x2 - x1) / origW) * 100;
              const heightPct = ((y2 - y1) / origH) * 100;

              const isSelected = selectedRegionId === region.region_id;

              return (
                <div
                  key={region.region_id}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectRegion(isSelected ? null : region.region_id);
                  }}
                  className={`absolute cursor-pointer transition-all duration-150 rounded-sm ${
                    isSelected
                      ? "border-2 border-red-600 bg-red-500/25 ring-4 ring-red-400/40 z-30"
                      : "border-2 border-red-500/90 bg-red-500/15 hover:bg-red-500/30 hover:border-red-600 z-20"
                  }`}
                  style={{
                    left: `${leftPct}%`,
                    top: `${topPct}%`,
                    width: `${widthPct}%`,
                    height: `${heightPct}%`,
                  }}
                >
                  {/* Region ID Badge */}
                  <span
                    className={`absolute -top-5 left-0 px-2 py-0.5 rounded text-[10px] font-mono font-bold tracking-tight shadow-sm ${
                      isSelected
                        ? "bg-red-700 text-white ring-1 ring-red-800"
                        : "bg-red-600 text-white"
                    }`}
                  >
                    Area #{region.region_id}
                  </span>
                </div>
              );
            })}
        </div>
      </div>

      {/* Viewer Footer Bar */}
      <div className="px-5 py-3 bg-slate-50 border-t border-slate-200 text-xs text-slate-500 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span>
            Dimensions: <strong className="text-slate-800 font-mono">{origW} &times; {origH} px</strong>
          </span>
          <span>&bull;</span>
          <span>
            Flagged Areas: <strong className="text-slate-800 font-mono">{report.suspicious_region_count}</strong>
          </span>
        </div>
        <div className="text-slate-400 hidden sm:block">
          Click highlighted areas to inspect text evidence &amp; scores
        </div>
      </div>
    </div>
  );
};
