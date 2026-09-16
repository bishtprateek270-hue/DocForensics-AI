"use client";

import React, { useState } from "react";
import { UploadZone } from "./UploadZone";
import { AnalysisProgress } from "./AnalysisProgress";
import { DocumentViewer } from "./DocumentViewer";
import { RegionInspector } from "./RegionInspector";
import { ForensicReportModal } from "./ForensicReportModal";
import { ForensicReport } from "../../types/forensic";
import { analyzeDocument, ApiError } from "../../lib/api";
import { AlertCircle, RotateCcw, ShieldCheck, ArrowLeft } from "lucide-react";

export const WorkspaceContainer: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [pageNumber, setPageNumber] = useState<number>(1);
  const [totalPages, setTotalPages] = useState<number>(1);

  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [report, setReport] = useState<ForensicReport | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [selectedRegionId, setSelectedRegionId] = useState<number | null>(null);
  const [isReportModalOpen, setIsReportModalOpen] = useState<boolean>(false);

  const handleFileSelect = (file: File, initialPage: number = 1) => {
    setSelectedFile(file);
    setPageNumber(initialPage);
    setReport(null);
    setErrorMessage(null);
    setSelectedRegionId(null);

    // Create object URL for preview
    if (file.type.startsWith("image/")) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      // PDF or non-image
      setPreviewUrl(null);
    }
  };

  const handleClear = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setReport(null);
    setErrorMessage(null);
    setSelectedRegionId(null);
    setPageNumber(1);
  };

  const handleRunAnalysis = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setErrorMessage(null);
    setReport(null);
    setSelectedRegionId(null);

    try {
      const analysisResult = await analyzeDocument(selectedFile, pageNumber);
      setReport(analysisResult);
    } catch (err: any) {
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage(err.message || "An unexpected error occurred during document analysis.");
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <section id="workspace" className="py-12 md:py-16 bg-surface-50 min-h-[700px] border-b border-surface-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Workspace Title & Controls Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">
              Forensic Investigation Suite
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold text-surface-900 tracking-tight">
              Document Analysis Workspace
            </h2>
          </div>

          {report && (
            <button
              onClick={handleClear}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-700 bg-white hover:bg-surface-100 rounded-lg border border-surface-200 transition-colors shadow-subtle self-start sm:self-auto"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Analyze Another Document</span>
            </button>
          )}
        </div>

        {/* Error Banner */}
        {errorMessage && (
          <div className="mb-6 p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 text-xs flex items-start justify-between gap-3 shadow-subtle">
            <div className="flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold block mb-0.5">Analysis Request Failed</span>
                <p className="text-red-700 leading-relaxed">{errorMessage}</p>
              </div>
            </div>
            <button
              onClick={handleRunAnalysis}
              className="shrink-0 inline-flex items-center gap-1 px-3 py-1.5 rounded bg-red-100 hover:bg-red-200 text-red-900 font-medium transition-colors"
            >
              <RotateCcw className="w-3 h-3" />
              <span>Retry</span>
            </button>
          </div>
        )}

        {/* Analysis Progress View */}
        {isAnalyzing && (
          <AnalysisProgress
            previewUrl={previewUrl}
            filename={selectedFile?.name || "document"}
          />
        )}

        {/* File Upload Stage */}
        {!isAnalyzing && !report && (
          <div className="max-w-3xl mx-auto">
            <UploadZone
              onFileSelect={handleFileSelect}
              onAnalyze={handleRunAnalysis}
              selectedFile={selectedFile}
              previewUrl={previewUrl}
              onClear={handleClear}
              pageNumber={pageNumber}
              setPageNumber={setPageNumber}
              totalPages={totalPages}
              isAnalyzing={isAnalyzing}
            />
          </div>
        )}

        {/* Forensic Results Workspace */}
        {!isAnalyzing && report && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
            {/* Left Column: Interactive Document & Heatmap Viewer */}
            <div className="lg:col-span-8">
              <DocumentViewer
                report={report}
                selectedRegionId={selectedRegionId}
                onSelectRegion={setSelectedRegionId}
              />
            </div>

            {/* Right Column: Forensic Inspector & OCR Evidence */}
            <div className="lg:col-span-4">
              <RegionInspector
                report={report}
                selectedRegionId={selectedRegionId}
                onSelectRegion={setSelectedRegionId}
                onOpenReportModal={() => setIsReportModalOpen(true)}
              />
            </div>
          </div>
        )}
      </div>

      {/* Forensic Full Report Modal */}
      {report && (
        <ForensicReportModal
          report={report}
          isOpen={isReportModalOpen}
          onClose={() => setIsReportModalOpen(false)}
        />
      )}
    </section>
  );
};
