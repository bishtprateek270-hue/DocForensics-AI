"use client";

import React, { useState, useRef, ChangeEvent, DragEvent } from "react";
import { Upload, FileText, Image as ImageIcon, X, AlertCircle, ArrowRight, Check } from "lucide-react";

interface UploadZoneProps {
  onFileSelect: (file: File, pageNumber: number) => void;
  onAnalyze: () => void;
  selectedFile: File | null;
  previewUrl: string | null;
  onClear: () => void;
  pageNumber: number;
  setPageNumber: (p: number) => void;
  totalPages: number;
  isAnalyzing: boolean;
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  onFileSelect,
  onAnalyze,
  selectedFile,
  previewUrl,
  onClear,
  pageNumber,
  setPageNumber,
  totalPages,
  isAnalyzing,
}) => {
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndSelectFile = (file: File) => {
    setErrorMsg(null);
    const validTypes = ["image/jpeg", "image/png", "image/jpg", "application/pdf"];
    const ext = file.name.split(".").pop()?.toLowerCase();

    if (!validTypes.includes(file.type) && !["jpg", "jpeg", "png", "pdf"].includes(ext || "")) {
      setErrorMsg("Unsupported file format. Please provide a PNG, JPG, or PDF document.");
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      setErrorMsg("File size exceeds 25 MB limit.");
      return;
    }

    onFileSelect(file, 1);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSelectFile(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSelectFile(e.target.files[0]);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="w-full">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleInputChange}
        accept=".png,.jpg,.jpeg,.pdf"
        className="hidden"
      />

      {!selectedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-200 ${
            isDragging
              ? "border-blue-500 bg-blue-50/50"
              : "border-surface-300 bg-white hover:border-surface-400 hover:bg-surface-50/50"
          }`}
        >
          <div className="w-12 h-12 rounded-full bg-surface-100 flex items-center justify-center mx-auto mb-4 text-slate-700">
            <Upload className="w-6 h-6" />
          </div>
          <h3 className="text-base font-semibold text-surface-900 mb-1">
            Drag &amp; drop document to analyze
          </h3>
          <p className="text-xs text-slate-500 max-w-sm mx-auto mb-4">
            Supports high-resolution PNG, JPG, and single/multi-page PDF documents up to 25 MB.
          </p>
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium bg-surface-100 text-surface-800 border border-surface-200">
            <FileText className="w-3.5 h-3.5" />
            <span>Select File from Disk</span>
          </div>

          {errorMsg && (
            <div className="mt-4 inline-flex items-center gap-2 px-3 py-1.5 rounded bg-red-50 text-red-700 border border-red-200 text-xs text-left">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}
        </div>
      ) : (
        /* Selected File Card */
        <div className="bg-white rounded-xl border border-surface-200 p-5 shadow-card">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-surface-200 pb-4 mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-surface-100 flex items-center justify-center text-surface-800 shrink-0">
                {selectedFile.name.endsWith(".pdf") ? (
                  <FileText className="w-5 h-5 text-red-600" />
                ) : (
                  <ImageIcon className="w-5 h-5 text-blue-600" />
                )}
              </div>
              <div className="overflow-hidden">
                <h4 className="font-semibold text-sm text-surface-900 truncate max-w-xs sm:max-w-md">
                  {selectedFile.name}
                </h4>
                <div className="flex items-center gap-2 text-xs text-slate-500 font-mono">
                  <span>{formatFileSize(selectedFile.size)}</span>
                  <span>&bull;</span>
                  <span>{selectedFile.type || "Document"}</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={onClear}
                disabled={isAnalyzing}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 hover:bg-surface-100 rounded-md transition-colors disabled:opacity-50"
              >
                <X className="w-3.5 h-3.5" />
                <span>Replace File</span>
              </button>

              <button
                onClick={onAnalyze}
                disabled={isAnalyzing}
                className="inline-flex items-center gap-2 px-5 py-2 rounded-lg text-xs font-semibold bg-surface-900 text-white hover:bg-surface-800 active:scale-[0.99] transition-all shadow-subtle disabled:opacity-50"
              >
                <span>Run Forensic Analysis</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* PDF Page Selector (if PDF) */}
          {selectedFile.name.toLowerCase().endsWith(".pdf") && (
            <div className="bg-surface-50 rounded-lg p-3 border border-surface-200 flex items-center justify-between text-xs mb-4">
              <div className="text-slate-600">
                <span className="font-semibold text-surface-900">PDF Document:</span> Page to analyze
              </div>
              <div className="flex items-center gap-2">
                <label className="text-slate-500">Page:</label>
                <input
                  type="number"
                  min={1}
                  value={pageNumber}
                  onChange={(e) => setPageNumber(Math.max(1, parseInt(e.target.value) || 1))}
                  className="w-16 px-2 py-1 bg-white border border-surface-300 rounded text-center font-mono text-xs focus:outline-none focus:border-blue-500"
                  disabled={isAnalyzing}
                />
              </div>
            </div>
          )}

          {/* Quick Preview Thumbnail */}
          {previewUrl && (
            <div className="relative aspect-[16/9] max-h-56 bg-surface-100 rounded-lg border border-surface-200 overflow-hidden flex items-center justify-center">
              <img
                src={previewUrl}
                alt="Document preview"
                className="max-h-full max-w-full object-contain"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};
