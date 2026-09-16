"use client";

import React, { useState, useRef, ChangeEvent, DragEvent } from "react";
import { Upload, FileText, Image as ImageIcon, X, AlertCircle, ArrowRight } from "lucide-react";

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
      setErrorMsg("Unsupported file format. Please upload a PNG, JPG, or PDF document.");
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      setErrorMsg("File size exceeds the 25 MB limit.");
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
          className={`border-2 border-dashed rounded-2xl p-10 sm:p-16 text-center cursor-pointer transition-all duration-200 ${
            isDragging
              ? "border-blue-500 bg-blue-50/60 shadow-lg scale-[1.01]"
              : "border-slate-300 bg-white hover:border-slate-400 hover:bg-slate-50/70 shadow-sm"
          }`}
        >
          <div className="w-16 h-16 rounded-2xl bg-slate-100 flex items-center justify-center mx-auto mb-5 text-slate-700 shadow-inner">
            <Upload className="w-8 h-8" />
          </div>
          <h3 className="text-lg sm:text-xl font-bold text-slate-900 mb-2">
            Drag &amp; drop document to analyze
          </h3>
          <p className="text-sm text-slate-500 max-w-md mx-auto mb-6">
            Supports high-resolution PNG, JPG, and PDF documents (invoices, certificates, contracts) up to 25 MB.
          </p>
          <div className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold bg-slate-100 text-slate-800 border border-slate-200 shadow-sm hover:bg-slate-200 transition-colors">
            <FileText className="w-4 h-4" />
            <span>Select Document from Computer</span>
          </div>

          {errorMsg && (
            <div className="mt-5 inline-flex items-center gap-2.5 px-4 py-2 rounded-lg bg-red-50 text-red-700 border border-red-200 text-sm text-left">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}
        </div>
      ) : (
        /* Selected File Card */
        <div className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 shadow-md">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-5 border-b border-slate-200 pb-6 mb-6">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-xl bg-slate-100 flex items-center justify-center text-slate-800 shrink-0 shadow-inner">
                {selectedFile.name.endsWith(".pdf") ? (
                  <FileText className="w-7 h-7 text-red-600" />
                ) : (
                  <ImageIcon className="w-7 h-7 text-blue-600" />
                )}
              </div>
              <div className="overflow-hidden">
                <h4 className="font-bold text-base sm:text-lg text-slate-900 truncate max-w-xs sm:max-w-md">
                  {selectedFile.name}
                </h4>
                <div className="flex items-center gap-2.5 text-xs sm:text-sm text-slate-500 font-mono mt-0.5">
                  <span>{formatFileSize(selectedFile.size)}</span>
                  <span>&bull;</span>
                  <span>{selectedFile.type || "Document"}</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={onClear}
                disabled={isAnalyzing}
                className="inline-flex items-center gap-2 px-4 py-2.5 text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-xl transition-colors disabled:opacity-50"
              >
                <X className="w-4 h-4" />
                <span>Replace</span>
              </button>

              <button
                onClick={onAnalyze}
                disabled={isAnalyzing}
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold bg-slate-900 text-white hover:bg-slate-800 active:scale-[0.99] transition-all shadow-md disabled:opacity-50"
              >
                <span>Run Analysis</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* PDF Page Selector (if PDF) */}
          {selectedFile.name.toLowerCase().endsWith(".pdf") && (
            <div className="bg-slate-50 rounded-xl p-4 border border-slate-200 flex items-center justify-between text-sm mb-6">
              <div className="text-slate-700">
                <span className="font-bold text-slate-900">PDF Document:</span> Select page to inspect
              </div>
              <div className="flex items-center gap-2.5">
                <label className="text-slate-500 text-xs">Page:</label>
                <input
                  type="number"
                  min={1}
                  value={pageNumber}
                  onChange={(e) => setPageNumber(Math.max(1, parseInt(e.target.value) || 1))}
                  className="w-16 px-2.5 py-1.5 bg-white border border-slate-300 rounded-lg text-center font-mono text-sm focus:outline-none focus:border-blue-500 shadow-sm"
                  disabled={isAnalyzing}
                />
              </div>
            </div>
          )}

          {/* Quick Preview Thumbnail */}
          {previewUrl && (
            <div className="relative aspect-[16/9] max-h-72 bg-slate-100 rounded-xl border border-slate-200 overflow-hidden flex items-center justify-center shadow-inner">
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
