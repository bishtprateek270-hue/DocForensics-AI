"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { checkHealth } from "../../lib/api";
import { HealthResponse } from "../../types/forensic";
import { Shield, Cpu, Activity, AlertCircle, RefreshCw } from "lucide-react";

export const Navbar: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loadingHealth, setLoadingHealth] = useState<boolean>(true);
  const [backendError, setBackendError] = useState<boolean>(false);

  const fetchHealthStatus = async () => {
    setLoadingHealth(true);
    try {
      const data = await checkHealth();
      setHealth(data);
      setBackendError(false);
    } catch (err) {
      setBackendError(true);
      setHealth(null);
    } finally {
      setLoadingHealth(false);
    }
  };

  useEffect(() => {
    fetchHealthStatus();
    const interval = setInterval(fetchHealthStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-surface-200 bg-white/95 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand Logo & Name */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-lg bg-surface-900 flex items-center justify-center text-white shadow-subtle group-hover:bg-forensic-700 transition-colors">
            {/* Custom forensic document inspection icon */}
            <svg
              className="w-5 h-5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="9" y1="13" x2="15" y2="13" />
              <circle cx="15" cy="15" r="3" stroke="#ef4444" strokeWidth="2" />
              <line x1="17.2" y1="17.2" x2="19.5" y2="19.5" stroke="#ef4444" strokeWidth="2" />
            </svg>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-base text-surface-900 tracking-tight">
                DocForensics AI
              </span>
              <span className="text-[10px] font-medium tracking-wide uppercase px-1.5 py-0.5 rounded bg-surface-100 text-surface-800 border border-surface-200">
                v1.0 (Phase 9)
              </span>
            </div>
            <p className="text-xs text-slate-500 font-normal hidden sm:block">
              Document Tampering Detection & Localization
            </p>
          </div>
        </Link>

        {/* Center Navigation */}
        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600">
          <a
            href="#workspace"
            className="hover:text-surface-900 transition-colors py-1"
          >
            Analysis Workspace
          </a>
          <a
            href="#how-it-works"
            className="hover:text-surface-900 transition-colors py-1"
          >
            How It Works
          </a>
          <a
            href="#capabilities"
            className="hover:text-surface-900 transition-colors py-1"
          >
            Detection Scope
          </a>
          <a
            href="#architecture"
            className="hover:text-surface-900 transition-colors py-1"
          >
            Dual-Stream Model
          </a>
          <a
            href="#limitations"
            className="hover:text-surface-900 transition-colors py-1"
          >
            Limitations
          </a>
        </nav>

        {/* Right Status Pill */}
        <div className="flex items-center gap-3">
          {loadingHealth && !health && !backendError ? (
            <div className="flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-surface-100 text-slate-600 border border-surface-200">
              <RefreshCw className="w-3 h-3 animate-spin text-slate-400" />
              <span>Connecting API...</span>
            </div>
          ) : backendError ? (
            <button
              onClick={fetchHealthStatus}
              title="Backend unreachable at http://localhost:8000. Click to retry."
              className="flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-200 hover:bg-red-100 transition-colors"
            >
              <span className="w-2 h-2 rounded-full bg-red-500" />
              <span>API Offline (Retry)</span>
            </button>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-surface-100 text-slate-700 border border-surface-200 shadow-subtle">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="font-mono text-[11px] text-slate-800">
                {health?.gpu_name ? health.gpu_name.replace("NVIDIA ", "") : "CUDA Ready"}
              </span>
              <span className="text-slate-300">|</span>
              <span className="text-[11px] text-slate-500">Dual-Stream Active</span>
            </div>
          )}

          <a
            href="#workspace"
            className="inline-flex items-center justify-center px-3.5 py-1.5 rounded-md text-xs font-medium bg-surface-900 text-white hover:bg-surface-800 transition-colors shadow-subtle"
          >
            Start Analysis
          </a>
        </div>
      </div>
    </header>
  );
};
