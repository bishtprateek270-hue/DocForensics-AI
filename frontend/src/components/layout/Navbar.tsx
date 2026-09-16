"use client";

import React from "react";
import Link from "next/link";
import { ShieldCheck, ArrowRight } from "lucide-react";

export const Navbar: React.FC = () => {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-200 bg-white/95 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-18 flex items-center justify-between">
        {/* Brand Logo & Name */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-xl bg-slate-900 flex items-center justify-center text-white shadow-sm group-hover:bg-slate-800 transition-colors shrink-0">
            <ShieldCheck className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="font-bold text-lg text-slate-900 tracking-tight block">
              DocForensics AI
            </span>
            <span className="text-xs text-slate-500 font-normal hidden sm:block">
              Document Tampering Detection
            </span>
          </div>
        </Link>

        {/* Simplified User Navigation */}
        <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-600">
          <a
            href="#workspace"
            className="hover:text-slate-900 transition-colors py-1"
          >
            Analyze
          </a>
          <a
            href="#how-it-works"
            className="hover:text-slate-900 transition-colors py-1"
          >
            How It Works
          </a>
          <a
            href="#capabilities"
            className="hover:text-slate-900 transition-colors py-1"
          >
            Capabilities
          </a>
          <a
            href="#about"
            className="hover:text-slate-900 transition-colors py-1"
          >
            About
          </a>
        </nav>

        {/* Primary Action Button */}
        <div className="flex items-center gap-4">
          <a
            href="#workspace"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold bg-slate-900 text-white hover:bg-slate-800 active:scale-[0.99] transition-all shadow-sm"
          >
            <span>Analyze Document</span>
            <ArrowRight className="w-4 h-4" />
          </a>
        </div>
      </div>
    </header>
  );
};
