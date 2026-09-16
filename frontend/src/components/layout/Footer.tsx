import React from "react";
import Link from "next/link";

export const Footer: React.FC = () => {
  return (
    <footer id="about" className="w-full border-t border-slate-200 bg-slate-50 text-slate-600 text-sm py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          <div className="md:col-span-2 space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-slate-900 text-white flex items-center justify-center font-bold text-xs">
                DF
              </div>
              <span className="font-bold text-base text-slate-900">
                DocForensics AI
              </span>
            </div>
            <p className="text-slate-500 text-xs sm:text-sm leading-relaxed max-w-md">
              AI-powered document tampering detection and localization platform. Inspects visual textures, noise residuals, and text structure to assist investigators and organizations in spotting digital alterations.
            </p>
          </div>

          <div>
            <h4 className="font-bold text-slate-900 text-xs uppercase tracking-wider mb-3">
              Navigation
            </h4>
            <ul className="space-y-2 text-xs text-slate-500">
              <li>
                <a href="#workspace" className="hover:text-slate-900 transition-colors">
                  Document Analysis
                </a>
              </li>
              <li>
                <a href="#how-it-works" className="hover:text-slate-900 transition-colors">
                  How It Works
                </a>
              </li>
              <li>
                <a href="#capabilities" className="hover:text-slate-900 transition-colors">
                  Inspection Capabilities
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="font-bold text-slate-900 text-xs uppercase tracking-wider mb-3">
              Disclaimer
            </h4>
            <p className="text-slate-500 text-xs leading-relaxed">
              Results indicate potential visual manipulation and should not be treated as legal proof of document authenticity. Always conduct manual inspection for legal or official proceedings.
            </p>
          </div>
        </div>

        <div className="pt-6 border-t border-slate-200 flex flex-col sm:flex-row items-center justify-between text-slate-400 text-xs gap-3">
          <span>&copy; {new Date().getFullYear()} DocForensics AI. All rights reserved.</span>
          <div className="flex items-center gap-4">
            <span>Secure Verification</span>
            <span>&bull;</span>
            <span>Privacy-First Ingestion</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
