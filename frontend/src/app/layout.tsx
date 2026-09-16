import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "../components/layout/Navbar";
import { Footer } from "../components/layout/Footer";

export const metadata: Metadata = {
  title: "DocForensics AI — Document Tampering Detection & Localization",
  description:
    "High-precision forensic neural system combining RGB spatial semantics and SRM noise residuals to detect and localize pixel-level document tampering with OCR evidence extraction.",
  keywords: [
    "document forensics",
    "tampering detection",
    "digital forgery",
    "splicing localization",
    "SRM noise analysis",
    "OCR verification",
    "dual stream neural network",
  ],
  authors: [{ name: "DocForensics AI Research Team" }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className="min-h-screen flex flex-col bg-slate-50 text-slate-900 selection:bg-red-500 selection:text-white">
        <Navbar />
        <main className="flex-1">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
