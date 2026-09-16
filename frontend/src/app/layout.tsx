import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "../components/layout/Navbar";
import { Footer } from "../components/layout/Footer";

export const metadata: Metadata = {
  title: "DocForensics AI — Document Tampering Detection & Localization",
  description:
    "AI-powered digital forensics system for document tampering detection, localization, and OCR evidence extraction.",
  keywords: [
    "document forensics",
    "tampering detection",
    "digital forgery",
    "splicing localization",
    "OCR verification",
  ],
  authors: [{ name: "DocForensics AI Team" }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className="min-h-screen flex flex-col bg-slate-50 text-slate-900 selection:bg-blue-100 selection:text-blue-900">
        <Navbar />
        <main className="flex-1">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
