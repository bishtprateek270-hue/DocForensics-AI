import React from "react";
import { HeroSection } from "../components/landing/HeroSection";
import { WorkspaceContainer } from "../components/workspace/WorkspaceContainer";
import { HowItWorks } from "../components/landing/HowItWorks";
import { Capabilities } from "../components/landing/Capabilities";
import { TechArchitecture } from "../components/landing/TechArchitecture";
import { LimitationsSection } from "../components/landing/LimitationsSection";

export default function HomePage() {
  return (
    <div className="flex flex-col w-full">
      {/* 1. Hero Section with Scanning Line Simulation */}
      <HeroSection />

      {/* 2. Interactive Document Analysis Workspace */}
      <WorkspaceContainer />

      {/* 3. Verification Pipeline / How It Works */}
      <HowItWorks />

      {/* 4. Supported Tampering Modalities & Scope */}
      <Capabilities />

      {/* 5. Dual-Stream Feature Fusion Architecture */}
      <TechArchitecture />

      {/* 6. Forensic Operating Boundaries & Assumptions */}
      <LimitationsSection />
    </div>
  );
}
