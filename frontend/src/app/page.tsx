import React from "react";
import { HeroSection } from "../components/landing/HeroSection";
import { WorkspaceContainer } from "../components/workspace/WorkspaceContainer";
import { HowItWorks } from "../components/landing/HowItWorks";
import { Capabilities } from "../components/landing/Capabilities";

export default function HomePage() {
  return (
    <div className="flex flex-col w-full">
      {/* 1. Hero Section */}
      <HeroSection />

      {/* 2. Interactive Document Analysis Workspace */}
      <WorkspaceContainer />

      {/* 3. Simple How It Works (4 Steps) */}
      <HowItWorks />

      {/* 4. Supported Manipulation Types & Inspection Scope */}
      <Capabilities />
    </div>
  );
}
