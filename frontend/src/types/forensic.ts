/**
 * DocForensics AI — Frontend TypeScript Types
 * Exact 1:1 match with backend/schemas.py Pydantic schemas (Dual-Specialist Fusion)
 */

export type AnalysisStatus =
  | "no_significant_tampering_evidence_detected"
  | "suspicious_visual_manipulation_detected"
  | "manual_review_recommended";

export interface SuspiciousRegion {
  region_id: number;
  bbox: [number, number, number, number];
  center: [number, number];
  pixel_area: number;
  percentage_of_document_area: number;
  mean_tampering_score: number;
  max_tampering_score: number;
  ocr_text: string;
  ocr_confidence: number;
  has_associated_text: boolean;
  region_type: string;
  evidence_sources?: string[];
  evidence_score?: number;
  explanation?: string;
  crop_url: string;
}

export interface EvidenceSummary {
  physical_visual_evidence: boolean;
  digital_text_visual_evidence: boolean;
  content_inconsistency: boolean;
  reference_mismatch: boolean;
  physical_region_count: number;
  text_region_count: number;
  fused_region_count: number;
  status_message: string;
  disclaimer: string;
}

export interface PerformanceLatency {
  model_inference_ms: number;
  ocr_processing_ms: number;
  post_processing_ms: number;
  total_pipeline_latency_ms: number;
  peak_vram_mb: number;
}

export interface ConsistencyCheck {
  check_id: string;
  check_name: string;
  displayed_value: string;
  calculated_value: string;
  status: "match" | "mismatch" | "impossible_value" | "unavailable" | string;
  explanation: string;
}

export interface ContentAnalysis {
  document_type: string;
  document_type_confidence: number;
  status: "content_consistent" | "content_inconsistency_detected" | "insufficient_information" | string;
  summary: string;
  checks: ConsistencyCheck[];
  extracted_fields: Record<string, any>;
}

export interface RecordVerification {
  status: "verified_match" | "authoritative_mismatch" | "record_not_found" | "not_available" | string;
  source: string;
  message: string;
  mismatches?: Array<Record<string, any>> | null;
}

export interface VisualAnalysis {
  analysis_status: string;
  suspicious_region_count: number;
  highest_tampering_score: number;
  total_suspicious_area_percent: number;
}

export interface ForensicReport {
  session_id: string;
  document_id: string;
  filename: string;
  file_type: string;
  page_number?: number | null;
  total_pages?: number | null;
  timestamp_utc: string;
  analysis_status: AnalysisStatus;
  assessment_summary: string;
  model_architecture: string;
  inference_threshold: number;
  original_resolution: [number, number];
  suspicious_region_count: number;
  total_suspicious_area_percent: number;
  highest_tampering_score: number;
  visual_analysis?: VisualAnalysis | null;
  content_analysis?: ContentAnalysis | null;
  record_verification?: RecordVerification | null;
  evidence_summary?: EvidenceSummary | null;
  performance_latency: PerformanceLatency;
  suspicious_regions: SuspiciousRegion[];
  image_url: string;
  heatmap_url: string;
  overlay_url: string;
  pdf_report_url?: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  cuda_available: boolean;
  gpu_name: string;
  vram_total_gb: number;
  vram_allocated_mb: number;
  model_checkpoint_loaded: string;
  ocr_engine: string;
}

export interface ErrorResponse {
  error: string;
  detail?: string;
}
