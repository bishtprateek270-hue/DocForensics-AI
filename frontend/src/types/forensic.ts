/**
 * DocForensics AI — Frontend TypeScript Types
 * Exact 1:1 match with backend/schemas.py Pydantic schemas
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
  crop_url: string;
}

export interface PerformanceLatency {
  model_inference_ms: number;
  ocr_processing_ms: number;
  post_processing_ms: number;
  total_pipeline_latency_ms: number;
  peak_vram_mb: number;
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
  performance_latency: PerformanceLatency;
  suspicious_regions: SuspiciousRegion[];
  image_url: string;
  heatmap_url: string;
  overlay_url: string;
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
