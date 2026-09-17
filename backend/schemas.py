"""
DocForensics AI — FastAPI Pydantic Schemas
Defines clean, sanitized request and response data models without exposing local filesystem paths.
Includes separate schemas for Visual Forensics, Content Consistency, and Authoritative Verification.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PerformanceLatencySchema(BaseModel):
    model_inference_ms: float = Field(..., description="Duration of dual-stream model inference in milliseconds")
    ocr_processing_ms: float = Field(..., description="Duration of OCR text extraction in milliseconds")
    post_processing_ms: float = Field(..., description="Duration of connected-component extraction in milliseconds")
    total_pipeline_latency_ms: float = Field(..., description="Total pipeline latency in milliseconds")
    peak_vram_mb: float = Field(..., description="Peak GPU memory allocated in MB")


class SuspiciousRegionSchema(BaseModel):
    region_id: int = Field(..., description="1-indexed unique region identifier")
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2] in original document coordinates")
    center: List[float] = Field(..., description="Center coordinate [cx, cy]")
    pixel_area: int = Field(..., description="Area of the suspicious region in pixels")
    percentage_of_document_area: float = Field(..., description="Percentage of document area covered by this region")
    mean_tampering_score: float = Field(..., description="Mean model tampering probability within region mask")
    max_tampering_score: float = Field(..., description="Maximum model tampering probability within region mask")
    ocr_text: str = Field(..., description="Associated OCR text string detected inside or overlapping region")
    ocr_confidence: float = Field(..., description="Recognition confidence score of extracted text [0.0 - 1.0]")
    has_associated_text: bool = Field(..., description="Whether OCR detected text in this region")
    region_type: str = Field(..., description="Conservative heuristic visual category")
    crop_url: Optional[str] = Field(None, description="API URL to download high-resolution region crop image")


class ConsistencyCheckSchema(BaseModel):
    check_id: str = Field(..., description="Unique check identifier")
    check_name: str = Field(..., description="Human-readable check title")
    displayed_value: str = Field(..., description="Value extracted directly from the document")
    calculated_value: str = Field(..., description="Value derived mathematically or from grading rules")
    status: str = Field(..., description="match | mismatch | impossible_value | unavailable")
    explanation: str = Field(..., description="Detailed explanation of the check result")


class ContentAnalysisSchema(BaseModel):
    document_type: str = Field(..., description="academic_result | invoice | certificate | generic_document | unknown")
    document_type_confidence: float = Field(default=0.8, description="Classification confidence [0.0 - 1.0]")
    status: str = Field(
        ...,
        description="content_consistent | content_inconsistency_detected | insufficient_information"
    )
    summary: str = Field(..., description="High-level content consistency summary")
    checks: List[ConsistencyCheckSchema] = Field(default_factory=list)
    extracted_fields: Dict[str, Any] = Field(default_factory=dict)


class RecordVerificationSchema(BaseModel):
    status: str = Field(
        default="not_available",
        description="verified_match | authoritative_mismatch | record_not_found | not_available"
    )
    source: str = Field(default="Local System", description="Authoritative record source name")
    message: str = Field(default="No external database configured for this document.", description="Verification status explanation")
    mismatches: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of mismatched fields if any")


class VisualAnalysisSchema(BaseModel):
    analysis_status: str = Field(..., description="Visual tampering classification")
    suspicious_region_count: int = Field(..., description="Number of suspicious visual regions")
    highest_tampering_score: float = Field(..., description="Peak visual tampering score")
    total_suspicious_area_percent: float = Field(..., description="Percentage of document area flagged")


class ForensicAnalysisReport(BaseModel):
    session_id: str = Field(..., description="Unique analysis session UUID")
    document_id: str = Field(..., description="Clean document filename or identifier")
    filename: str = Field(..., description="Uploaded document original filename")
    file_type: str = Field(..., description="File format (JPG, PNG, PDF)")
    page_number: Optional[int] = Field(1, description="Analyzed page number for multi-page documents")
    total_pages: Optional[int] = Field(1, description="Total pages in source document")
    timestamp_utc: str = Field(..., description="ISO 8601 UTC timestamp of analysis")
    
    # Combined Overall Status
    analysis_status: str = Field(
        ...,
        description="no_significant_tampering_evidence_detected | suspicious_visual_manipulation_detected | manual_review_recommended",
    )
    assessment_summary: str = Field(..., description="Conservative forensic summary text")
    model_architecture: str = Field(default="dual_stream_rgb_srm_forensic", description="Model architecture used")
    inference_threshold: float = Field(default=0.50, description="Decision threshold applied")
    original_resolution: List[int] = Field(..., description="Original image dimensions [width, height]")
    
    # Layer 1: Visual Forensics
    suspicious_region_count: int = Field(..., description="Number of suspicious regions detected")
    total_suspicious_area_percent: float = Field(..., description="Total suspicious pixel area percentage")
    highest_tampering_score: float = Field(..., description="Highest localized tampering score")
    suspicious_regions: List[SuspiciousRegionSchema] = Field(default_factory=list)
    visual_analysis: Optional[VisualAnalysisSchema] = Field(None, description="Structured visual forensic layer findings")

    # Layer 2: Content Consistency Analysis
    content_analysis: Optional[ContentAnalysisSchema] = Field(None, description="Structured content consistency layer findings")

    # Layer 3: Authoritative Record Verification (Modular interface)
    record_verification: Optional[RecordVerificationSchema] = Field(None, description="Authoritative record verification findings")

    # Performance & Streaming Assets
    performance_latency: PerformanceLatencySchema
    image_url: str = Field(..., description="API endpoint to fetch original image/page")
    heatmap_url: str = Field(..., description="API endpoint to fetch probability heatmap PNG")
    overlay_url: str = Field(..., description="API endpoint to fetch localization overlay PNG")


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="API health status")
    version: str = Field(default="2.0.0", description="DocForensics AI API version")
    cuda_available: bool = Field(..., description="Whether CUDA is active")
    gpu_name: str = Field(..., description="NVIDIA GPU model name")
    vram_total_gb: float = Field(..., description="Total GPU VRAM in GB")
    vram_allocated_mb: float = Field(..., description="Currently allocated VRAM in MB")
    model_checkpoint_loaded: str = Field(..., description="Active checkpoint name")
    ocr_engine: str = Field(default="EasyOCR (CRAFT + CRNN with PyTorch CUDA)", description="Active OCR engine")
    content_consistency_engine: str = Field(default="Active (Deterministic Rules & Layout Extractor)", description="Content engine status")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message description")
    detail: Optional[str] = Field(None, description="Detailed diagnostic context")
