"""
DocForensics AI — Content Consistency & Inspect-Element Analysis Layer
Provides deterministic rule-based checks, structured OCR extraction,
and academic / invoice arithmetic validation.
"""

from .classifier import detect_document_type
from .extractor import extract_structured_fields
from .academic import verify_academic_consistency
from .invoice import verify_invoice_consistency
from .verification_provider import VerificationProvider, MockVerificationProvider
from .pipeline import ContentConsistencyPipeline

__all__ = [
    "detect_document_type",
    "extract_structured_fields",
    "verify_academic_consistency",
    "verify_invoice_consistency",
    "VerificationProvider",
    "MockVerificationProvider",
    "ContentConsistencyPipeline",
]
