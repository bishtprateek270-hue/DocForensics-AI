"""
DocForensics AI — Content Consistency Pipeline
Coordinates document classification, field extraction, rule evaluation, and provider checking.
"""

from typing import List, Dict, Any, Optional
from .classifier import detect_document_type
from .extractor import extract_structured_fields
from .academic import verify_academic_consistency
from .invoice import verify_invoice_consistency
from .verification_provider import VerificationProvider, MockVerificationProvider


class ContentConsistencyPipeline:
    """Manages content consistency analysis alongside visual forensics."""

    def __init__(self, verification_provider: Optional[VerificationProvider] = None):
        self.provider = verification_provider or MockVerificationProvider()

    def analyze_content(
        self,
        ocr_results: List[Dict[str, Any]],
        rule_scheme_id: Optional[str] = "standard_ugc_10point"
    ) -> Dict[str, Any]:
        """
        Runs document type detection, entity extraction, and consistency analysis.
        """
        # 1. Classify document type
        doc_type, type_confidence = detect_document_type(ocr_results)

        # 2. Extract structured fields
        extracted_fields = extract_structured_fields(doc_type, ocr_results)

        # 3. Apply rule-based consistency checks
        if doc_type == "academic_result":
            consistency_result = verify_academic_consistency(extracted_fields, scheme_id=rule_scheme_id)
        elif doc_type == "invoice":
            consistency_result = verify_invoice_consistency(extracted_fields)
        else:
            consistency_result = {
                "status": "insufficient_information",
                "summary": "Document type has no deterministic mathematical relationships configured.",
                "checks": []
            }

        # 4. Check authoritative record provider (if identifier present)
        roll_no_field = extracted_fields.get("roll_number")
        identifier = roll_no_field.get("value") if roll_no_field else ""
        record_verification = self.provider.verify_record(identifier, doc_type, extracted_fields)

        return {
            "document_type": doc_type,
            "document_type_confidence": type_confidence,
            "status": consistency_result["status"],
            "summary": consistency_result["summary"],
            "checks": consistency_result.get("checks", []),
            "extracted_fields": extracted_fields,
            "record_verification": record_verification
        }
