"""
DocForensics AI — Authoritative Record Verification Interface
Defines modular provider interfaces for future authorized database / API record cross-referencing.
Strictly adheres to security protocols (no credential bypass, no web scraping, no session theft).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class VerificationProvider(ABC):
    """Abstract interface for external authoritative record verification."""

    @abstractmethod
    def verify_record(
        self,
        identifier: str,
        document_type: str,
        extracted_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Cross-references extracted document fields against an authorized source.

        Returns:
            - status: 'verified_match' | 'authoritative_mismatch' | 'record_not_found' | 'not_available'
            - source: Name or URI of provider
            - details: Field-by-field comparison details
        """
        pass


class MockVerificationProvider(VerificationProvider):
    """
    Default offline verification provider.
    Marks record verification as not available unless explicit reference data is provided.
    """

    def __init__(self, reference_db: Optional[Dict[str, Dict[str, Any]]] = None):
        self.reference_db = reference_db or {}

    def verify_record(
        self,
        identifier: str,
        document_type: str,
        extracted_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not identifier or identifier not in self.reference_db:
            return {
                "status": "not_available",
                "source": "Local System",
                "message": "No external authoritative database configured for this document."
            }

        official_record = self.reference_db[identifier]
        mismatches = []

        for k, official_val in official_record.items():
            extracted_entry = extracted_fields.get(k)
            extracted_val = extracted_entry.get("value") if isinstance(extracted_entry, dict) else extracted_entry

            if extracted_val is not None:
                if str(extracted_val).strip().lower() != str(official_val).strip().lower():
                    mismatches.append({
                        "field": k,
                        "displayed_value": extracted_val,
                        "official_value": official_val
                    })

        if mismatches:
            return {
                "status": "authoritative_mismatch",
                "source": "Authorized Database",
                "mismatches": mismatches,
                "message": f"Authoritative record mismatch detected across {len(mismatches)} field(s)."
            }

        return {
            "status": "verified_match",
            "source": "Authorized Database",
            "message": "Extracted fields match authoritative database record."
        }
